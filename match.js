'use strict';
// ============================================================
// SERVER MATCH: authoritative 120Hz simulation of one multiplayer match.
// Clients send only their controls; the server simulates every car, the ball,
// bots, boost pads and power-ups, decides goals and the clock, and streams
// full-state binary snapshots (60Hz) that clients rewind to and replay their unacked
// inputs on top of. Mirrors MatchMode/OnlineMatchMode logic from index.html.
// ============================================================
const { createSandbox } = require('./sim');

const TICKS_PER_INPUT = 2;      // one input slot = 2 physics ticks (60Hz input & snapshots)
const MAX_AHEAD_SLOTS = 10;      // buffered beyond this (~166ms) -> skip ahead to bound input latency
const SKIP_TO_SLOTS = 3;
const MAX_FUTURE_SEQ = 240;      // reject inputs more than 4s ahead of the server timeline
const IDLE_CLEAR_SLOTS = 30;     // no input for 0.5s -> release all controls
const COUNTDOWN_S = 3.0;
const GOAL_PAUSE_S = 3.4;

const r3 = v => Number.isFinite(v) ? Math.round(v * 1000) / 1000 : 0;
const r4 = v => Number.isFinite(v) ? Math.round(v * 10000) / 10000 : 0;

// Controls arrive packed: [throttle, steer, pitch, yaw, roll, bits]
const clamp1 = v => (typeof v === 'number' && Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0);
function isValidPackedControls(arr) {
  if (!Array.isArray(arr) || arr.length !== 6) return false;
  for (let i = 0; i < 5; i++) {
    const v = arr[i];
    if (typeof v !== 'number' || !Number.isFinite(v) || v < -1.001 || v > 1.001) return false;
  }
  const bits = arr[5];
  return Number.isInteger(bits) && bits >= 0 && bits <= 31;
}
function unpackControls(arr, into) {
  into.throttle = clamp1(arr[0]); into.steer = clamp1(arr[1]); into.pitch = clamp1(arr[2]);
  into.yaw = clamp1(arr[3]); into.roll = clamp1(arr[4]);
  const bits = Number.isInteger(arr[5]) ? arr[5] : 0;
  into.jump = !!(bits & 1); into.boost = !!(bits & 2); into.handbrake = !!(bits & 4);
  into.airRollFree = !!(bits & 8); into.item = !!(bits & 16);
}
class ServerMatch {
  /**
   * @param {object} opts
   * @param {object} opts.settings  sanitized lobby settings (arena already resolved)
   * @param {Array}  opts.players   [{ userId, username, team, car }]
   * @param {function} opts.send    (userId, msg) => void
   * @param {function} opts.broadcast (msg) => void
   * @param {function} opts.onGoal  (info) => void
   * @param {function} opts.onEnd   ({ score, stats }) => void
   */
  constructor(opts) {
    this.opts = opts;
    const s = this.settings = opts.settings;
    const S = this.S = createSandbox();
    if (s.mutator === 'lowgrav') S.PHYS.g = 3.9;
    if (s.mutator === 'fastball') S.PHYS.ballMax = 78;
    const def = S.ARENAS[s.arena] || S.ARENAS[0];
    const w = this.world = new S.World(def, { layout: s.layout || 'default', scale: s.teamSize >= 4 ? 1.12 : 1 });
    w.recorder = { record() {}, clear() {}, snapshot() { return []; }, tail: null };
    if (s.boostMode === 'unlimited') w.unlimitedBoost = true;
    if (s.boostMode === 'recharge') w.boostRecharge = true;
    if (s.mode === 'mayhem') w.powers = new S.PowerSystem(w, { interval: s.interval || 10 });

    this.humans = new Map(); // userId -> input timeline state (see makeHuman)
    this.carInfo = [];       // index-aligned descriptors sent to clients
    let botIdx = 0;
    for (const team of [0, 1]) {
      const teamPlayers = opts.players.filter(p => p.team === team);
      for (const p of teamPlayers) {
        const body = p.car && Number.isInteger(p.car.body) ? p.car.body : 0;
        const car = this.addCar(team, p.username.toUpperCase(), body);
        car.netUserId = p.userId;
        car.explosion = p.car && Number.isInteger(p.car.explosion) ? p.car.explosion : 0;
        this.humans.set(p.userId, this.makeHuman(car, p.username));
        this.carInfo.push({ userId: p.userId, botId: null, team, name: car.name, car: p.car || null });
      }
      if (s.botFill) {
        for (let i = teamPlayers.length; i < s.teamSize; i++) {
          const name = S.BOT_NAMES[(botIdx++) % S.BOT_NAMES.length];
          const car = this.addCar(team, name, Math.floor(Math.random() * S.HITBOXES.length));
          const botId = `bot_${team}_${i}`;
          car.netBotId = botId;
          const diff = (opts.teammateDiff !== undefined && teamPlayers.length > 0) ? opts.teammateDiff : (s.botDiff || 1);
          car.bot = new S.BotAI(car, diff, botIdx % S.PERSONALITIES.length);
          car.explosion = 0;
          this.carInfo.push({ userId: null, botId, team, name, car: null, body: car.design });
        }
      }
    }

    this.score = [0, 0];
    this.timeLeft = s.duration || 300;
    this.unlimited = !s.duration;
    this.overtime = false; this.otTime = 0;
    this.timerRunning = false; this.matchOver = false;
    this.state = 'idle'; this.stateT = 0; this.timeScale = 1; this.simAcc = 0;
    this.slot = 0; this.events = [];
    this.lastShotCar = null; this.lastShotT = 0;
    this.finished = false;
    this.paused = false;
    this.createdAt = Date.now();
    this.timer = null;
    this._running = false;
    this._lastNs = 0n;
    this._acc = 0;
    this._slotSec = this.S.PHYS.dt * TICKS_PER_INPUT;
  }

  // Input timeline: every server slot consumes exactly one input sequence number, whether or not
  // that input has arrived. The client therefore always knows which of its inputs the
  // authoritative state includes (ack) and can replay the rest on top of it exactly.
  makeHuman(car, username) {
    return {
      car, username,
      buf: new Map(),        // seq -> packed controls, only seqs >= nextSeq
      nextSeq: 0,            // seq the next slot will consume (0 = timeline not started)
      applied: [0, 0, 0, 0, 0, 0], // packed controls used for the current slot
      ack: 0,                // seq consumed by the most recent slot
      depth: 0,              // buffered inputs at the moment the last slot consumed (incl. its own)
      starves: 0,            // slots in play where the input had not arrived yet
      idle: 0,
      edge: 0                // jump/item press bits from inputs that arrived too late
    };
  }

  addCar(team, name, body) {
    const car = new this.S.Car(this.world.cars.length, team, name, body);
    this.world.addCar(car);
    return car;
  }

  start(opts) {
    this.world.allowDemos = true;
    this.beginKickoff();
    this._running = true;
    this._lastNs = process.hrtime.bigint();
    this._acc = 0;
    if (opts && opts.managed) return;
    this.timer = setInterval(() => this.stepClock(process.hrtime.bigint()), 5);
    if (this.timer.unref) this.timer.unref();
  }

  stepClock(nowNs) {
    if (!this._running || this.finished) return;
    try {
      if (this.paused) { this._lastNs = nowNs; this._acc = 0; return; }
      this._acc += Number(nowNs - this._lastNs) / 1e9;
      this._lastNs = nowNs;
      let n = 0;
      let shouldSnap = false;
      const slotSec = this._slotSec;
      while (this._acc >= slotSec && n < 8 && !this.finished) {
        if (this.runSlot()) shouldSnap = true;
        this._acc -= slotSec;
        n++;
      }
      if (n >= 8) this._acc = 0; // server hiccup: don't spiral
      if (shouldSnap && !this.finished) this.sendSnapshots();
    } catch (err) {
      console.error('[SIM] Match simulation error:', err);
      this.stop();
      if (this.opts.onError) this.opts.onError(err);
    }
  }

  setPaused(p) {
    this.paused = !!p;
  }

  stop() {
    this.finished = true;
    this._running = false;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  // ---- inputs ----
  pushInput(userId, seq, packed) {
    const h = this.humans.get(userId);
    if (!h || !Number.isSafeInteger(seq) || seq <= 0 || !isValidPackedControls(packed)) return false;
    if (h.nextSeq === 0) h.nextSeq = seq; // first input starts this player's timeline
    if (seq < h.nextSeq) {
      // Arrived after its slot ran: keep button presses so taps are never lost
      h.edge |= packed[5] & 17;
      return true;
    }
    if (seq > h.nextSeq + MAX_FUTURE_SEQ) return true; // absurdly far ahead: ignore
    h.buf.set(seq, packed);
    return true;
  }

  consumeInput(h, inPlay) {
    if (h.nextSeq === 0) return; // no input yet
    let depth = h.buf.size;
    if (depth > MAX_AHEAD_SLOTS) {
      // Client is running too far ahead (burst after a stall): drop oldest to bound latency
      while (depth > SKIP_TO_SLOTS) {
        const old = h.buf.get(h.nextSeq);
        if (old) { h.edge |= old[5] & 17; h.buf.delete(h.nextSeq); depth--; }
        h.nextSeq++;
      }
    }
    h.depth = depth;
    const inp = h.buf.get(h.nextSeq);
    if (inp) {
      h.buf.delete(h.nextSeq);
      h.applied = inp; h.idle = 0;
    } else {
      // Not here in time: repeat the previous controls for this slot (and say so via `starves`)
      if (inPlay) h.starves++;
      if (++h.idle > IDLE_CLEAR_SLOTS) h.applied = [0, 0, 0, 0, 0, 0];
    }
    if (h.edge) { h.applied = h.applied.slice(); h.applied[5] |= h.edge; h.edge = 0; }
    h.ack = h.nextSeq;
    h.nextSeq++;
  }

  // A player who leaves keeps their car in the match, driven by a bot
  removePlayer(userId) {
    const h = this.humans.get(userId);
    if (!h) return;
    this.humans.delete(userId);
    h.car.bot = new this.S.BotAI(h.car, this.settings.botDiff || 1, 0);
  }

  // ---- main loop ----
  runSlot() {
    const inPlay = this.state === 'play';
    for (const h of this.humans.values()) this.consumeInput(h, inPlay);
    for (let i = 0; i < TICKS_PER_INPUT; i++) {
      // Re-apply the raw input every tick: power-ups (scramble) mutate car.controls in place
      for (const h of this.humans.values()) unpackControls(h.applied, h.car.controls);
      this.tick(this.S.PHYS.dt);
    }
    this.slot++;
    return !this.finished; // snapshot every slot (60Hz)
  }

  tick(dt) {
    this.preUpdate(dt);
    if (this.finished) return;
    if (this.state === 'ended') return;
    // Slow-motion during the goal celebration, like the client
    this.simAcc += this.timeScale;
    while (this.simAcc >= 1) {
      this.simAcc -= 1;
      const w = this.world;
      w.step(dt);
      this.collectEvents();
      if (this.state === 'play') {
        const gt = w.shape.inGoal(w.ball.pos, this.S.PHYS.ballR);
        if (gt >= 0) this.onGoal(gt);
      }
    }
  }

  setState(s) { this.state = s; this.stateT = 0; }

  preUpdate(dt) {
    this.stateT += dt;
    const w = this.world;
    if (this.state === 'countdown') {
      if (COUNTDOWN_S - this.stateT <= 0) { w.frozen = false; this.setState('play'); }
    } else if (this.state === 'play') {
      if (w.kickoff && (w.ball.lastTouch || this.stateT > 6)) w.kickoff = false;
      if (!this.timerRunning && (w.ball.lastTouch || this.stateT > 4)) this.timerRunning = true;
      if (this.timerRunning && !this.unlimited) {
        if (this.overtime) this.otTime += dt;
        else {
          this.timeLeft = Math.max(0, this.timeLeft - dt);
          if (this.timeLeft <= 0 && (w.ball.onGround || w.ball.pos.y < this.S.PHYS.ballR + 0.15)) this.timeUp();
        }
      }
    } else if (this.state === 'goal') {
      if (this.stateT < 1.3) {
        const t = Math.max(0, Math.min(1, (this.stateT - 0.5) / 0.8));
        this.timeScale = 0.22 + (1 - 0.22) * (t * t * (3 - 2 * t));
      } else this.timeScale = 1;
      if (this.stateT > GOAL_PAUSE_S) {
        this.timeScale = 1;
        if (this.matchOver) this.finish();
        else this.beginKickoff();
      }
    }
  }

  timeUp() {
    if (this.score[0] === this.score[1]) {
      this.overtime = true; this.otTime = 0;
      this.opts.broadcast({ type: 'match_overtime' });
      this.beginKickoff();
    } else {
      this.matchOver = true;
      this.finish();
    }
  }

  kickoffPositions() {
    const S = this.S, w = this.world, s = w.shape;
    const sx = s.W / 40.96, sz = s.L / 51.2;
    const set = S.pick(S.KICKOFF_SETS[Math.min(5, this.settings.teamSize)] || S.KICKOFF_SETS[1]);
    const order = set.slice().sort(() => Math.random() - 0.5);
    for (const team of [0, 1]) {
      const mem = w.cars.filter(c => c.team === team);
      mem.forEach((c, i) => {
        const spot = S.KICKOFF_SPOTS[order[i % order.length]] || S.KICKOFF_SPOTS[4];
        let x = spot[0] * sx, z = spot[1] * sz; if (team === 1) { x = -x; z = -z; }
        if (i >= order.length) x += (i - order.length + 1) * 4;
        c.place(x, 0.35, z, S.yawToward(x, z, 0, 0)); c.boost = 33; c.demolished = false; if (c.bot) c.bot.reset();
        c._kick = [r3(x), 0.35, r3(z), r4(S.yawToward(x, z, 0, 0))];
      });
    }
  }

  beginKickoff() {
    const w = this.world, S = this.S;
    w.ball.reset(0, S.PHYS.ballR, 0); w.ball.frozen = false;
    this.kickoffPositions();
    w.resetPads(); w.frozen = true; w.kickoff = true; w.brain.chooseKickers(w.cars, w.ball);
    if (w.powers) w.powers.reset();
    this.setState('countdown'); this.timerRunning = false; this.timeScale = 1; this.simAcc = 0;
    this.opts.broadcast({ type: 'match_kickoff', cars: w.cars.map(c => c._kick), overtime: this.overtime });
  }

  // Mirrors MatchMode.onTouch: credits shots and saves from hit events
  onTouch(hit) {
    if (this.state !== 'play') return;
    const car = hit.car, w = this.world;
    if (hit.save) { car.stats.saves++; car.stats.score += hit.epic ? 75 : 50; }
    if (hit.shot && (!this.lastShotCar || this.lastShotCar !== car || w.time - this.lastShotT > 2)) {
      car.stats.shots++; car.stats.score += 20; this.lastShotCar = car; this.lastShotT = w.time;
    }
    car.stats.score += 2;
  }

  collectEvents() {
    for (const e of this.world.events) {
      if (e.type === 'hit') this.onTouch(e);
      if (e.type === 'demo') e.attacker.stats.score += 25;
      if (e.type === 'demo' || e.type === 'power' || e.type === 'respawn') this.events.push(this.serializeEvent(e));
    }
  }

  // Replace object references with indices so events can be sent as JSON
  serializeEvent(e, depth = 0) {
    const out = {};
    for (const k of Object.keys(e)) {
      const v = e[k];
      if (v && v instanceof this.S.Car) out[k] = { $c: this.world.cars.indexOf(v) };
      else if (v && v === this.world.ball) out[k] = { $b: 1 };
      else if (v && v.isVector3) out[k] = { $v: [r3(v.x), r3(v.y), r3(v.z)] };
      else if (v === null || typeof v !== 'object') out[k] = typeof v === 'number' ? r3(v) : v;
      else if (depth < 1 && !Array.isArray(v)) out[k] = this.serializeEvent(v, depth + 1);
    }
    return out;
  }

  onGoal(goalTeam) {
    const w = this.world, S = this.S;
    const scoringTeam = 1 - goalTeam;
    this.score[scoringTeam]++;
    const hist = w.ball.touchHistory; let scorer = null, assist = null;
    for (let i = hist.length - 1; i >= 0; i--) {
      const h = hist[i];
      if (h.car.team === scoringTeam && w.time - h.time < 8) {
        scorer = h.car;
        for (let j = i - 1; j >= 0; j--) {
          const a = hist[j];
          if (a.car !== scorer && a.car.team === scoringTeam && h.time - a.time < 5) { assist = a.car; break; }
          if (a.car.team !== scoringTeam) break;
        }
        break;
      }
    }
    const own = !!(w.ball.lastTouch && w.ball.lastTouch.team === goalTeam);
    if (scorer) { scorer.stats.goals++; scorer.stats.score += 100; }
    if (assist) { assist.stats.assists++; assist.stats.score += 50; }
    const speed = Math.round(w.ball.vel.length() * 3.6);
    const pos = w.ball.pos.clone();
    // Goal blast: shove nearby cars away (same as the client)
    const tmp = new S.V3();
    for (const c of w.cars) {
      const d = c.pos.distanceTo(pos); if (d >= 26 || c.demolished) continue;
      const k = 1 - d / 26, kk = Math.sqrt(k);
      const dir = tmp.copy(c.pos).sub(pos); dir.y = Math.max(dir.y, 0) + d * 0.35;
      if (dir.lengthSq() < 1e-4) dir.set(0, 1, 0); dir.normalize();
      c.vel.addScaledVector(dir, 12 + 28 * kk); c.vel.y += 3 + 8 * kk;
      c.angVel.add(new S.V3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(9 * kk));
      c.blastT = 1.4; c.groundLock = Math.max(c.groundLock || 0, 0.3);
    }
    if (w.powers) w.powers.clearEffects();
    w.ball.frozen = true; w.ball.pos.set(0, -50, 0); w.ball.vel.set(0, 0, 0);
    this.setState('goal'); w.kickoff = false;
    if (this.overtime) this.matchOver = true;
    this.opts.onGoal({
      goalTeam, scoringTeam, score: this.score.slice(),
      scorerName: scorer ? scorer.name : '', assistName: assist ? assist.name : '',
      scorerId: scorer && scorer.netUserId ? scorer.netUserId : null,
      own: own && !scorer, speed, pos: [r3(pos.x), r3(pos.y), r3(pos.z)],
      explosion: scorer ? scorer.explosion || 0 : 0
    });
  }

  finish() {
    if (this.finished) return;
    this.setState('ended');
    this.world.frozen = true;
    this.sendSnapshots();
    this.stop();
    const stats = {};
    for (const c of this.world.cars) {
      if (c.netUserId) {
        const st = c.stats;
        stats[c.netUserId] = { goals: st.goals, assists: st.assists, saves: st.saves, shots: st.shots, demos: st.demos };
      }
    }
    this.opts.onEnd({ score: this.score.slice(), stats });
  }

  // ---- snapshots ----
  powersSnap() {
    const P = this.world.powers; if (!P) return undefined;
    const w = this.world, idx = c => w.cars.indexOf(c);
    const b = w.ball;
    return {
      c: w.cars.map(c => [c.spikeT || 0, c.magT || 0, c.ovT || 0, c.nitroT || 0, c.scrT || 0, c.tornT || 0, c.speedMul || 1, c.hookT || 0].map(r3)),
      b: [r3(b.iceT || 0), r3(b.antiT || 0), r3(b.gScale || 1), b.stuckTo ? idx(b.stuckTo) : -1],
      t: P.tornados.map(t => [r3(t.pos.x), r3(t.pos.y), r3(t.pos.z), r3(t.t), r3(t.life), idx(t.owner), t.id]),
      s: P.shots.map(s => ({ k: s.kind, f: idx(s.from), to: s.to === b ? -1 : idx(s.to), p: [r3(s.pos.x), r3(s.pos.y), r3(s.pos.z)], ph: s.phase, t: r3(s.t || 0), pt: r3(s.pt || 0) }))
    };
  }

  // Frozen world-steps left before kickoff releases, computed with the exact float ops
  // preUpdate uses, so the client can unfreeze its prediction on the very same tick.
  countdownTicks() {
    if (this.state !== 'countdown') return null;
    const dt = this.S.PHYS.dt;
    let st = this.stateT, n = 0;
    for (;;) { st += dt; if (COUNTDOWN_S - st <= 0 || n > 4000) break; n++; }
    return n;
  }

  sendSnapshots() {
    const S = this.S, w = this.world;
    const ctls = w.cars.map(c => {
      if (c.netUserId) { const h = this.humans.get(c.netUserId); if (h) return h.applied; }
      return S.netPackControls(c.controls);
    });
    const meta = {};
    if (w.powers) { meta.pw = this.powersSnap(); meta.it = w.cars.map(c => c.item || null); meta.inx = w.cars.map(c => Math.round((c.itemNext || 0) * 10) / 10); }
    if (this.events.length) meta.ev = this.events.splice(0);
    if (this.slot % 30 === 0 || this.state === 'ended') meta.cs = w.cars.map(c => [c.stats.score, c.stats.goals, c.stats.assists, c.stats.saves, c.stats.shots, c.stats.demos]);
    const shared = S.netEncodeSnapshot(w, {
      state: this.state, overtime: this.overtime, cd: this.countdownTicks(),
      timeLeft: this.timeLeft, otTime: this.otTime, score: this.score, ctls,
      meta: Object.keys(meta).length ? JSON.stringify(meta) : null
    });
    const to = [];
    for (const [userId, h] of this.humans) {
      to.push([userId, S.netPatchSnapshot(shared.slice(0), h.ack, h.depth, h.starves)]);
    }
    if (this.opts.onSnapshot) this.opts.onSnapshot(to, shared);
    else {
      for (const [userId, buf] of to) this.opts.sendRaw(userId, buf);
      if (this.opts.spectators) this.opts.spectators(shared);
    }
  }

  // Milliseconds until this match's next slot is due (for the worker's precise scheduler)
  msUntilNextSlot(nowNs) {
    if (!this._running || this.finished) return Infinity;
    if (this.paused) return 50;
    const elapsed = Number(nowNs - this._lastNs) / 1e9;
    return Math.max(0, (this._slotSec - this._acc - elapsed) * 1000);
  }
}

module.exports = { ServerMatch, unpackControls, isValidPackedControls };
