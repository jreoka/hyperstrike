'use strict';
// ============================================================
// SERVER MATCH: authoritative 120Hz simulation of one multiplayer match.
// Clients send only their controls; the server simulates every car, the ball,
// bots, boost pads and power-ups, decides goals and the clock, and streams
// snapshots back. Mirrors MatchMode/OnlineMatchMode logic from index.html.
// ============================================================
const { createSandbox } = require('./sim');

const TICKS_PER_INPUT = 2;     // one input packet covers 2 physics ticks (60Hz)
const SLOTS_PER_SNAPSHOT = 2;  // snapshot every 2 input slots (30Hz)
const MAX_INPUT_QUEUE = 6;     // ~100ms of buffered input max; older input is dropped
const COUNTDOWN_S = 3.0;
const GOAL_PAUSE_S = 3.4;

// Car fields that are identity/visual/derived and never part of physics state
const CAR_SKIP = new Set([
  'id', 'team', 'name', 'design', 'half', 'inertia', 'controls', 'basis', 'bot', 'stats', 'events',
  'samplePts', 'bumpCooldown', 'netTarget', 'prevPos', 'prevQuat', 'isRemote', 'isPlayer', 'netUserId',
  'netBotId', 'explosion', 'wheels', 'lastBallTouchTick', 'visErr'
]);

const r3 = v => Number.isFinite(v) ? Math.round(v * 1000) / 1000 : 0;
const r4 = v => Number.isFinite(v) ? Math.round(v * 10000) / 10000 : 0;
const r5 = v => Number.isFinite(v) ? Math.round(v * 100000) / 100000 : 0;

// Full physics state of one car, used by its owner to reconcile prediction
function serializeCar(car) {
  const out = {};
  for (const k in car) {
    if (CAR_SKIP.has(k)) continue;
    const v = car[k];
    if (typeof v === 'number') out[k] = Number.isFinite(v) ? r4(v) : 0;
    else if (typeof v === 'boolean' || typeof v === 'string' || v === null) out[k] = v;
    else if (v && v.isVector3) out[k] = [r4(v.x), r4(v.y), r4(v.z)];
    else if (v && v.isQuaternion) out[k] = [r5(v.x), r5(v.y), r5(v.z), r5(v.w)];
  }
  const wheels = car.wheels;
  const wOut = new Array(wheels.length);
  for (let i = 0; i < wheels.length; i++) {
    const w = wheels[i];
    wOut[i] = [w.contact ? 1 : 0, r4(w.len), r4(w.prevLen), r3(w.spin), r4(w.steer), r4(w.normal.x), r4(w.normal.y), r4(w.normal.z)];
  }
  out.wheels = wOut;
  return out;
}

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
function clearControls(into) {
  into.throttle = 0; into.steer = 0; into.pitch = 0; into.yaw = 0; into.roll = 0;
  into.jump = false; into.boost = false; into.handbrake = false; into.airRollFree = false; into.item = false;
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

    this.humans = new Map(); // userId -> { car, queue, lastSeq, username, explosion }
    this.carInfo = [];       // index-aligned descriptors sent to clients
    let botIdx = 0;
    for (const team of [0, 1]) {
      const teamPlayers = opts.players.filter(p => p.team === team);
      for (const p of teamPlayers) {
        const body = p.car && Number.isInteger(p.car.body) ? p.car.body : 0;
        const car = this.addCar(team, p.username.toUpperCase(), body);
        car.netUserId = p.userId;
        car.explosion = p.car && Number.isInteger(p.car.explosion) ? p.car.explosion : 0;
        this.humans.set(p.userId, { car, queue: [], lastSeq: 0, lastQueuedSeq: 0, idleSlots: 0, username: p.username });
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
      const slotSec = this._slotSec;
      while (this._acc >= slotSec && n < 8 && !this.finished) {
        this.runSlot();
        this._acc -= slotSec;
        n++;
      }
      if (n >= 8) this._acc = 0; // server hiccup: don't spiral
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
    if (!h || !Number.isSafeInteger(seq) || seq <= (h.lastQueuedSeq || 0) || !isValidPackedControls(packed)) return false;
    h.lastQueuedSeq = seq;
    h.queue.push({ seq, c: packed });
    if (h.queue.length > MAX_INPUT_QUEUE) h.queue.splice(0, h.queue.length - MAX_INPUT_QUEUE);
    return true;
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
    for (const h of this.humans.values()) {
      const inp = h.queue.shift();
      if (inp) { unpackControls(inp.c, h.car.controls); h.lastSeq = inp.seq; h.idleSlots = 0; }
      else if (++h.idleSlots > 30) clearControls(h.car.controls);
    }
    for (let i = 0; i < TICKS_PER_INPUT; i++) this.tick(this.S.PHYS.dt);
    this.slot++;
    if (this.slot % SLOTS_PER_SNAPSHOT === 0 && !this.finished) this.sendSnapshots();
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
  carSnap(c) {
    const flags = (c.demolished ? 1 : 0) | (c.isBoosting ? 2 : 0) | (c.supersonic ? 4 : 0) | (c.sliding > 0.5 ? 8 : 0) | (c.onGround ? 16 : 0);
    return [r3(c.pos.x), r3(c.pos.y), r3(c.pos.z), r3(c.vel.x), r3(c.vel.y), r3(c.vel.z),
      r4(c.quat.x), r4(c.quat.y), r4(c.quat.z), r4(c.quat.w),
      r3(c.angVel.x), r3(c.angVel.y), r3(c.angVel.z), Math.round(c.boost * 10) / 10, flags, c.item || null];
  }

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

  sendSnapshots() {
    const w = this.world, b = w.ball;
    let pads = '';
    for (const p of w.pads) pads += p.active ? '1' : '0';
    const base = {
      type: 'snap',
      k: w.tick,
      st: this.state,
      tl: r3(this.timeLeft), ot: this.overtime, ott: r3(this.otTime),
      sc: this.score,
      fz: w.frozen,
      b: [r3(b.pos.x), r3(b.pos.y), r3(b.pos.z), r3(b.vel.x), r3(b.vel.y), r3(b.vel.z),
        r4(b.quat.x), r4(b.quat.y), r4(b.quat.z), r4(b.quat.w), r3(b.angVel.x), r3(b.angVel.y), r3(b.angVel.z), b.frozen ? 1 : 0],
      c: w.cars.map(c => this.carSnap(c)),
      pd: pads
    };
    if (w.powers) base.pw = this.powersSnap();
    if (this.events.length) base.ev = this.events.splice(0);
    if (this.slot % 30 === 0 || this.state === 'ended') base.cs = w.cars.map(c => [c.stats.score, c.stats.goals, c.stats.assists, c.stats.saves, c.stats.shots, c.stats.demos]);
    const shared = JSON.stringify(base).slice(0, -1); // reuse common part, append per-player fields
    if (this.opts.onSnapshot) {
      const to = [];
      for (const [userId, h] of this.humans) {
        to.push([userId, `${shared},"a":${h.lastSeq},"me":${JSON.stringify(serializeCar(h.car))}}`]);
      }
      this.opts.onSnapshot(to, shared + '}');
    } else {
      for (const [userId, h] of this.humans) {
        this.opts.sendRaw(userId, `${shared},"a":${h.lastSeq},"me":${JSON.stringify(serializeCar(h.car))}}`);
      }
      if (this.opts.spectators) this.opts.spectators(shared + '}');
    }
  }
}

module.exports = { ServerMatch, serializeCar, unpackControls, isValidPackedControls };
