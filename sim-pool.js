'use strict';
// ============================================================
// MULTICORE SIMULATION WORKER POOL
// Distributes ServerMatch instances across Node.js worker_threads (one per CPU
// core by default, or SIM_WORKERS env) using least-loaded scheduling.
// Exposes a MatchHandle with the same runtime interface as ServerMatch.
// ============================================================
const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');

const WORKER_SCRIPT = path.join(__dirname, 'sim-worker.js');

// Keep validation local so the main thread never needs to load three.js / sim.js
function isValidPackedControls(arr) {
  if (!Array.isArray(arr) || arr.length !== 6) return false;
  for (let i = 0; i < 5; i++) {
    const v = arr[i];
    if (typeof v !== 'number' || !Number.isFinite(v) || v < -1.001 || v > 1.001) return false;
  }
  const bits = arr[5];
  return Number.isInteger(bits) && bits >= 0 && bits <= 31;
}

function detectWorkerCount() {
  const envRaw = parseInt(process.env.SIM_WORKERS || '', 10);
  if (Number.isInteger(envRaw) && envRaw >= 1) return Math.min(envRaw, 64);
  const cores = typeof os.availableParallelism === 'function'
    ? os.availableParallelism()
    : (os.cpus() ? os.cpus().length : 1);
  return Math.max(1, cores || 1);
}

class MatchHandle {
  constructor(pool, workerSlot, id, opts, weight) {
    this._pool = pool;
    this._slot = workerSlot;
    this.id = id;
    this.opts = opts;
    this._weight = weight;
    this.carInfo = [];
    this.createdAt = Date.now();
    this.stopped = false;
    // Mirror human map on main thread for synchronous anti-cheat validation of pushInput
    this._humans = new Map();
    for (const p of (opts.players || [])) {
      this._humans.set(p.userId, { lastQueuedSeq: 0 });
    }
  }

  start() {
    if (this.stopped) return;
    this._slot.worker.postMessage({ type: 'start', id: this.id });
  }

  // Returns false only for structurally invalid packets (the anti-cheat strikes on those).
  // Stale/duplicate sequence numbers, packets for a stopped match or a removed player are
  // normal side effects of restarts, reconnects and match transitions: drop them silently.
  pushInput(userId, seq, packed) {
    if (!Number.isSafeInteger(seq) || seq <= 0 || !isValidPackedControls(packed)) return false;
    if (this.stopped) return true;
    const h = this._humans.get(userId);
    if (!h || seq <= h.lastQueuedSeq) return true;
    h.lastQueuedSeq = seq;
    this._slot.worker.postMessage({ type: 'in', id: this.id, u: userId, s: seq, c: packed });
    return true;
  }

  removePlayer(userId) {
    this._humans.delete(userId);
    if (this.stopped) return;
    this._slot.worker.postMessage({ type: 'removePlayer', id: this.id, u: userId });
  }

  setPaused(paused) {
    if (this.stopped) return;
    this._slot.worker.postMessage({ type: 'pause', id: this.id, p: !!paused });
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this._pool._releaseMatch(this._slot, this.id);
    if (this._slot.worker) {
      try { this._slot.worker.postMessage({ type: 'stop', id: this.id }); } catch (_) {}
    }
  }
}

class SimPool {
  constructor(workerCount = detectWorkerCount()) {
    this.workerCount = Math.max(1, workerCount);
    this.indexFile = null;
    this.slots = [];
    this.pendingCreates = new Map(); // matchId -> { resolve, reject, handle, slot }
    this.nextMatchId = 1;
    this.shuttingDown = false;
  }

  init(indexFile) {
    this.indexFile = indexFile;
    const readyPromises = [];
    for (let i = 0; i < this.workerCount; i++) {
      readyPromises.push(this._spawnWorker(i, i === 0));
    }
    return Promise.all(readyPromises);
  }

  _spawnWorker(index, smokeTest) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const slot = this.slots[index] || {
        index,
        worker: null,
        ready: false,
        matches: new Map(), // matchId -> MatchHandle
        weight: 0
      };
      this.slots[index] = slot;

      const worker = new Worker(WORKER_SCRIPT, {
        workerData: { indexFile: this.indexFile, workerIndex: index, smokeTest }
      });
      slot.worker = worker;
      slot.ready = false;

      worker.on('message', (msg) => {
        if (!msg || typeof msg.type !== 'string') return;
        if (msg.type === 'ready') {
          slot.ready = true;
          if (!settled) { settled = true; resolve(); }
          return;
        }
        this._handleWorkerMessage(slot, msg);
      });

      worker.on('error', (err) => {
        console.error(`[SIM-POOL] Worker #${index} error:`, err);
        if (!settled) { settled = true; reject(err); }
      });

      worker.on('exit', (code) => {
        const wasReady = slot.ready;
        slot.ready = false;
        if (!settled) {
          settled = true;
          reject(new Error(`[SIM-POOL] Worker #${index} exited during startup with code ${code}`));
        }
        this._failSlotMatches(slot, `Simulation worker #${index} exited (code ${code})`);
        if (!this.shuttingDown && wasReady) {
          console.warn(`[SIM-POOL] Respawning worker #${index}...`);
          this._spawnWorker(index, false).catch((err) => {
            console.error(`[SIM-POOL] Failed to respawn worker #${index}:`, err);
          });
        }
      });
    });
  }

  _handleWorkerMessage(slot, msg) {
    const id = msg.id;
    switch (msg.type) {
      case 'created': {
        const pending = this.pendingCreates.get(id);
        if (!pending) {
          // Match was cancelled before creation finished
          try { slot.worker.postMessage({ type: 'stop', id }); } catch (_) {}
          return;
        }
        this.pendingCreates.delete(id);
        pending.handle.carInfo = msg.carInfo;
        pending.resolve(pending.handle);
        break;
      }

      case 'create_error': {
        const pending = this.pendingCreates.get(id);
        if (pending) {
          this.pendingCreates.delete(id);
          this._releaseMatch(slot, id);
          pending.reject(new Error(msg.message || 'Failed to create match'));
        }
        break;
      }

      case 'snap': {
        const handle = slot.matches.get(id);
        if (!handle || handle.stopped) return;
        const { sendRaw, spectators } = handle.opts;
        if (sendRaw && Array.isArray(msg.to)) {
          for (let i = 0; i < msg.to.length; i++) {
            const pair = msg.to[i];
            sendRaw(pair[0], pair[1]);
          }
        }
        if (spectators && msg.spec) spectators(msg.spec);
        break;
      }

      case 'broadcast': {
        const handle = slot.matches.get(id);
        if (!handle || handle.stopped) return;
        if (handle.opts.broadcast) handle.opts.broadcast(msg.msg);
        break;
      }

      case 'goal': {
        const handle = slot.matches.get(id);
        if (!handle || handle.stopped) return;
        if (handle.opts.onGoal) handle.opts.onGoal(msg.info);
        break;
      }

      case 'end': {
        const handle = slot.matches.get(id);
        if (!handle || handle.stopped) return;
        handle.stopped = true;
        this._releaseMatch(slot, id);
        if (handle.opts.onEnd) handle.opts.onEnd({ score: msg.score, stats: msg.stats });
        break;
      }

      case 'error': {
        const handle = slot.matches.get(id);
        if (!handle || handle.stopped) return;
        handle.stopped = true;
        this._releaseMatch(slot, id);
        if (handle.opts.onError) handle.opts.onError(new Error(msg.message || 'Simulation error'));
        break;
      }
    }
  }

  _pickSlot() {
    let best = null;
    for (const slot of this.slots) {
      if (!slot || !slot.ready) continue;
      if (
        !best ||
        slot.weight < best.weight ||
        (slot.weight === best.weight && slot.matches.size < best.matches.size)
      ) {
        best = slot;
      }
    }
    return best;
  }

  _releaseMatch(slot, id) {
    const handle = slot.matches.get(id);
    if (!handle) return;
    slot.matches.delete(id);
    slot.weight = Math.max(0, slot.weight - handle._weight);
  }

  _failSlotMatches(slot, reason) {
    for (const [id, pending] of this.pendingCreates.entries()) {
      if (pending.slot === slot) {
        this.pendingCreates.delete(id);
        pending.reject(new Error(reason));
      }
    }
    const active = [...slot.matches.values()];
    slot.matches.clear();
    slot.weight = 0;
    for (const handle of active) {
      if (handle.stopped) continue;
      handle.stopped = true;
      if (handle.opts.onError) handle.opts.onError(new Error(reason));
    }
  }

  createMatch(opts) {
    const slot = this._pickSlot();
    if (!slot) {
      return Promise.reject(new Error('No simulation workers available'));
    }
    const id = this.nextMatchId++;
    const teamSize = (opts.settings && opts.settings.teamSize) || 2;
    const weight = Math.max(2, teamSize * 2);
    const handle = new MatchHandle(this, slot, id, opts, weight);
    slot.matches.set(id, handle);
    slot.weight += weight;

    return new Promise((resolve, reject) => {
      this.pendingCreates.set(id, { resolve, reject, handle, slot });
      slot.worker.postMessage({
        type: 'create',
        id,
        settings: opts.settings,
        teammateDiff: opts.teammateDiff,
        players: opts.players
      });
    });
  }

  getStats() {
    let readyWorkers = 0;
    let activeMatches = 0;
    const perWorker = this.slots.map((s) => {
      if (s && s.ready) readyWorkers++;
      const count = s ? s.matches.size : 0;
      activeMatches += count;
      return count;
    });
    return {
      workers: this.workerCount,
      readyWorkers,
      activeMatches,
      perWorker
    };
  }

  async shutdown() {
    this.shuttingDown = true;
    const terminations = [];
    for (const slot of this.slots) {
      if (slot && slot.worker) {
        terminations.push(slot.worker.terminate().catch(() => {}));
      }
    }
    await Promise.all(terminations);
  }
}

module.exports = { SimPool, detectWorkerCount };
