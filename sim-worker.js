'use strict';
// ============================================================
// SIMULATION WORKER THREAD
// Runs authoritative 120Hz ServerMatch instances on its own V8 isolate / CPU core.
// Receives player controls & lifecycle commands from the main thread and posts
// back batched snapshots, broadcasts, goals and match-end results.
// ============================================================
const { parentPort, workerData } = require('worker_threads');
const { loadSim } = require('./sim');
const { ServerMatch } = require('./match');

if (!parentPort) {
  throw new Error('[SIM-WORKER] Must be started as a worker_threads Worker');
}

const { indexFile, workerIndex, smokeTest } = workerData;
loadSim(indexFile, { smokeTest });

const matches = new Map(); // matchId -> ServerMatch
// Precise slot scheduler: sleep with a timer until ~1.5ms before the earliest due slot,
// then yield with setImmediate (which still services incoming input messages) until it is due.
// The old fixed 4ms interval added 0-4ms of jitter to every snapshot.
let loopTimer = null;
let loopImmediate = null;

function runLoop() {
  loopTimer = null; loopImmediate = null;
  if (matches.size === 0) return;
  const nowNs = process.hrtime.bigint();
  let wait = Infinity;
  for (const sim of matches.values()) {
    sim.stepClock(nowNs);
    const ms = sim.msUntilNextSlot(nowNs);
    if (ms < wait) wait = ms;
  }
  if (matches.size === 0) return;
  if (!Number.isFinite(wait)) wait = 50;
  if (wait > 2.5) loopTimer = setTimeout(runLoop, Math.floor(wait - 1.5));
  else loopImmediate = setImmediate(runLoop);
}

function ensureLoop() {
  if (loopTimer) { clearTimeout(loopTimer); loopTimer = null; }
  if (!loopImmediate) loopImmediate = setImmediate(runLoop);
}

function stopLoopIfIdle() {
  if (matches.size !== 0) return;
  if (loopTimer) { clearTimeout(loopTimer); loopTimer = null; }
  if (loopImmediate) { clearImmediate(loopImmediate); loopImmediate = null; }
}

parentPort.on('message', (msg) => {
  if (!msg || typeof msg.type !== 'string') return;
  const id = msg.id;

  switch (msg.type) {
    case 'create': {
      try {
        const sim = new ServerMatch({
          settings: msg.settings,
          teammateDiff: msg.teammateDiff,
          players: msg.players,
          broadcast: (m) => parentPort.postMessage({ type: 'broadcast', id, msg: m }),
          onSnapshot: (to, spec) => {
            const transfer = [spec];
            for (const pair of to) transfer.push(pair[1]);
            parentPort.postMessage({ type: 'snap', id, to, spec }, transfer);
          },
          onGoal: (info) => parentPort.postMessage({ type: 'goal', id, info }),
          onEnd: ({ score, stats }) => {
            matches.delete(id);
            stopLoopIfIdle();
            parentPort.postMessage({ type: 'end', id, score, stats });
          },
          onError: (err) => {
            matches.delete(id);
            stopLoopIfIdle();
            parentPort.postMessage({ type: 'error', id, message: err && err.message ? err.message : 'Simulation error' });
          }
        });
        matches.set(id, sim);
        parentPort.postMessage({ type: 'created', id, carInfo: sim.carInfo });
      } catch (err) {
        parentPort.postMessage({
          type: 'create_error',
          id,
          message: err && err.message ? err.message : 'Failed to create match'
        });
      }
      break;
    }

    case 'start': {
      const sim = matches.get(id);
      if (sim) {
        sim.start({ managed: true });
        ensureLoop();
      }
      break;
    }

    case 'in': {
      const sim = matches.get(id);
      if (sim) sim.pushInput(msg.u, msg.s, msg.c);
      break;
    }

    case 'removePlayer': {
      const sim = matches.get(id);
      if (sim) sim.removePlayer(msg.u);
      break;
    }

    case 'pause': {
      const sim = matches.get(id);
      if (sim) { sim.setPaused(msg.p); ensureLoop(); }
      break;
    }

    case 'stop': {
      const sim = matches.get(id);
      if (sim) {
        matches.delete(id);
        sim.stop();
        stopLoopIfIdle();
      }
      break;
    }
  }
});

parentPort.postMessage({ type: 'ready', workerIndex });
