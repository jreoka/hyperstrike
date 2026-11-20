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
let loopTimer = null;

function ensureLoop() {
  if (loopTimer || matches.size === 0) return;
  loopTimer = setInterval(() => {
    const nowNs = process.hrtime.bigint();
    for (const sim of matches.values()) {
      sim.stepClock(nowNs);
    }
    if (matches.size === 0 && loopTimer) {
      clearInterval(loopTimer);
      loopTimer = null;
    }
  }, 4);
}

function stopLoopIfIdle() {
  if (matches.size === 0 && loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
  }
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
          onSnapshot: (to, spec) => parentPort.postMessage({ type: 'snap', id, to, spec }),
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
      if (sim) sim.setPaused(msg.p);
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
