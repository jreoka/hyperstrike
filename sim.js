'use strict';
// ============================================================
// SERVER-AUTHORITATIVE SIMULATION
// The physics, arena, bot AI and power-up code is extracted from index.html at
// startup, so the server runs exactly the same simulation code as the client.
// Compiled as a native V8 closure factory (no vm.Context cross-realm penalty)
// so each match gets its own isolated PHYS/World/Car scope at full JIT speed.
// ============================================================
const fs = require('fs');
const THREE = require('three');

function slice(src, startMarker, endMarker) {
  const a = src.indexOf(startMarker);
  if (a < 0) throw new Error(`[SIM] marker not found: ${startMarker}`);
  const b = endMarker ? src.indexOf(endMarker, a + startMarker.length) : src.length;
  if (b < 0) throw new Error(`[SIM] marker not found: ${endMarker}`);
  return src.slice(a, b);
}

function buildSimSource(indexHtml) {
  const parts = [
    '"use strict";',
    slice(indexHtml, 'const V3 = THREE.Vector3', '/* ---------------- Procedural noise'),
    'const V3Up = new V3(0, 1, 0);',
    slice(indexHtml, 'function makeControls()', '\n'),
    slice(indexHtml, '// ===== 02_arena_shape.js =====', '// ===== 04_render.js ====='),
    slice(indexHtml, '// ===== 09_ai.js =====', '// ===== 10_world.js ====='),
    slice(indexHtml, '// ===== 10_world.js =====', 'const BALL_GLSL'),
    slice(indexHtml, '// ===== 11_modes.js =====', 'class ModeBase')
  ];
  return parts.join('\n') + `
return { V3, QT, PHYS, ARENAS, World, Car, Ball, BotAI, PERSONALITIES, BOT_NAMES, PowerSystem,
  makeControls, KICKOFF_SPOTS, KICKOFF_SETS, yawToward, pick, clamp, ballIntegrate, HITBOXES, PI };`;
}

let factory = null;
function loadSim(indexFile, opts = {}) {
  const { smokeTest = true } = opts;
  const html = fs.readFileSync(indexFile, 'utf8');
  factory = new Function('THREE', buildSimSource(html));
  if (!smokeTest) return;
  // Smoke test so a broken extraction (including car-car collisions) fails at boot, not mid-match
  const S = createSandbox();
  const w = new S.World(S.ARENAS[0], { layout: 'default' });
  w.recorder = { record() {}, clear() {}, snapshot() { return []; }, tail: null };
  const c1 = w.addCar(new S.Car(0, 0, 'T1', 0));
  const c2 = w.addCar(new S.Car(1, 1, 'T2', 0));
  c1.place(-1.2, 0.35, 0, 0); c1.vel.set(20, 0, 0);
  c2.place(1.2, 0.35, 0, S.PI); c2.vel.set(-20, 0, 0);
  c1.bot = new S.BotAI(c1, 1, 0);
  c2.bot = new S.BotAI(c2, 1, 1);
  for (let i = 0; i < 240; i++) w.step(S.PHYS.dt);
}

function createSandbox() {
  if (!factory) throw new Error('[SIM] loadSim() not called');
  return factory(THREE);
}

module.exports = { loadSim, createSandbox };
