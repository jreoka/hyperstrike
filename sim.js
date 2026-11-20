'use strict';
// ============================================================
// SERVER-AUTHORITATIVE SIMULATION
// The physics, arena, bot AI and power-up code is extracted from index.html at
// startup, so the server runs exactly the same simulation code as the client.
// Each match gets its own sandbox (PHYS constants such as gravity are mutated
// per match by mutators, so matches must not share globals).
// ============================================================
const fs = require('fs');
const vm = require('vm');
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
    slice(indexHtml, 'const V3 = THREE.Vector3', '/* ---------------- Procedural noise'),
    slice(indexHtml, 'function makeControls()', '\n'),
    slice(indexHtml, '// ===== 02_arena_shape.js =====', '// ===== 04_render.js ====='),
    slice(indexHtml, '// ===== 09_ai.js =====', '// ===== 10_world.js ====='),
    slice(indexHtml, '// ===== 10_world.js =====', 'const BALL_GLSL'),
    slice(indexHtml, '// ===== 11_modes.js =====', 'class ModeBase')
  ];
  return parts.join('\n') + `
;({ V3, QT, PHYS, ARENAS, World, Car, Ball, BotAI, PERSONALITIES, BOT_NAMES, PowerSystem,
    makeControls, KICKOFF_SPOTS, KICKOFF_SETS, yawToward, pick, clamp, ballIntegrate, HITBOXES, PI });`;
}

let compiled = null;
function loadSim(indexFile) {
  const html = fs.readFileSync(indexFile, 'utf8');
  compiled = new vm.Script(buildSimSource(html), { filename: 'hyperstrike-sim.js' });
  // Smoke test so a broken extraction fails at boot, not mid-match
  const S = createSandbox();
  const w = new S.World(S.ARENAS[0], { layout: 'default' });
  const c = w.addCar(new S.Car(0, 0, 'T', 0));
  c.bot = new S.BotAI(c, 1, 0);
  for (let i = 0; i < 240; i++) w.step(S.PHYS.dt);
}

function createSandbox() {
  if (!compiled) throw new Error('[SIM] loadSim() not called');
  const ctx = vm.createContext({ THREE, Math, console });
  return compiled.runInContext(ctx);
}

module.exports = { loadSim, createSandbox };
