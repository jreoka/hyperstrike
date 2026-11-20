'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { WebSocketServer } = require('ws');
const { SimPool } = require('./sim-pool');

const PORT = parseInt(process.env.PORT || '8080', 10);
const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, 'data');
const DB_FILE = path.join(DATA_DIR, 'db.json');
const INDEX_FILE = path.join(__dirname, 'index.html');
let cachedIndexHtml = null;
const simPool = new SimPool();
const REGION = process.env.FLY_REGION || 'local';
const ALLOWED_URL_RAW = (process.env.URL || '').trim();
let ALLOWED_ORIGIN = '';
let ALLOWED_HOST = '';
if (ALLOWED_URL_RAW) {
  try {
    const u = new URL(ALLOWED_URL_RAW);
    ALLOWED_ORIGIN = u.origin.toLowerCase();
    ALLOWED_HOST = u.host.toLowerCase();
  } catch (err) {
    console.error('[CONFIG] Invalid URL env:', ALLOWED_URL_RAW);
  }
}

function requestHost(req) {
  return String(req.headers['x-forwarded-host'] || req.headers.host || '').split(',')[0].trim().toLowerCase();
}

// The origin the game page must be loaded from. Uses the URL env when set,
// otherwise the server's own scheme://host (so local runs are protected too).
function expectedOrigin(req) {
  if (ALLOWED_ORIGIN) return ALLOWED_ORIGIN;
  const host = requestHost(req);
  if (!host) return '';
  const proto = String(req.headers['x-forwarded-proto'] || (req.socket && req.socket.encrypted ? 'https' : 'http'))
    .split(',')[0].trim().toLowerCase();
  return `${proto}://${host}`;
}

function isRequestAllowed(req) {
  const origin = String(req.headers.origin || '').trim().toLowerCase();
  // Browsers always send Origin on WebSocket upgrades and cross-origin fetches.
  // A page opened from disk sends "null"; a page on another host sends that host.
  if (origin) return origin === expectedOrigin(req);
  if (!ALLOWED_ORIGIN) return true;
  const host = requestHost(req);
  const referer = String(req.headers.referer || '').trim().toLowerCase();
  if (host === ALLOWED_HOST) return true;
  if (referer && (referer === ALLOWED_ORIGIN || referer.startsWith(ALLOWED_ORIGIN + '/'))) return true;
  return false;
}

function corsOriginHeader() {
  return ALLOWED_ORIGIN;
}

// ============================================================
// CLIENT PAGE TOKEN
// The server sets a signed, HttpOnly, SameSite=Strict cookie whenever it
// serves index.html. The WebSocket and /api/* endpoints require it, so a
// game client must have been loaded from this server. Local copies of the
// HTML (file://, localhost, other sites) never receive or send this cookie.
// ============================================================
const CLIENT_COOKIE = 'hs_client';
const CLIENT_COOKIE_TTL_MS = 24 * 60 * 60 * 1000;
const CLIENT_SECRET = (() => {
  if (process.env.CLIENT_SECRET) return process.env.CLIENT_SECRET;
  const file = path.join(DATA_DIR, 'client_secret');
  try {
    const existing = fs.readFileSync(file, 'utf8').trim();
    if (existing.length >= 32) return existing;
  } catch (_) {}
  const fresh = crypto.randomBytes(32).toString('hex');
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(file, fresh, { mode: 0o600 });
  } catch (err) {
    console.error('[CONFIG] Could not persist client secret:', err.message);
  }
  return fresh;
})();

function signClientToken(ts, nonce) {
  return crypto.createHmac('sha256', CLIENT_SECRET).update(`${ts}.${nonce}`).digest('base64url');
}

function issueClientCookie(req) {
  const ts = Date.now().toString(36);
  const nonce = crypto.randomBytes(12).toString('base64url');
  const value = `${ts}.${nonce}.${signClientToken(ts, nonce)}`;
  const secure = expectedOrigin(req).startsWith('https:') ? '; Secure' : '';
  return `${CLIENT_COOKIE}=${value}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${Math.floor(CLIENT_COOKIE_TTL_MS / 1000)}${secure}`;
}

function readCookie(req, name) {
  const header = String(req.headers.cookie || '');
  for (const part of header.split(';')) {
    const idx = part.indexOf('=');
    if (idx < 0) continue;
    if (part.slice(0, idx).trim() === name) return part.slice(idx + 1).trim();
  }
  return '';
}

function hasValidClientCookie(req) {
  const value = readCookie(req, CLIENT_COOKIE);
  const parts = value.split('.');
  if (parts.length !== 3) return false;
  const [ts, nonce, sig] = parts;
  const issued = parseInt(ts, 36);
  if (!Number.isFinite(issued) || Date.now() - issued > CLIENT_COOKIE_TTL_MS || issued - Date.now() > 60000) return false;
  const expected = Buffer.from(signClientToken(ts, nonce));
  const given = Buffer.from(sig);
  return expected.length === given.length && crypto.timingSafeEqual(expected, given);
}

// ============================================================
// PROGRESSION: INFINITE LEVELING & RANKS
// ============================================================
function xpForLevel(level) {
  const L = Math.max(1, Math.floor(level));
  return Math.floor(250 + (L - 1) * 140 + 20 * Math.pow(L - 1, 1.35));
}

function addXp(user, amount) {
  const gain = Math.max(0, Math.floor(amount));
  const oldLevel = user.level || 1;
  user.xp = (user.xp || 0) + gain;
  user.totalXp = (user.totalXp || 0) + gain;
  user.level = oldLevel;
  let safety = 0;
  while (user.xp >= xpForLevel(user.level) && safety < 500) {
    user.xp -= xpForLevel(user.level);
    user.level += 1;
    safety++;
  }
  return {
    xpGained: gain,
    oldLevel,
    newLevel: user.level,
    leveledUp: user.level > oldLevel
  };
}

const RANK_TIERS = [
  { name: 'BRONZE', min: 0, color: '#cd7f32' },
  { name: 'SILVER', min: 300, color: '#c4d0e0' },
  { name: 'GOLD', min: 600, color: '#ffd23f' },
  { name: 'PLATINUM', min: 900, color: '#4ff2e8' },
  { name: 'DIAMOND', min: 1200, color: '#4d9fff' },
  { name: 'CHAMPION', min: 1500, color: '#c06bff' },
  { name: 'GRAND CHAMPION', min: 1800, color: '#ff4f7a' }
];

const ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'];
function toRoman(n) {
  if (n <= 10) return ROMAN[n] || String(n);
  return String(n);
}

function computeRank(rpRaw) {
  const rp = Math.max(0, Math.floor(rpRaw || 0));
  if (rp >= 2100) {
    const over = rp - 2100;
    const legendTier = Math.floor(over / 150) + 1;
    const base = 2100 + (legendTier - 1) * 150;
    const nextRp = base + 150;
    return {
      tier: 'HYPERSTRIKE LEGEND',
      division: toRoman(legendTier),
      name: `LEGEND ${toRoman(legendTier)}`,
      short: `LEG ${toRoman(legendTier)}`,
      color: '#ffdf6b',
      rp,
      tierMin: base,
      nextRp,
      progress: Math.min(1, (rp - base) / 150)
    };
  }
  let t = RANK_TIERS[0];
  for (let i = RANK_TIERS.length - 1; i >= 0; i--) {
    if (rp >= RANK_TIERS[i].min) { t = RANK_TIERS[i]; break; }
  }
  const offset = rp - t.min;
  const divIdx = Math.min(2, Math.floor(offset / 100));
  const divBase = t.min + divIdx * 100;
  const divRoman = ['I', 'II', 'III'][divIdx];
  return {
    tier: t.name,
    division: divRoman,
    name: `${t.name} ${divRoman}`,
    short: `${t.name.slice(0, 3)} ${divRoman}`,
    color: t.color,
    rp,
    tierMin: divBase,
    nextRp: divBase + 100,
    progress: Math.min(1, (rp - divBase) / 100)
  };
}

// ============================================================
// PERSISTENCE (ATOMIC JSON STORE - MINIMAL RAM & ZERO NATIVE DEPS)
// ============================================================
fs.mkdirSync(DATA_DIR, { recursive: true });

const db = {
  users: {},     // usernameLower -> user object
  sessions: {}   // token -> { usernameLower, createdAt, expiresAt }
};

function loadDb() {
  try {
    if (fs.existsSync(DB_FILE)) {
      const parsed = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
      if (parsed && typeof parsed === 'object') {
        if (parsed.users) db.users = parsed.users;
        if (parsed.sessions) db.sessions = parsed.sessions;
      }
      // Clean expired sessions on boot
      const now = Date.now();
      for (const [tok, sess] of Object.entries(db.sessions)) {
        if (!sess || sess.expiresAt < now || !db.users[sess.usernameLower]) {
          delete db.sessions[tok];
        }
      }
    }
  } catch (err) {
    console.error('[DB] Failed to load db.json:', err.message);
  }
}

let dbDirty = false;
let dbFlushing = false;
function markDirty() { dbDirty = true; }

function flushDb(sync = false) {
  if (!dbDirty) return;
  if (sync) {
    dbDirty = false;
    try {
      const tmp = DB_FILE + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify(db), 'utf8');
      fs.renameSync(tmp, DB_FILE);
    } catch (err) {
      console.error('[DB] Flush error:', err.message);
      dbDirty = true;
    }
    return;
  }
  if (dbFlushing) return;
  dbDirty = false;
  dbFlushing = true;
  const payload = JSON.stringify(db);
  const tmp = DB_FILE + '.tmp';
  fs.promises.writeFile(tmp, payload, 'utf8')
    .then(() => fs.promises.rename(tmp, DB_FILE))
    .catch((err) => {
      console.error('[DB] Async flush error:', err.message);
      dbDirty = true;
    })
    .finally(() => { dbFlushing = false; });
}

loadDb();
setInterval(flushDb, 2000).unref();

// ============================================================
// AUTH & USER HELPERS
// ============================================================
const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;

// Async scrypt runs on libuv's thread pool. scryptSync blocked the main thread (which relays
// every snapshot and input for every match) for tens of milliseconds on each login.
function hashPassword(password, salt) {
  return new Promise((resolve, reject) => {
    crypto.scrypt(password, salt, 64, (err, key) => (err ? reject(err) : resolve(key.toString('hex'))));
  });
}

async function verifyPassword(password, salt, expectedHex) {
  const actual = Buffer.from(await hashPassword(password, salt), 'hex');
  const expected = Buffer.from(expectedHex, 'hex');
  if (actual.length !== expected.length) return false;
  return crypto.timingSafeEqual(actual, expected);
}

function createSession(usernameLower) {
  const token = crypto.randomBytes(32).toString('hex');
  const now = Date.now();
  db.sessions[token] = {
    usernameLower,
    createdAt: now,
    expiresAt: now + SESSION_TTL_MS
  };
  markDirty();
  return token;
}

function getUserByToken(token) {
  if (!token || typeof token !== 'string') return null;
  const sess = db.sessions[token];
  if (!sess) return null;
  if (sess.expiresAt < Date.now()) {
    delete db.sessions[token];
    markDirty();
    return null;
  }
  return db.users[sess.usernameLower] || null;
}

function sanitizeCar(car) {
  const def = { body: 0, primary: -1, secondary: 0, finish: 0, decal: 0, wheels: 0, trail: 0, explosion: 0 };
  if (!car || typeof car !== 'object') return def;
  const num = (v, min, max, fallback) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, Math.floor(n)));
  };
  return {
    body: num(car.body, 0, 4, 0),
    primary: num(car.primary, -1, 11, -1),
    secondary: num(car.secondary, 0, 11, 0),
    finish: num(car.finish, 0, 4, 0),
    decal: num(car.decal, 0, 6, 0),
    wheels: num(car.wheels, 0, 5, 0),
    trail: num(car.trail, 0, 6, 0),
    explosion: num(car.explosion, 0, 4, 0)
  };
}

function publicUser(user) {
  const level = user.level || 1;
  const xp = user.xp || 0;
  const xpNext = xpForLevel(level);
  const rp = user.rp || 0;
  return {
    id: user.id,
    username: user.username,
    level,
    xp,
    xpNext,
    xpProgress: Math.min(1, xp / xpNext),
    totalXp: user.totalXp || 0,
    rp,
    peakRp: user.peakRp || rp,
    rank: computeRank(rp),
    stats: user.stats || { matches: 0, wins: 0, losses: 0, goals: 0, assists: 0, saves: 0, shots: 0, demos: 0, mpMatches: 0, mpWins: 0 },
    car: user.car || sanitizeCar(null)
  };
}

// ============================================================
// RATE LIMITER (HTTP + WS)
// ============================================================
const ipBuckets = new Map();
function checkRateLimit(key, maxHits, windowMs) {
  const now = Date.now();
  let b = ipBuckets.get(key);
  if (!b || now > b.resetAt) {
    b = { count: 0, resetAt: now + windowMs };
    ipBuckets.set(key, b);
  }
  b.count++;
  return b.count <= maxHits;
}

setInterval(() => {
  const now = Date.now();
  for (const [k, b] of ipBuckets.entries()) {
    if (now > b.resetAt) ipBuckets.delete(k);
  }
}, 60000).unref();

// ============================================================
// CODE GENERATOR
// ============================================================
const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
function randomCode(len, prefix = '') {
  let s = prefix;
  for (let i = 0; i < len; i++) {
    s += CODE_CHARS[crypto.randomInt(0, CODE_CHARS.length)];
  }
  return s;
}

// ============================================================
// MULTIPLAYER STATE: CLIENTS, SQUADS, LOBBIES & ANTI-CHEAT
// ============================================================
const clientsByUserId = new Map(); // userId -> ws
const squads = new Map();          // squadCode -> Squad
const lobbies = new Map();         // lobbyCode -> Lobby

const VALID_LAYOUTS = new Set(['default', 'standard', 'octagon', 'hexagon', 'round', 'capsule']);
const VALID_MODES = new Set(['standard', 'mayhem']);
const VALID_BOOST = new Set(['normal', 'unlimited', 'recharge']);
const VALID_MUTATORS = new Set(['normal', 'lowgrav', 'fastball']);

function defaultLobbySettings(hostName) {
  return {
    name: `${hostName}'S LOBBY`,
    isPublic: true,
    ranked: false,
    teamSize: 2,
    mode: 'standard',
    interval: 10,
    arena: 0,
    layout: 'default',
    duration: 300,
    botFill: true,
    botDiff: 1,
    boostMode: 'normal',
    mutator: 'normal'
  };
}

const ARENA_COUNT = 12;

function sanitizeLobbySettings(raw, current) {
  const s = Object.assign({}, current);
  if (!raw || typeof raw !== 'object') return s;
  if (typeof raw.name === 'string') s.name = raw.name.trim().slice(0, 28).toUpperCase() || s.name;
  if (typeof raw.isPublic === 'boolean') s.isPublic = raw.isPublic;
  if (typeof raw.ranked === 'boolean') s.ranked = raw.ranked;
  if (Number.isInteger(raw.teamSize) && raw.teamSize >= 1 && raw.teamSize <= 4) s.teamSize = raw.teamSize;
  if (VALID_MODES.has(raw.mode)) s.mode = raw.mode;
  if ([6, 10, 15].includes(raw.interval)) s.interval = raw.interval;
  if (Number.isInteger(raw.arena) && raw.arena >= -1 && raw.arena <= 11) s.arena = raw.arena;
  if (VALID_LAYOUTS.has(raw.layout)) s.layout = raw.layout;
  if ([0, 120, 180, 300].includes(raw.duration)) s.duration = raw.duration;
  if (typeof raw.botFill === 'boolean') s.botFill = raw.botFill;
  if (Number.isInteger(raw.botDiff) && raw.botDiff >= 0 && raw.botDiff <= 3) s.botDiff = raw.botDiff;
  if (VALID_BOOST.has(raw.boostMode)) s.boostMode = raw.boostMode;
  if (VALID_MUTATORS.has(raw.mutator)) s.mutator = raw.mutator;
  // Ranked enforces fair rules
  if (s.ranked) {
    s.boostMode = 'normal';
    s.mutator = 'normal';
    if (s.duration === 0) s.duration = 300;
  }
  return s;
}

function send(ws, obj) {
  if (ws && ws.readyState === 1) {
    try { ws.send(JSON.stringify(obj)); } catch (_) {}
  }
}

function squadSnapshot(sq) {
  return {
    code: sq.code,
    leaderId: sq.leaderId,
    members: [...sq.members.values()].map(m => {
      const u = db.users[m.usernameLower];
      const rank = computeRank(u ? u.rp : 0);
      return {
        id: m.userId,
        username: m.username,
        level: u ? u.level : 1,
        rank: rank.short,
        rankColor: rank.color,
        isLeader: m.userId === sq.leaderId,
        online: clientsByUserId.has(m.userId)
      };
    })
  };
}

function broadcastSquad(sq, msg) {
  for (const m of sq.members.values()) {
    const ws = clientsByUserId.get(m.userId);
    if (ws) send(ws, msg);
  }
}

function lobbySnapshot(lb) {
  return {
    code: lb.code,
    hostId: lb.hostId,
    status: lb.status,
    settings: lb.settings,
    players: [...lb.players.values()].map(p => {
      const u = db.users[p.usernameLower];
      const rank = computeRank(u ? u.rp : 0);
      return {
        id: p.userId,
        username: p.username,
        team: p.team,
        ready: p.ready,
        isHost: p.userId === lb.hostId,
        level: u ? u.level : 1,
        rp: u ? u.rp : 0,
        rank: rank.short,
        rankColor: rank.color,
        squadCode: p.squadCode || null,
        car: u ? u.car : sanitizeCar(null)
      };
    })
  };
}

function broadcastLobby(lb, msg, excludeUserId = null) {
  for (const p of lb.players.values()) {
    if (excludeUserId && p.userId === excludeUserId) continue;
    const ws = clientsByUserId.get(p.userId);
    if (ws) send(ws, msg);
  }
}

function pickTeamForPlayer(lb, preferredTeam = null) {
  const maxPerTeam = lb.settings.teamSize;
  let blue = 0, orange = 0;
  for (const p of lb.players.values()) {
    if (p.team === 0) blue++;
    else if (p.team === 1) orange++;
  }
  if (preferredTeam === 0 && blue < maxPerTeam) return 0;
  if (preferredTeam === 1 && orange < maxPerTeam) return 1;
  if (blue <= orange && blue < maxPerTeam) return 0;
  if (orange < maxPerTeam) return 1;
  if (blue < maxPerTeam) return 0;
  return -1; // spectator
}

function leaveSquad(ws) {
  if (!ws.squadCode) return;
  const sq = squads.get(ws.squadCode);
  ws.squadCode = null;
  if (!sq) return;
  sq.members.delete(ws.userId);
  if (sq.members.size === 0) {
    squads.delete(sq.code);
    return;
  }
  if (sq.leaderId === ws.userId) {
    const next = sq.members.values().next().value;
    sq.leaderId = next.userId;
  }
  broadcastSquad(sq, { type: 'squad_update', squad: squadSnapshot(sq) });
}

function leaveLobby(ws, reason = 'left') {
  if (!ws.lobbyCode) return;
  const lb = lobbies.get(ws.lobbyCode);
  ws.lobbyCode = null;
  if (!lb) return;
  const wasHost = lb.hostId === ws.userId;
  lb.players.delete(ws.userId);
  if (lb.ac) lb.ac.delete(ws.userId);
  if (lb.sim) lb.sim.removePlayer(ws.userId);

  if (lb.players.size === 0) {
    if (lb.sim) { lb.sim.stop(); lb.sim = null; }
    lobbies.delete(lb.code);
    return;
  }

  if (wasHost) {
    const next = lb.players.values().next().value;
    lb.hostId = next.userId;
    if (lb.status === 'in_match') {
      broadcastLobby(lb, { type: 'host_migrated', hostId: lb.hostId, hostName: next.username });
    }
  }

  if (lb.status === 'in_match') {
    broadcastLobby(lb, { type: 'player_left_match', userId: ws.userId, username: ws.username, reason });
  }
  broadcastLobby(lb, { type: 'lobby_update', lobby: lobbySnapshot(lb) });
}

function joinLobbyInternal(ws, lb, preferredTeam = null) {
  if (ws.lobbyCode === lb.code) return true;
  const banned = db.users[ws.usernameLower];
  if (acBanRemainingMs(banned) > 0) {
    send(ws, { type: 'error', text: acBanMessage(banned) });
    return false;
  }
  if (ws.lobbyCode) leaveLobby(ws);
  const team = pickTeamForPlayer(lb, preferredTeam);
  lb.players.set(ws.userId, {
    userId: ws.userId,
    username: ws.username,
    usernameLower: ws.usernameLower,
    team,
    ready: lb.hostId === ws.userId,
    squadCode: ws.squadCode || null
  });
  ws.lobbyCode = lb.code;
  send(ws, { type: 'lobby_joined', lobby: lobbySnapshot(lb) });
  broadcastLobby(lb, { type: 'lobby_update', lobby: lobbySnapshot(lb) }, ws.userId);
  return true;
}

function pullSquadIntoLobby(leaderWs, lb, preferredTeam) {
  if (!leaderWs.squadCode) return;
  const sq = squads.get(leaderWs.squadCode);
  if (!sq || sq.leaderId !== leaderWs.userId) return;
  for (const m of sq.members.values()) {
    if (m.userId === leaderWs.userId) continue;
    const mws = clientsByUserId.get(m.userId);
    if (mws && mws.readyState === 1) {
      joinLobbyInternal(mws, lb, preferredTeam);
      send(mws, { type: 'toast', text: `SQUAD LEADER PULLED YOU INTO LOBBY ${lb.code}` });
    }
  }
}

// ============================================================
// SERVER-SIDE ANTI-CHEAT ENGINE
// ALL matches (multiplayer, solo & tournament) are 100% simulated on the server
// (see match.js): clients only send controls, so positions, speed, boost,
// power-ups, goals, score, clock, stats, XP and RP cannot be forged.
// This section handles malformed/flooded input, ranked eligibility, abandon
// penalties, and escalating bans (warn -> kick -> timed ban doubling per offense).
// ============================================================
const AC_KICK_STRIKES = 12;
const AC_WARN_STRIKES = 6;
const AC_DECAY_MS = 15000;
const AC_BASE_BAN_MS = 10 * 60 * 1000;
const AC_MAX_BAN_MS = 7 * 24 * 60 * 60 * 1000;
const AC_INPUT_MAX_PER_SEC = 105;   // client sends ~60Hz (+ jitter bursts)
const AC_INPUT_STRIKE_AT = 135;     // sustained flood above rate limit
// Snapshots are full state, so when a client's socket is backed up (slow link, Wi-Fi stall)
// skip sending more: queueing them only adds latency, the next one supersedes them anyway.
const SNAP_BACKLOG_MAX = 48 * 1024;

function acLog(user, reason) {
  console.warn(`[ANTI-CHEAT] ${user ? user.username : '?'}: ${reason}`);
  if (!user) return;
  if (!Array.isArray(user.acLog)) user.acLog = [];
  user.acLog.push({ t: Date.now(), reason: String(reason).slice(0, 120) });
  if (user.acLog.length > 25) user.acLog.splice(0, user.acLog.length - 25);
  markDirty();
}

function acBanRemainingMs(user) {
  if (!user || !user.acBanUntil) return 0;
  return Math.max(0, user.acBanUntil - Date.now());
}

function acBanMessage(user) {
  const mins = Math.ceil(acBanRemainingMs(user) / 60000);
  return `Anti-cheat matchmaking ban: ${mins >= 60 ? Math.ceil(mins / 60) + 'h' : mins + 'm'} remaining.`;
}

// Kicked for cheating: ban length doubles with each offense (10m, 20m, 40m ... 7d).
function acPenalize(user, reason) {
  if (!user) return;
  user.antiCheatFlags = (user.antiCheatFlags || 0) + 1;
  const ms = Math.min(AC_MAX_BAN_MS, AC_BASE_BAN_MS * Math.pow(2, user.antiCheatFlags - 1));
  user.acBanUntil = Date.now() + ms;
  acLog(user, `KICK+BAN ${Math.round(ms / 60000)}m: ${reason}`);
}

function makeAcRec(now) {
  return { strikes: 0, lastDecay: now, inPkt: { pktWindow: now, pktCount: 0 } };
}

function decayStrikes(rec, now) {
  if (rec.strikes > 0 && now - rec.lastDecay >= AC_DECAY_MS) {
    const steps = Math.floor((now - rec.lastDecay) / AC_DECAY_MS);
    rec.strikes = Math.max(0, rec.strikes - steps);
    rec.lastDecay += steps * AC_DECAY_MS;
  } else if (rec.strikes === 0) {
    rec.lastDecay = now;
  }
}

function acRecord(lb, userId) {
  if (!lb.ac) lb.ac = new Map();
  let rec = lb.ac.get(userId);
  if (!rec) {
    rec = makeAcRec(Date.now());
    lb.ac.set(userId, rec);
  }
  return rec;
}

function addStrike(lb, ws, reason, severity = 1) {
  const rec = acRecord(lb, ws.userId);
  decayStrikes(rec, Date.now());
  rec.strikes += severity;
  if (rec.strikes >= AC_KICK_STRIKES) {
    const user = db.users[ws.usernameLower];
    acPenalize(user, reason);
    send(ws, { type: 'anticheat_kick', reason: `Anti-Cheat Kick: ${reason}` });
    leaveLobby(ws, 'anticheat');
    broadcastLobby(lb, { type: 'lobby_chat', system: true, text: `[ANTI-CHEAT] ${ws.username} was removed (${reason}).` });
    return false;
  }
  if (rec.strikes >= AC_WARN_STRIKES && rec.strikes - severity < AC_WARN_STRIKES) {
    send(ws, { type: 'anticheat_warn', reason: `Suspicious activity detected (${reason}). Continued violations will kick you.` });
  }
  return true;
}

function addSoloStrike(ws, reason, severity = 1) {
  if (!ws.soloAc) ws.soloAc = makeAcRec(Date.now());
  const rec = ws.soloAc;
  decayStrikes(rec, Date.now());
  rec.strikes += severity;
  if (rec.strikes >= AC_KICK_STRIKES) {
    const user = db.users[ws.usernameLower];
    acPenalize(user, reason);
    stopSoloMatch(ws);
    send(ws, { type: 'anticheat_kick', reason: `Anti-Cheat Kick: ${reason}` });
    return false;
  }
  if (rec.strikes >= AC_WARN_STRIKES && rec.strikes - severity < AC_WARN_STRIKES) {
    send(ws, { type: 'anticheat_warn', reason: `Suspicious activity detected (${reason}). Continued violations will kick you.` });
  }
  return true;
}

function packetRateOk(rec, now, maxPerSec) {
  if (now - rec.pktWindow >= 1000) { rec.pktWindow = now; rec.pktCount = 0; }
  rec.pktCount++;
  return rec.pktCount <= maxPerSec;
}

// Ranked points require a real opponent: at least one human on each side at match start,
// and no opposing humans connecting from the same IP (alt-account boosting).
function rankedEligibilityFromParticipants(participants) {
  const teams = [[], []];
  for (const p of participants || []) {
    if (p.team !== 0 && p.team !== 1) continue;
    teams[p.team].push(p.ip || 'unknown:' + p.userId);
  }
  if (!teams[0].length || !teams[1].length) return { ok: false, reason: 'Ranked needs human players on both teams' };
  for (const ip of teams[0]) {
    if (teams[1].includes(ip)) return { ok: false, reason: 'Opposing players share a network' };
  }
  return { ok: true };
}

// ============================================================
// MATCH REWARDS (SERVER-AUTHORITATIVE XP + INFINITE LEVELS + RP)
// ============================================================
function ensureStats(user) {
  if (!user.stats) user.stats = { matches: 0, wins: 0, losses: 0, goals: 0, assists: 0, saves: 0, shots: 0, demos: 0, mpMatches: 0, mpWins: 0 };
  return user.stats;
}

function stopSoloMatch(ws) {
  ws.soloGen = (ws.soloGen || 0) + 1;
  if (ws.soloSim) {
    try { ws.soloSim.stop(); } catch (_) {}
    ws.soloSim = null;
  }
  ws.soloMatch = null;
  ws.soloAc = null;
}

function finalizeSoloMatch(ws, simScore, playerStatsMap) {
  const sm = ws.soloMatch;
  if (!sm || sm.finished) return;
  sm.finished = true;
  if (ws.soloSim) { ws.soloSim.stop(); ws.soloSim = null; }
  ws.soloMatch = null;
  ws.soloAc = null;

  const blueScore = Math.max(0, Math.min(99, simScore[0]));
  const orangeScore = Math.max(0, Math.min(99, simScore[1]));
  const winnerTeam = blueScore > orangeScore ? 0 : (orangeScore > blueScore ? 1 : -1);

  let reward = null;
  const user = db.users[ws.usernameLower];
  if (user && sm.duration > 0) {
    const stats = ensureStats(user);
    const st = (playerStatsMap && playerStatsMap[ws.userId]) || {};
    const goals = Math.min(20, st.goals || 0);
    const assists = Math.min(20, st.assists || 0);
    const saves = Math.min(25, st.saves || 0);
    const shots = Math.min(35, st.shots || 0);
    const demos = Math.min(20, st.demos || 0);
    const won = winnerTeam === sm.playerTeam;
    const lost = winnerTeam >= 0 && winnerTeam !== sm.playerTeam;

    stats.matches++;
    if (won) stats.wins++;
    else if (lost) stats.losses++;
    stats.goals += goals;
    stats.assists += assists;
    stats.saves += saves;
    stats.shots += shots;
    stats.demos += demos;

    const diffMult = 1 + sm.difficulty * 0.15 + (sm.tournament ? 0.25 : 0);
    const timeFactor = Math.min(1, sm.duration / 60);
    const rawXp = (75 + (won ? 65 : 20) + goals * 30 + assists * 20 + saves * 25 + shots * 6 + demos * 10) * diffMult * timeFactor;
    const xpRes = addXp(user, Math.min(750, Math.round(rawXp)));

    let rpDelta = 0;
    if (won && (user.rp || 0) < 600) {
      rpDelta = 8 + sm.difficulty * 3;
      user.rp = Math.min(600, (user.rp || 0) + rpDelta);
      user.peakRp = Math.max(user.peakRp || 0, user.rp);
    }
    markDirty();
    reward = { ...xpRes, rpDelta, user: publicUser(user) };
  }

  send(ws, {
    type: 'solo_match_finished',
    score: [blueScore, orangeScore],
    winnerTeam,
    reward
  });
}

function finalizeMultiplayerMatch(lb, simScore, playerStatsMap) {
  const m = lb.match;
  if (!m || m.finished) return;
  m.finished = true;
  lb.status = 'waiting';

  const durationSec = Math.max(1, (Date.now() - m.startedAt) / 1000);
  if (lb.sim) { lb.sim.stop(); lb.sim = null; }
  // Score and stats come from the server's own simulation
  const blueScore = Math.max(0, Math.min(99, simScore[0]));
  const orangeScore = Math.max(0, Math.min(99, simScore[1]));
  const winnerTeam = blueScore > orangeScore ? 0 : (orangeScore > blueScore ? 1 : -1);

  const rewards = {};
  let isRanked = !!lb.settings.ranked && durationSec >= 25;
  let rankedVoid = null;
  if (isRanked) {
    const elig = rankedEligibilityFromParticipants(m.participants);
    if (!elig.ok) { isRanked = false; rankedVoid = elig.reason; }
  }

  const stayedIds = new Set();
  for (const p of lb.players.values()) {
    p.ready = (p.userId === lb.hostId);
    if (p.team !== 0 && p.team !== 1) continue;
    stayedIds.add(p.userId);
    const user = db.users[p.usernameLower];
    if (!user) continue;
    ensureStats(user);

    const st = (playerStatsMap && playerStatsMap[p.userId]) || {};
    const goals = Math.min(20, st.goals || 0);
    const assists = Math.min(20, st.assists || 0);
    const saves = Math.min(25, st.saves || 0);
    const shots = Math.min(35, st.shots || 0);
    const demos = Math.min(20, st.demos || 0);

    const won = winnerTeam === p.team;
    const lost = winnerTeam >= 0 && winnerTeam !== p.team;

    user.stats.matches++;
    user.stats.mpMatches = (user.stats.mpMatches || 0) + 1;
    if (won) { user.stats.wins++; user.stats.mpWins = (user.stats.mpWins || 0) + 1; }
    else if (lost) user.stats.losses++;
    user.stats.goals += goals;
    user.stats.assists += assists;
    user.stats.saves += saves;
    user.stats.shots += shots;
    user.stats.demos += demos;

    // Anti-cheat: scale XP down if match was very short (<30s)
    const timeFactor = Math.min(1, durationSec / 45);
    const rawXp = (120 + (won ? 100 : 35) + goals * 45 + assists * 30 + saves * 35 + shots * 10 + demos * 15) * (isRanked ? 1.25 : 1.0) * timeFactor;
    const xpRes = addXp(user, Math.min(1200, Math.round(rawXp)));

    let rpDelta = 0;
    if (isRanked) {
      const perfBonus = Math.min(12, goals * 3 + saves * 2 + assists * 2);
      if (won) rpDelta = 28 + perfBonus;
      else if (lost) rpDelta = -Math.max(10, 22 - Math.floor(perfBonus * 0.5));
      else rpDelta = 5;
      user.rp = Math.max(0, (user.rp || 0) + rpDelta);
      user.peakRp = Math.max(user.peakRp || 0, user.rp);
    }

    rewards[p.userId] = {
      ...xpRes,
      rpDelta,
      user: publicUser(user)
    };
  }

  // Ranked abandon penalty: players who quit early cannot dodge RP loss
  if (isRanked && Array.isArray(m.participants)) {
    for (const part of m.participants) {
      if (stayedIds.has(part.userId)) continue;
      const user = db.users[part.usernameLower];
      if (!user) continue;
      ensureStats(user);
      user.stats.matches++;
      user.stats.mpMatches = (user.stats.mpMatches || 0) + 1;
      user.stats.losses++;
      user.rp = Math.max(0, (user.rp || 0) - 22);
    }
  }
  markDirty();

  broadcastLobby(lb, {
    type: 'match_finished',
    score: [blueScore, orangeScore],
    winnerTeam,
    rewards,
    rankedVoid,
    lobby: lobbySnapshot(lb)
  });
}

// ============================================================
// HTTP SERVER & REST API
// ============================================================
function readJsonBody(req, maxBytes = 32768) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', chunk => {
      size += chunk.length;
      if (size > maxBytes) {
        reject(new Error('Payload too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (!chunks.length) return resolve({});
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch (_) { reject(new Error('Invalid JSON')); }
    });
    req.on('error', reject);
  });
}

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    ...(corsOriginHeader() ? { 'Access-Control-Allow-Origin': corsOriginHeader(), 'Access-Control-Allow-Credentials': 'true' } : {}),
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Vary': 'Origin',
    'Cache-Control': 'no-store'
  });
  res.end(body);
}

const TRUST_PROXY = !!process.env.FLY_APP_NAME || process.env.TRUST_PROXY === '1';
function getClientIp(req) {
  if (TRUST_PROXY) {
    const fwd = req.headers['fly-client-ip'] || req.headers['x-forwarded-for'];
    if (fwd) return fwd.toString().split(',')[0].trim();
  }
  return (req.socket && req.socket.remoteAddress ? req.socket.remoteAddress : 'unknown').toString().trim();
}

function getBearerToken(req) {
  const h = req.headers.authorization || '';
  if (h.startsWith('Bearer ')) return h.slice(7).trim();
  return null;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const pathname = url.pathname;

  // Health check for Fly.io (always allowed for internal machine health checks)
  if (req.method === 'GET' && pathname === '/api/health') {
    const sim = simPool.getStats();
    return sendJson(res, 200, {
      ok: true,
      region: REGION,
      uptime: Math.floor(process.uptime()),
      online: clientsByUserId.size,
      lobbies: lobbies.size,
      workers: sim.readyWorkers,
      activeMatches: sim.activeMatches
    });
  }

  // Enforce allowed URL / Origin when URL env is configured
  if (!isRequestAllowed(req)) {
    res.writeHead(403, {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store'
    });
    return res.end(JSON.stringify({ error: 'Forbidden' }));
  }

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      ...(corsOriginHeader() ? { 'Access-Control-Allow-Origin': corsOriginHeader(), 'Access-Control-Allow-Credentials': 'true' } : {}),
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Vary': 'Origin',
      'Access-Control-Max-Age': '86400'
    });
    return res.end();
  }

  // Static index.html serving (cached in memory so HTTP loads never block the event loop)
  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    try {
      if (!cachedIndexHtml) {
        let html = fs.readFileSync(INDEX_FILE, 'utf8');
        if (ALLOWED_ORIGIN) {
          html = html.replace('</head>', `<script>window.__HYPERSTRIKE_URL__=${JSON.stringify(ALLOWED_ORIGIN)};</script></head>`);
        }
        cachedIndexHtml = Buffer.from(html, 'utf8');
      }
      res.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'no-cache',
        'Set-Cookie': issueClientCookie(req),
        'Content-Security-Policy': "frame-ancestors 'self'",
        'X-Frame-Options': 'SAMEORIGIN'
      });
      return res.end(cachedIndexHtml);
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      return res.end('index.html not found');
    }
  }

  // Every API call must come from a page this server served.
  if (pathname.startsWith('/api/') && !hasValidClientCookie(req)) {
    return sendJson(res, 403, { error: 'Client not recognized. Reload the game from the official site.' });
  }

  const ip = getClientIp(req);

  try {
    // REGISTER
    if (req.method === 'POST' && pathname === '/api/register') {
      if (!checkRateLimit('auth:' + ip, 20, 60000)) {
        return sendJson(res, 429, { error: 'Too many requests. Please wait a minute.' });
      }
      const body = await readJsonBody(req);
      const username = typeof body.username === 'string' ? body.username.trim() : '';
      const password = typeof body.password === 'string' ? body.password : '';
      const confirmPassword = typeof body.confirmPassword === 'string' ? body.confirmPassword : '';

      if (username.length < 3 || username.length > 18) {
        return sendJson(res, 400, { error: 'Username must be 3 to 18 characters.' });
      }
      if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
        return sendJson(res, 400, { error: 'Username can only use letters, numbers, _ and -.' });
      }
      if (password.length < 6 || password.length > 128) {
        return sendJson(res, 400, { error: 'Password must be at least 6 characters.' });
      }
      if (password !== confirmPassword) {
        return sendJson(res, 400, { error: 'Passwords do not match.' });
      }

      const key = username.toLowerCase();
      if (db.users[key]) {
        return sendJson(res, 409, { error: 'That username is already taken.' });
      }

      const salt = crypto.randomBytes(16).toString('hex');
      const passHash = await hashPassword(password, salt);
      if (db.users[key]) {
        return sendJson(res, 409, { error: 'That username is already taken.' });
      }
      const user = {
        id: crypto.randomBytes(8).toString('hex'),
        username,
        passHash,
        salt,
        createdAt: Date.now(),
        level: 1,
        xp: 0,
        totalXp: 0,
        rp: 100,
        peakRp: 100,
        stats: { matches: 0, wins: 0, losses: 0, goals: 0, assists: 0, saves: 0, shots: 0, demos: 0, mpMatches: 0, mpWins: 0 },
        car: sanitizeCar(null),
        antiCheatFlags: 0,
        lastReportAt: 0
      };
      db.users[key] = user;
      const token = createSession(key);
      markDirty();
      return sendJson(res, 201, { token, user: publicUser(user) });
    }

    // LOGIN
    if (req.method === 'POST' && pathname === '/api/login') {
      if (!checkRateLimit('auth:' + ip, 25, 60000)) {
        return sendJson(res, 429, { error: 'Too many login attempts. Please wait a minute.' });
      }
      const body = await readJsonBody(req);
      const username = typeof body.username === 'string' ? body.username.trim() : '';
      const password = typeof body.password === 'string' ? body.password : '';
      const key = username.toLowerCase();
      const user = db.users[key];
      if (!user || !(await verifyPassword(password, user.salt, user.passHash))) {
        return sendJson(res, 401, { error: 'Invalid username or password.' });
      }
      const token = createSession(key);
      return sendJson(res, 200, { token, user: publicUser(user) });
    }

    // LOGOUT
    if (req.method === 'POST' && pathname === '/api/logout') {
      const token = getBearerToken(req);
      if (token && db.sessions[token]) {
        delete db.sessions[token];
        markDirty();
      }
      return sendJson(res, 200, { ok: true });
    }

    // CHANGE PASSWORD
    if (req.method === 'POST' && pathname === '/api/password') {
      const token = getBearerToken(req);
      const user = getUserByToken(token);
      if (!user) return sendJson(res, 401, { error: 'Not authenticated' });
      if (!checkRateLimit('pass:' + user.id, 10, 60000)) {
        return sendJson(res, 429, { error: 'Too many password change attempts. Please wait a minute.' });
      }
      const body = await readJsonBody(req);
      const currentPassword = typeof body.currentPassword === 'string' ? body.currentPassword : '';
      const newPassword = typeof body.newPassword === 'string' ? body.newPassword : '';
      const confirmPassword = typeof body.confirmPassword === 'string' ? body.confirmPassword : '';

      if (!currentPassword || !(await verifyPassword(currentPassword, user.salt, user.passHash))) {
        return sendJson(res, 401, { error: 'Current password is incorrect.' });
      }
      if (newPassword.length < 6 || newPassword.length > 128) {
        return sendJson(res, 400, { error: 'New password must be at least 6 characters.' });
      }
      if (newPassword !== confirmPassword) {
        return sendJson(res, 400, { error: 'New passwords do not match.' });
      }
      if (newPassword === currentPassword) {
        return sendJson(res, 400, { error: 'New password must be different from your current password.' });
      }

      const salt = crypto.randomBytes(16).toString('hex');
      const newHash = await hashPassword(newPassword, salt);
      user.salt = salt;
      user.passHash = newHash;
      const key = user.username.toLowerCase();
      for (const [tok, sess] of Object.entries(db.sessions)) {
        if (sess.usernameLower === key && tok !== token) {
          delete db.sessions[tok];
        }
      }
      markDirty();
      return sendJson(res, 200, { ok: true });
    }

    // CURRENT USER
    if (req.method === 'GET' && pathname === '/api/me') {
      const user = getUserByToken(getBearerToken(req));
      if (!user) return sendJson(res, 401, { error: 'Not authenticated' });
      return sendJson(res, 200, {
        user: publicUser(user),
        server: { region: REGION, online: clientsByUserId.size, lobbies: lobbies.size }
      });
    }

    // SAVE CAR CUSTOMIZATION
    if (req.method === 'POST' && pathname === '/api/car') {
      const user = getUserByToken(getBearerToken(req));
      if (!user) return sendJson(res, 401, { error: 'Not authenticated' });
      const body = await readJsonBody(req);
      user.car = sanitizeCar(body.car);
      markDirty();
      return sendJson(res, 200, { car: user.car });
    }

    // LEADERBOARD
    if (req.method === 'GET' && pathname === '/api/leaderboard') {
      const sort = url.searchParams.get('sort') === 'level' ? 'level' : 'rp';
      const list = Object.values(db.users)
        .map(u => {
          const r = computeRank(u.rp || 0);
          return {
            username: u.username,
            level: u.level || 1,
            totalXp: u.totalXp || 0,
            rp: u.rp || 0,
            rank: r.name,
            rankColor: r.color,
            wins: (u.stats && u.stats.wins) || 0,
            goals: (u.stats && u.stats.goals) || 0,
            matches: (u.stats && u.stats.matches) || 0
          };
        })
        .sort((a, b) => sort === 'level' ? (b.level - a.level || b.totalXp - a.totalXp) : (b.rp - a.rp || b.level - a.level))
        .slice(0, 50);
      return sendJson(res, 200, { leaderboard: list, sort });
    }

    return sendJson(res, 404, { error: 'Not found' });
  } catch (err) {
    return sendJson(res, 400, { error: err.message || 'Bad request' });
  }
});

// ============================================================
// WEBSOCKET SERVER
// ============================================================
const wss = new WebSocketServer({
  server,
  maxPayload: 16384,
  verifyClient: (info, done) => {
    if (!isRequestAllowed(info.req)) return done(false, 403, 'Forbidden');
    if (!hasValidClientCookie(info.req)) return done(false, 403, 'Forbidden');
    done(true);
  }
});

// Refresh the page cookie on every successful upgrade so long sessions keep working.
wss.on('headers', (headers, req) => {
  headers.push('Set-Cookie: ' + issueClientCookie(req));
});

wss.on('connection', (ws, req) => {
  if (req.socket && req.socket.setNoDelay) req.socket.setNoDelay(true);
  ws.ip = getClientIp(req);
  ws.isAlive = true;
  ws.authed = false;
  ws.userId = null;
  ws.username = null;
  ws.usernameLower = null;
  ws.squadCode = null;
  ws.lobbyCode = null;
  ws.soloSim = null;
  ws.soloMatch = null;
  ws.soloAc = null;
  ws.soloGen = 0;
  ws.msgCount = 0;
  ws.msgWindowStart = Date.now();
  ws.chatTimes = [];

  ws.on('pong', () => { ws.isAlive = true; });

  ws.on('message', (raw) => {
    const now = Date.now();
    if (now - ws.msgWindowStart > 1000) {
      ws.msgWindowStart = now;
      ws.msgCount = 0;
    }
    ws.msgCount++;
    if (ws.msgCount > 200) {
      return; // drop extreme flood
    }

    let msg;
    try { msg = JSON.parse(raw.toString('utf8')); } catch (_) { return; }
    if (!msg || typeof msg.type !== 'string') return;

    // Latency probe for the client's ping meter: echo the client's timestamp straight back
    if (msg.type === 'ping') {
      if (typeof msg.t === 'number' && Number.isFinite(msg.t)) send(ws, { type: 'pong', t: msg.t });
      return;
    }

    // Auth required before any other WS message
    if (msg.type === 'auth') {
      const user = getUserByToken(msg.token);
      if (!user) {
        send(ws, { type: 'auth_error', error: 'Invalid session' });
        return;
      }
      // Disconnect previous socket for same account if any
      const prev = clientsByUserId.get(user.id);
      if (prev && prev !== ws) {
        send(prev, { type: 'kicked', reason: 'Signed in from another tab/device.' });
        stopSoloMatch(prev);
        leaveLobby(prev);
        leaveSquad(prev);
        try { prev.close(); } catch (_) {}
      }
      ws.authed = true;
      ws.userId = user.id;
      ws.username = user.username;
      ws.usernameLower = user.username.toLowerCase();
      clientsByUserId.set(user.id, ws);
      send(ws, {
        type: 'auth_ok',
        user: publicUser(user),
        region: REGION,
        online: clientsByUserId.size
      });
      return;
    }

    if (!ws.authed) {
      send(ws, { type: 'auth_error', error: 'Please sign in first.' });
      return;
    }

    switch (msg.type) {
      // ---- SQUADS ----
      case 'squad_create': {
        if (ws.squadCode) leaveSquad(ws);
        let code;
        do { code = randomCode(4, 'SQ'); } while (squads.has(code));
        const sq = { code, leaderId: ws.userId, members: new Map() };
        sq.members.set(ws.userId, { userId: ws.userId, username: ws.username, usernameLower: ws.usernameLower });
        squads.set(code, sq);
        ws.squadCode = code;
        send(ws, { type: 'squad_update', squad: squadSnapshot(sq) });
        break;
      }
      case 'squad_join': {
        const code = String(msg.code || '').trim().toUpperCase();
        const sq = squads.get(code);
        if (!sq) return send(ws, { type: 'error', text: `Squad "${code}" not found.` });
        if (sq.members.size >= 5) return send(ws, { type: 'error', text: 'Squad is full (max 5).' });
        if (ws.squadCode && ws.squadCode !== code) leaveSquad(ws);
        sq.members.set(ws.userId, { userId: ws.userId, username: ws.username, usernameLower: ws.usernameLower });
        ws.squadCode = code;
        broadcastSquad(sq, { type: 'squad_update', squad: squadSnapshot(sq) });
        // If leader is already in a waiting lobby, pull joiner in too
        const leaderWs = clientsByUserId.get(sq.leaderId);
        if (leaderWs && leaderWs.lobbyCode) {
          const lb = lobbies.get(leaderWs.lobbyCode);
          if (lb && lb.status === 'waiting') {
            const lp = lb.players.get(leaderWs.userId);
            joinLobbyInternal(ws, lb, lp ? lp.team : null);
          }
        }
        break;
      }
      case 'squad_leave': {
        leaveSquad(ws);
        send(ws, { type: 'squad_update', squad: null });
        break;
      }
      case 'squad_kick': {
        const sq = squads.get(ws.squadCode);
        if (!sq || sq.leaderId !== ws.userId) return;
        const targetId = String(msg.userId || '');
        if (!targetId || targetId === ws.userId) return;
        const targetWs = clientsByUserId.get(targetId);
        if (targetWs) {
          leaveSquad(targetWs);
          send(targetWs, { type: 'squad_update', squad: null });
          send(targetWs, { type: 'toast', text: 'YOU WERE REMOVED FROM THE SQUAD' });
        } else {
          sq.members.delete(targetId);
          broadcastSquad(sq, { type: 'squad_update', squad: squadSnapshot(sq) });
        }
        break;
      }

      // ---- LOBBIES ----
      case 'lobby_list': {
        const list = [];
        for (const lb of lobbies.values()) {
          if (!lb.settings.isPublic) continue;
          const host = lb.players.get(lb.hostId);
          list.push({
            code: lb.code,
            name: lb.settings.name,
            host: host ? host.username : 'HOST',
            players: lb.players.size,
            maxPlayers: lb.settings.teamSize * 2,
            status: lb.status,
            ranked: lb.settings.ranked,
            mode: lb.settings.mode,
            teamSize: lb.settings.teamSize,
            arena: lb.settings.arena
          });
        }
        send(ws, { type: 'lobby_list', lobbies: list, online: clientsByUserId.size });
        break;
      }
      case 'lobby_create': {
        const me = db.users[ws.usernameLower];
        if (acBanRemainingMs(me) > 0) return send(ws, { type: 'error', text: acBanMessage(me) });
        stopSoloMatch(ws);
        if (ws.lobbyCode) leaveLobby(ws);
        let code;
        do { code = randomCode(6); } while (lobbies.has(code));
        const settings = sanitizeLobbySettings(msg.settings, defaultLobbySettings(ws.username));
        const lb = {
          code,
          hostId: ws.userId,
          status: 'waiting',
          settings,
          players: new Map(),
          ac: new Map(),
          match: null
        };
        lobbies.set(code, lb);
        joinLobbyInternal(ws, lb, 0);
        pullSquadIntoLobby(ws, lb, 0);
        break;
      }
      case 'lobby_join': {
        const code = String(msg.code || '').trim().toUpperCase();
        const lb = lobbies.get(code);
        if (!lb) return send(ws, { type: 'error', text: `Lobby "${code}" not found.` });
        if (lb.status === 'in_match') return send(ws, { type: 'error', text: 'Match is already in progress.' });
        stopSoloMatch(ws);
        joinLobbyInternal(ws, lb);
        const myP = lb.players.get(ws.userId);
        pullSquadIntoLobby(ws, lb, myP ? myP.team : null);
        break;
      }
      case 'lobby_leave': {
        leaveLobby(ws);
        send(ws, { type: 'lobby_left' });
        break;
      }
      case 'lobby_settings': {
        const lb = lobbies.get(ws.lobbyCode);
        if (!lb || lb.hostId !== ws.userId || lb.status !== 'waiting') return;
        lb.settings = sanitizeLobbySettings(msg.settings, lb.settings);
        // Ensure team counts still fit new teamSize
        let b = 0, o = 0;
        for (const p of lb.players.values()) {
          if (p.team === 0) { if (++b > lb.settings.teamSize) p.team = -1; }
          else if (p.team === 1) { if (++o > lb.settings.teamSize) p.team = -1; }
        }
        broadcastLobby(lb, { type: 'lobby_update', lobby: lobbySnapshot(lb) });
        break;
      }
      case 'lobby_team': {
        const lb = lobbies.get(ws.lobbyCode);
        if (!lb || lb.status !== 'waiting') return;
        const p = lb.players.get(ws.userId);
        if (!p) return;
        let want;
        if (msg.team === 'random') {
          let blue = 0, orange = 0;
          for (const pl of lb.players.values()) {
            if (pl.userId === ws.userId) continue;
            if (pl.team === 0) blue++;
            else if (pl.team === 1) orange++;
          }
          const avail = [];
          if (blue < lb.settings.teamSize) avail.push(0);
          if (orange < lb.settings.teamSize) avail.push(1);
          if (!avail.length) {
            return send(ws, { type: 'error', text: 'Both teams are full.' });
          }
          want = avail[Math.floor(Math.random() * avail.length)];
        } else {
          want = Number(msg.team);
          if (![0, 1, -1].includes(want)) return;
          if (want === 0 || want === 1) {
            let count = 0;
            for (const pl of lb.players.values()) if (pl.team === want && pl.userId !== ws.userId) count++;
            if (count >= lb.settings.teamSize) {
              return send(ws, { type: 'error', text: 'That team is full.' });
            }
          }
        }
        p.team = want;
        broadcastLobby(lb, { type: 'lobby_update', lobby: lobbySnapshot(lb) });
        break;
      }
      case 'lobby_ready': {
        const lb = lobbies.get(ws.lobbyCode);
        if (!lb || lb.status !== 'waiting') return;
        const p = lb.players.get(ws.userId);
        if (!p) return;
        p.ready = !p.ready;
        broadcastLobby(lb, { type: 'lobby_update', lobby: lobbySnapshot(lb) });
        break;
      }
      case 'lobby_kick': {
        const lb = lobbies.get(ws.lobbyCode);
        if (!lb || lb.hostId !== ws.userId || lb.status !== 'waiting') return;
        const targetId = String(msg.userId || '');
        if (!targetId || targetId === ws.userId) return;
        const targetWs = clientsByUserId.get(targetId);
        if (targetWs) {
          leaveLobby(targetWs, 'kicked');
          send(targetWs, { type: 'lobby_left', kicked: true });
          send(targetWs, { type: 'toast', text: 'YOU WERE KICKED FROM THE LOBBY' });
        }
        break;
      }
      case 'lobby_chat': {
        const lb = lobbies.get(ws.lobbyCode);
        if (!lb) return;
        ws.chatTimes = ws.chatTimes.filter(t => now - t < 10000);
        if (ws.chatTimes.length >= 6) return;
        ws.chatTimes.push(now);
        const text = String(msg.text || '').trim().slice(0, 160);
        if (!text) return;
        broadcastLobby(lb, { type: 'lobby_chat', from: ws.username, text });
        break;
      }
      case 'lobby_start': {
        const lb = lobbies.get(ws.lobbyCode);
        if (!lb || lb.hostId !== ws.userId || lb.status !== 'waiting') return;
        // Need at least 1 active player
        const active = [...lb.players.values()].filter(p => p.team === 0 || p.team === 1);
        if (active.length === 0) {
          return send(ws, { type: 'error', text: 'At least one player must be on Blue or Orange.' });
        }
        lb.status = 'in_match';
        lb.ac = new Map();
        const participants = active.map(p => {
          const pws = clientsByUserId.get(p.userId);
          return {
            userId: p.userId,
            usernameLower: p.usernameLower,
            team: p.team,
            ip: pws && pws.ip ? pws.ip : 'unknown:' + p.userId
          };
        });
        const matchObj = { startedAt: now, finished: false, participants };
        lb.match = matchObj;
        const settings = lb.settings.arena === -1 ? Object.assign({}, lb.settings, { arena: Math.floor(Math.random() * ARENA_COUNT) }) : Object.assign({}, lb.settings);
        const snap = lobbySnapshot(lb);
        const matchPlayers = snap.players
          .filter(p => p.team === 0 || p.team === 1)
          .map(p => ({ userId: p.id, username: p.username, team: p.team, car: p.car }));
        simPool.createMatch({
          settings,
          players: matchPlayers,
          broadcast: m => broadcastLobby(lb, m),
          sendRaw: (userId, raw) => {
            const pws = clientsByUserId.get(userId);
            if (pws && pws.readyState === 1 && pws.lobbyCode === lb.code && pws.bufferedAmount < SNAP_BACKLOG_MAX) pws.send(raw);
          },
          spectators: raw => {
            for (const p of lb.players.values()) {
              if (p.team === 0 || p.team === 1) continue;
              const pws = clientsByUserId.get(p.userId);
              if (pws && pws.readyState === 1 && pws.bufferedAmount < SNAP_BACKLOG_MAX) pws.send(raw);
            }
          },
          onGoal: info => broadcastLobby(lb, Object.assign({ type: 'match_goal_ack' }, info)),
          onEnd: ({ score, stats }) => finalizeMultiplayerMatch(lb, score, stats),
          onError: err => {
            console.error('[SIM] Match aborted:', err.message);
            if (lb.match !== matchObj || matchObj.finished) return;
            matchObj.finished = true;
            lb.status = 'waiting';
            lb.sim = null;
            for (const p of lb.players.values()) p.ready = (p.userId === lb.hostId);
            broadcastLobby(lb, { type: 'error', text: 'Match ended unexpectedly due to a simulation error.' });
            broadcastLobby(lb, { type: 'lobby_update', lobby: lobbySnapshot(lb) });
          }
        }).then(sim => {
          if (!lobbies.has(lb.code) || lb.status !== 'in_match' || lb.match !== matchObj || matchObj.finished) {
            sim.stop();
            return;
          }
          lb.sim = sim;
          for (const mp of matchPlayers) {
            if (!lb.players.has(mp.userId)) sim.removePlayer(mp.userId);
          }
          const curSnap = lobbySnapshot(lb);
          broadcastLobby(lb, {
            type: 'match_start',
            lobby: curSnap,
            hostId: lb.hostId,
            settings,
            players: curSnap.players,
            cars: sim.carInfo,
            mid: sim.id,
            serverSim: true
          });
          sim.startWhenLoaded(matchPlayers.map(mp => mp.userId).filter(id => lb.players.has(id)));
        }).catch(err => {
          console.error('[SIM] Failed to start match:', err);
          if (lb.match === matchObj && !matchObj.finished) {
            lb.status = 'waiting';
            lb.match = null;
            lb.sim = null;
            broadcastLobby(lb, { type: 'error', text: 'Server could not start the match.' });
          }
        });
        break;
      }

      // ---- SOLO / AI / TOURNAMENT (100% SERVER-SIMULATED) ----
      case 'solo_start': {
        const me = db.users[ws.usernameLower];
        if (acBanRemainingMs(me) > 0) return send(ws, { type: 'error', text: acBanMessage(me) });
        if (!checkRateLimit('sstart:' + ws.userId, 20, 60000)) {
          return send(ws, { type: 'error', text: 'Too many matches started. Please wait a moment.' });
        }
        if (ws.lobbyCode) {
          const lb = lobbies.get(ws.lobbyCode);
          if (lb && lb.status === 'in_match') return send(ws, { type: 'error', text: 'Finish or leave your multiplayer match first.' });
          leaveLobby(ws);
        }
        stopSoloMatch(ws);
        const gen = ws.soloGen;
        const raw = (msg.cfg && typeof msg.cfg === 'object') ? msg.cfg : {};
        const tournament = !!raw.tournament;
        const teamSize = Number.isInteger(raw.teamSize) && raw.teamSize >= 1 && raw.teamSize <= 4 ? raw.teamSize : 2;
        const difficulty = Number.isInteger(raw.difficulty) && raw.difficulty >= 0 && raw.difficulty <= 3 ? raw.difficulty : 1;
        const duration = tournament ? 180 : ([0, 120, 180, 300].includes(Number(raw.duration)) ? Number(raw.duration) : 300);
        const arena = Number.isInteger(raw.arena) && raw.arena >= 0 && raw.arena < ARENA_COUNT ? raw.arena : Math.floor(Math.random() * ARENA_COUNT);
        const layout = VALID_LAYOUTS.has(raw.layout) ? raw.layout : 'default';
        const mode = VALID_MODES.has(raw.mode) ? raw.mode : 'standard';
        const interval = [6, 10, 15].includes(Number(raw.interval)) ? Number(raw.interval) : 10;
        const playerTeam = tournament ? 0 : (raw.side === 0 || raw.side === 1 ? raw.side : (Math.random() < 0.5 ? 0 : 1));
        const teamNames = Array.isArray(raw.teamNames) && raw.teamNames.length === 2
          ? [String(raw.teamNames[0] || 'BLUE').slice(0, 24).toUpperCase(), String(raw.teamNames[1] || 'ORANGE').slice(0, 24).toUpperCase()]
          : ['BLUE', 'ORANGE'];
        const settings = {
          teamSize,
          mode,
          interval,
          arena,
          layout,
          duration,
          botFill: true,
          botDiff: difficulty,
          boostMode: 'normal',
          mutator: 'normal'
        };
        const teammateDiff = tournament ? Math.min(3, difficulty + 1) : difficulty;
        simPool.createMatch({
          settings,
          teammateDiff,
          players: [{ userId: ws.userId, username: ws.username, team: playerTeam, car: me ? me.car : sanitizeCar(null) }],
          broadcast: m => send(ws, m),
          sendRaw: (userId, rawStr) => {
            if (userId === ws.userId && ws.readyState === 1 && ws.bufferedAmount < SNAP_BACKLOG_MAX) ws.send(rawStr);
          },
          onGoal: info => send(ws, Object.assign({ type: 'match_goal_ack' }, info)),
          onEnd: ({ score, stats }) => finalizeSoloMatch(ws, score, stats),
          onError: err => {
            console.error('[SIM] Solo match aborted:', err.message);
            if (ws.soloGen !== gen) return;
            stopSoloMatch(ws);
            send(ws, { type: 'error', text: 'Match ended unexpectedly due to a simulation error.' });
          }
        }).then(sim => {
          if (ws.soloGen !== gen || ws.readyState !== 1) {
            sim.stop();
            return;
          }
          ws.soloSim = sim;
          ws.soloAc = makeAcRec(Date.now());
          ws.soloMatch = { startedAt: Date.now(), duration, difficulty, tournament, playerTeam, finished: false };
          send(ws, {
            type: 'solo_match_start',
            settings,
            cars: sim.carInfo,
            teamNames,
            tournament,
            solo: true,
            mid: sim.id,
            serverSim: true
          });
          sim.startWhenLoaded([ws.userId]);
        }).catch(err => {
          console.error('[SIM] Failed to start solo match:', err);
          if (ws.soloGen === gen && ws.readyState === 1) {
            send(ws, { type: 'error', text: 'Server could not start the match.' });
          }
        });
        break;
      }
      case 'solo_leave': {
        stopSoloMatch(ws);
        break;
      }
      case 'solo_pause': {
        if (ws.soloSim) ws.soloSim.setPaused(!!msg.paused);
        break;
      }

      // Client finished building the arena for match `m`: the kickoff countdown can begin
      case 'match_loaded': {
        const mid = Number.isInteger(msg.m) ? msg.m : null;
        if (ws.soloSim && ws.soloSim.id === mid) { ws.soloSim.markLoaded(ws.userId); break; }
        const lb = lobbies.get(ws.lobbyCode);
        if (lb && lb.sim && lb.sim.id === mid) lb.sim.markLoaded(ws.userId);
        break;
      }

      // ---- IN-MATCH: clients send only their controls; the server simulates ----
      case 'in': {
        // Inputs are tagged with the match id they were produced for. Packets still in flight
        // from a previous match (restart / rematch) must not reach the new one.
        const inMid = Number.isInteger(msg.m) ? msg.m : null;
        if (ws.soloSim) {
          if (inMid !== null && inMid !== ws.soloSim.id) return;
          if (!ws.soloAc) ws.soloAc = makeAcRec(now);
          if (!packetRateOk(ws.soloAc.inPkt, now, AC_INPUT_MAX_PER_SEC)) {
            if (ws.soloAc.inPkt.pktCount === AC_INPUT_STRIKE_AT) addSoloStrike(ws, 'Input packet flood', 2);
            return;
          }
          if (!ws.soloSim.pushInput(ws.userId, msg.s, msg.c)) addSoloStrike(ws, 'Malformed input packet', 1);
          break;
        }
        const lb = lobbies.get(ws.lobbyCode);
        if (!lb || lb.status !== 'in_match' || !lb.sim) return;
        if (inMid !== null && inMid !== lb.sim.id) return;
        const rec = acRecord(lb, ws.userId);
        if (!packetRateOk(rec.inPkt, now, AC_INPUT_MAX_PER_SEC)) {
          if (rec.inPkt.pktCount === AC_INPUT_STRIKE_AT) addStrike(lb, ws, 'Input packet flood', 2);
          return;
        }
        if (!lb.sim.pushInput(ws.userId, msg.s, msg.c)) addStrike(lb, ws, 'Malformed input packet', 1);
        break;
      }
    }
  });

  ws.on('close', () => {
    if (ws.userId && clientsByUserId.get(ws.userId) === ws) {
      clientsByUserId.delete(ws.userId);
    }
    stopSoloMatch(ws);
    leaveLobby(ws, 'disconnected');
    leaveSquad(ws);
  });
});

// Heartbeat to drop dead connections & keep Fly.io proxy alive
setInterval(() => {
  const now = Date.now();
  for (const ws of wss.clients) {
    if (ws.soloSim && now - ws.soloSim.createdAt > 2 * 60 * 60 * 1000) {
      stopSoloMatch(ws);
    }
    if (!ws.isAlive) { ws.terminate(); continue; }
    ws.isAlive = false;
    ws.ping();
  }
}, 25000).unref();

simPool.init(INDEX_FILE).then(() => {
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`[HYPERSTRIKE] Server running on :${PORT} (region=${REGION}, data=${DATA_DIR}, simWorkers=${simPool.workerCount})`);
  });
}).catch((err) => {
  console.error('[HYPERSTRIKE] Failed to initialize simulation worker pool:', err);
  process.exit(1);
});

let shuttingDown = false;
function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('[HYPERSTRIKE] Shutting down, flushing DB...');
  flushDb(true);
  simPool.shutdown().finally(() => process.exit(0));
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
