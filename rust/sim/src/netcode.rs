use glam::{DQuat, DVec3};

use crate::math_ext::{Basis, Controls, ThreeQuatExt};
use crate::physics::{Ball, Car};
use crate::world::World;

pub const NET_MAGIC: u8 = 0x48; // 'H'
pub const NET_VERSION: u8 = 2;
pub const NET_TPI: usize = 2;
pub const NET_BALL_N: usize = 22;
pub const NET_CAR_N: usize = 47;
pub const NET_NO_CD: u16 = 0xFFFF;
pub const NET_STATES: [&str; 5] = ["idle", "countdown", "play", "goal", "ended"];

#[inline]
fn net_num(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

#[inline]
fn round3(v: f64) -> f64 {
    let n = net_num(v);
    let r = (n * 1000.0).round() / 1000.0;
    if r == 0.0 { 0.0 } else { r }
}

/// Matches JS `netPackControls(c)`: returns `[throttle, steer, pitch, yaw, roll, bits]`.
pub fn net_pack_controls(c: &Controls) -> [f64; 6] {
    let bits = (u32::from(c.jump))
        | (u32::from(c.boost) << 1)
        | (u32::from(c.handbrake) << 2)
        | (u32::from(c.air_roll_free) << 3)
        | (u32::from(c.item) << 4);
    [
        round3(c.throttle),
        round3(c.steer),
        round3(c.pitch),
        round3(c.yaw),
        round3(c.roll),
        bits as f64,
    ]
}

/// Matches JS `netUnpackControls(a, into)`.
pub fn net_unpack_controls(a: &[f64; 6], into: &mut Controls) {
    into.throttle = a[0];
    into.steer = a[1];
    into.pitch = a[2];
    into.yaw = a[3];
    into.roll = a[4];
    let b = a[5] as i32;
    into.jump = (b & 1) != 0;
    into.boost = (b & 2) != 0;
    into.handbrake = (b & 4) != 0;
    into.air_roll_free = (b & 8) != 0;
    into.item = (b & 16) != 0;
}

pub fn net_write_car(f: &mut [f32], o: usize, c: &Car, ctl: Option<&[f64; 6]>, tick: u64) {
    f[o] = c.pos.x as f32;
    f[o + 1] = c.pos.y as f32;
    f[o + 2] = c.pos.z as f32;
    f[o + 3] = c.vel.x as f32;
    f[o + 4] = c.vel.y as f32;
    f[o + 5] = c.vel.z as f32;
    f[o + 6] = c.quat.x as f32;
    f[o + 7] = c.quat.y as f32;
    f[o + 8] = c.quat.z as f32;
    f[o + 9] = c.quat.w as f32;
    f[o + 10] = c.ang_vel.x as f32;
    f[o + 11] = c.ang_vel.y as f32;
    f[o + 12] = c.ang_vel.z as f32;
    f[o + 13] = net_num(c.boost) as f32;
    f[o + 14] = net_num(c.ground_lock) as f32;
    f[o + 15] = c.contacts as f32;
    f[o + 16] = c.ground_normal.x as f32;
    f[o + 17] = c.ground_normal.y as f32;
    f[o + 18] = c.ground_normal.z as f32;
    f[o + 19] = net_num(c.jump_hold_t) as f32;
    f[o + 20] = net_num(c.time_since_jump) as f32;
    f[o + 21] = net_num(c.flip_t) as f32;
    f[o + 22] = c.flip_axis.x as f32;
    f[o + 23] = c.flip_axis.y as f32;
    f[o + 24] = c.flip_axis.z as f32;
    f[o + 25] = net_num(c.coyote_t) as f32;
    f[o + 26] = net_num(c.air_time) as f32;
    f[o + 27] = net_num(c.body_contact_t) as f32;
    f[o + 28] = net_num(c.boost_t) as f32;
    f[o + 29] = net_num(c.sliding) as f32;
    f[o + 30] = net_num(c.traction) as f32;
    f[o + 31] = net_num(c.respawn_t) as f32;
    f[o + 32] = net_num(c.blast_t) as f32;
    f[o + 33] = net_num(c.speed_mul) as f32;
    f[o + 34] = net_num(c.nitro_t) as f32;
    f[o + 35] = ((tick as i64 - c.last_ball_touch_tick).clamp(0, 255)) as f32;

    let mut bits: u32 = 0;
    let bools = [
        c.prev_jump,
        c.on_ground,
        c.jumping,
        c.jumped,
        c.has_flip,
        c.double_jumped,
        c.flipping,
        c.body_contact,
        c.is_boosting,
        c.supersonic,
        c.was_supersonic,
        c.handbrake,
        c.demolished,
    ];
    for (i, &b) in bools.iter().enumerate() {
        if b {
            bits |= 1 << i;
        }
    }
    for (i, w) in c.wheels.iter().enumerate() {
        f[o + 36 + i] = w.len as f32;
        if w.contact {
            bits |= 1 << (13 + i);
        }
    }
    f[o + 40] = bits as f32;
    for i in 0..6 {
        f[o + 41 + i] = ctl.map(|a| net_num(a[i]) as f32).unwrap_or(0.0);
    }
}

pub fn net_read_car(
    f: &[f32],
    o: usize,
    c: &mut Car,
    tick: u64,
    ctl_out: Option<&mut Controls>,
) {
    c.pos = DVec3::new(f[o] as f64, f[o + 1] as f64, f[o + 2] as f64);
    c.vel = DVec3::new(f[o + 3] as f64, f[o + 4] as f64, f[o + 5] as f64);
    let q = DQuat::from_xyzw(f[o + 6] as f64, f[o + 7] as f64, f[o + 8] as f64, f[o + 9] as f64);
    c.quat = if q.length_squared() > 1e-6 {
        q.normalize_or_identity()
    } else {
        DQuat::IDENTITY
    };
    c.ang_vel = DVec3::new(f[o + 10] as f64, f[o + 11] as f64, f[o + 12] as f64);
    c.boost = f[o + 13] as f64;
    c.ground_lock = f[o + 14] as f64;
    c.contacts = f[o + 15] as u8;
    c.ground_normal = DVec3::new(f[o + 16] as f64, f[o + 17] as f64, f[o + 18] as f64);
    c.jump_hold_t = f[o + 19] as f64;
    c.time_since_jump = f[o + 20] as f64;
    c.flip_t = f[o + 21] as f64;
    c.flip_axis = DVec3::new(f[o + 22] as f64, f[o + 23] as f64, f[o + 24] as f64);
    c.coyote_t = f[o + 25] as f64;
    c.air_time = f[o + 26] as f64;
    c.body_contact_t = f[o + 27] as f64;
    c.boost_t = f[o + 28] as f64;
    c.sliding = f[o + 29] as f64;
    c.traction = f[o + 30] as f64;
    c.respawn_t = f[o + 31] as f64;
    c.blast_t = f[o + 32] as f64;
    c.speed_mul = f[o + 33] as f64;
    c.nitro_t = f[o + 34] as f64;
    c.last_ball_touch_tick = tick as i64 - f[o + 35] as i64;

    let bits = f[o + 40] as i32;
    c.prev_jump = (bits & (1 << 0)) != 0;
    c.on_ground = (bits & (1 << 1)) != 0;
    c.jumping = (bits & (1 << 2)) != 0;
    c.jumped = (bits & (1 << 3)) != 0;
    c.has_flip = (bits & (1 << 4)) != 0;
    c.double_jumped = (bits & (1 << 5)) != 0;
    c.flipping = (bits & (1 << 6)) != 0;
    c.body_contact = (bits & (1 << 7)) != 0;
    c.is_boosting = (bits & (1 << 8)) != 0;
    c.supersonic = (bits & (1 << 9)) != 0;
    c.was_supersonic = (bits & (1 << 10)) != 0;
    c.handbrake = (bits & (1 << 11)) != 0;
    c.demolished = (bits & (1 << 12)) != 0;
    for (i, w) in c.wheels.iter_mut().enumerate() {
        w.len = f[o + 36 + i] as f64;
        w.contact = (bits & (1 << (13 + i))) != 0;
    }
    if let Some(out) = ctl_out {
        let arr = [
            f[o + 41] as f64,
            f[o + 42] as f64,
            f[o + 43] as f64,
            f[o + 44] as f64,
            f[o + 45] as f64,
            f[o + 46] as f64,
        ];
        net_unpack_controls(&arr, out);
    }
    c.basis = Basis::from_quat(c.quat);
}

pub fn net_write_ball(f: &mut [f32], o: usize, b: &Ball) {
    f[o] = b.pos.x as f32;
    f[o + 1] = b.pos.y as f32;
    f[o + 2] = b.pos.z as f32;
    f[o + 3] = b.vel.x as f32;
    f[o + 4] = b.vel.y as f32;
    f[o + 5] = b.vel.z as f32;
    f[o + 6] = b.ang_vel.x as f32;
    f[o + 7] = b.ang_vel.y as f32;
    f[o + 8] = b.ang_vel.z as f32;
    f[o + 9] = b.quat.x as f32;
    f[o + 10] = b.quat.y as f32;
    f[o + 11] = b.quat.z as f32;
    f[o + 12] = b.quat.w as f32;
    f[o + 13] = net_num(b.g_scale) as f32;
    f[o + 14] = net_num(b.ice_t) as f32;
    f[o + 15] = net_num(b.anti_t) as f32;
    f[o + 16] = b.stuck_to.map(|i| i as f32).unwrap_or(-1.0);
    f[o + 17] = b.stuck_local.x as f32;
    f[o + 18] = b.stuck_local.y as f32;
    f[o + 19] = b.stuck_local.z as f32;
    f[o + 20] = b.last_touch.map(|i| i as f32).unwrap_or(-1.0);
    f[o + 21] = (u32::from(b.frozen) | (u32::from(b.on_ground) << 1)) as f32;
}

pub fn net_read_ball(f: &[f32], o: usize, b: &mut Ball, n_cars: usize) {
    b.pos = DVec3::new(f[o] as f64, f[o + 1] as f64, f[o + 2] as f64);
    b.vel = DVec3::new(f[o + 3] as f64, f[o + 4] as f64, f[o + 5] as f64);
    b.ang_vel = DVec3::new(f[o + 6] as f64, f[o + 7] as f64, f[o + 8] as f64);
    let q = DQuat::from_xyzw(f[o + 9] as f64, f[o + 10] as f64, f[o + 11] as f64, f[o + 12] as f64);
    b.quat = if q.length_squared() > 1e-6 {
        q.normalize_or_identity()
    } else {
        DQuat::IDENTITY
    };
    b.g_scale = f[o + 13] as f64;
    b.ice_t = f[o + 14] as f64;
    b.anti_t = f[o + 15] as f64;
    let st = f[o + 16] as i32;
    b.stuck_to = if st >= 0 && (st as usize) < n_cars {
        Some(st as usize)
    } else {
        None
    };
    b.stuck_local = DVec3::new(f[o + 17] as f64, f[o + 18] as f64, f[o + 19] as f64);
    let lt = f[o + 20] as i32;
    if lt >= 0 && (lt as usize) < n_cars {
        b.last_touch = Some(lt as usize);
    }
    let fl = f[o + 21] as i32;
    b.frozen = (fl & 1) != 0;
    b.on_ground = (fl & 2) != 0;
}

pub struct SnapshotInfo<'a> {
    pub state: &'a str,
    pub overtime: bool,
    pub cd: Option<u16>,
    pub time_left: f64,
    pub ot_time: f64,
    pub score: [u8; 2],
    pub ctls: &'a [[f64; 6]],
    pub meta: Option<&'a str>,
}

/// Byte-for-byte compatible with JS `netEncodeSnapshot(w, info)` in `10c_netcore.js`.
pub fn net_encode_snapshot(w: &World, info: &SnapshotInfo<'_>) -> Vec<u8> {
    let n_cars = w.cars.len();
    let n_pads = w.pads.len();
    let pad_bytes = (n_pads + 7) >> 3;
    let f_off = (32 + pad_bytes + 3) & !3;
    let n_f = NET_BALL_N + n_cars * NET_CAR_N;
    let meta_bytes = info.meta.map(|s| s.as_bytes());
    let meta_len = meta_bytes.map(|b| b.len()).unwrap_or(0);
    let meta_off = f_off + n_f * 4;

    let mut buf = vec![0_u8; meta_off + 4 + meta_len];
    buf[0] = NET_MAGIC;
    buf[1] = NET_VERSION;
    buf[2] = NET_STATES
        .iter()
        .position(|&s| s == info.state)
        .unwrap_or(0) as u8;
    buf[3] = u8::from(w.frozen) | (u8::from(info.overtime) << 1);
    buf[4..8].copy_from_slice(&(w.tick as u32).to_le_bytes());
    // bytes 8..18 are zero-initialized (ack, depth, starves) and patched per-recipient
    let cd_val = match info.cd {
        None => NET_NO_CD,
        Some(v) => v.min(NET_NO_CD - 1),
    };
    buf[18..20].copy_from_slice(&cd_val.to_le_bytes());
    buf[20..24].copy_from_slice(&(net_num(info.time_left) as f32).to_le_bytes());
    buf[24..28].copy_from_slice(&(net_num(info.ot_time) as f32).to_le_bytes());
    buf[28] = info.score[0];
    buf[29] = info.score[1];
    buf[30] = n_cars as u8;
    buf[31] = n_pads as u8;

    for (i, p) in w.pads.iter().enumerate() {
        if p.active {
            buf[32 + (i >> 3)] |= 1 << (i & 7);
        }
    }

    let mut f = vec![0.0_f32; n_f];
    net_write_ball(&mut f, 0, &w.ball);
    for (i, c) in w.cars.iter().enumerate() {
        net_write_car(&mut f, NET_BALL_N + i * NET_CAR_N, c, info.ctls.get(i), w.tick);
    }
    for (i, &val) in f.iter().enumerate() {
        let off = f_off + i * 4;
        buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    buf[meta_off..meta_off + 4].copy_from_slice(&(meta_len as u32).to_le_bytes());
    if let Some(mb) = meta_bytes {
        buf[meta_off + 4..meta_off + 4 + meta_len].copy_from_slice(mb);
    }
    buf
}

/// Byte-for-byte compatible with JS `netPatchSnapshot(buf, ack, depth, starves)`.
pub fn net_patch_snapshot(buf: &mut [u8], ack: u32, depth: f32, starves: u32) {
    if buf.len() < 18 {
        return;
    }
    buf[8..12].copy_from_slice(&ack.to_le_bytes());
    buf[12..16].copy_from_slice(&depth.to_le_bytes());
    buf[16..18].copy_from_slice(&(starves.min(0xFFFF) as u16).to_le_bytes());
}

#[derive(Clone, Debug)]
pub struct DecodedSnapshot {
    pub state: &'static str,
    pub frozen: bool,
    pub overtime: bool,
    pub tick: u32,
    pub ack: u32,
    pub depth: f32,
    pub starves: u16,
    pub cd: i32,
    pub time_left: f32,
    pub ot_time: f32,
    pub score: [u8; 2],
    pub n_cars: usize,
    pub n_pads: usize,
    pub pads: Vec<u8>,
    pub f: Vec<f32>,
    pub meta: Option<serde_json::Value>,
}

/// Compatible with JS `netDecodeSnapshot(buf)`.
pub fn net_decode_snapshot(buf: &[u8]) -> Option<DecodedSnapshot> {
    if buf.len() < 36 || buf[0] != NET_MAGIC || buf[1] != NET_VERSION {
        return None;
    }
    let n_cars = buf[30] as usize;
    let n_pads = buf[31] as usize;
    let pad_bytes = (n_pads + 7) >> 3;
    let f_off = (32 + pad_bytes + 3) & !3;
    let n_f = NET_BALL_N + n_cars * NET_CAR_N;
    let meta_off = f_off + n_f * 4;
    if buf.len() < meta_off + 4 {
        return None;
    }

    let meta_len = u32::from_le_bytes(buf[meta_off..meta_off + 4].try_into().ok()?) as usize;
    let meta = if meta_len > 0 && meta_off + 4 + meta_len <= buf.len() {
        std::str::from_utf8(&buf[meta_off + 4..meta_off + 4 + meta_len])
            .ok()
            .and_then(|s| serde_json::from_str(s).ok())
    } else {
        None
    };

    let fl = buf[3];
    let cd_raw = u16::from_le_bytes(buf[18..20].try_into().ok()?);
    let mut f = Vec::with_capacity(n_f);
    for i in 0..n_f {
        let off = f_off + i * 4;
        f.push(f32::from_le_bytes(buf[off..off + 4].try_into().ok()?));
    }

    Some(DecodedSnapshot {
        state: NET_STATES.get(buf[2] as usize).copied().unwrap_or("idle"),
        frozen: (fl & 1) != 0,
        overtime: (fl & 2) != 0,
        tick: u32::from_le_bytes(buf[4..8].try_into().ok()?),
        ack: u32::from_le_bytes(buf[8..12].try_into().ok()?),
        depth: f32::from_le_bytes(buf[12..16].try_into().ok()?),
        starves: u16::from_le_bytes(buf[16..18].try_into().ok()?),
        cd: if cd_raw == NET_NO_CD {
            -1
        } else {
            cd_raw as i32
        },
        time_left: f32::from_le_bytes(buf[20..24].try_into().ok()?),
        ot_time: f32::from_le_bytes(buf[24..28].try_into().ok()?),
        score: [buf[28], buf[29]],
        n_cars,
        n_pads,
        pads: buf[32..32 + pad_bytes].to_vec(),
        f,
        meta,
    })
}
