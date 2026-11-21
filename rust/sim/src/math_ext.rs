use std::f64::consts::FRAC_PI_2;

use glam::{DQuat, DVec3};
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const PI: f64 = std::f64::consts::PI;
pub const TAU: f64 = std::f64::consts::TAU;

#[inline]
pub fn clamp(v: f64, a: f64, b: f64) -> f64 {
    if v < a {
        a
    } else if v > b {
        b
    } else {
        v
    }
}

#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[inline]
pub fn sstep(a: f64, b: f64, x: f64) -> f64 {
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn damp(a: f64, b: f64, lambda: f64, dt: f64) -> f64 {
    lerp(a, b, 1.0 - (-lambda * dt).exp())
}

/// Matches JS `const sign = (x) => x < 0 ? -1 : 1;` (returns +1 for 0.0).
#[inline]
pub fn sign(x: f64) -> f64 {
    if x < 0.0 { -1.0 } else { 1.0 }
}

/// Matches JS `Math.sign(x)`: returns 0.0 for 0.0, -1.0 for negative, 1.0 for positive.
#[inline]
pub fn math_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline]
pub fn yaw_toward(x: f64, z: f64, tx: f64, tz: f64) -> f64 {
    let dx = tx - x;
    let dz = tz - z;
    let l = dx.hypot(dz);
    let l = if l == 0.0 { 1.0 } else { l };
    (-dz / l).atan2(dx / l)
}

#[inline]
pub fn pick<'a, T, R: Rng + ?Sized>(arr: &'a [T], rng: &mut R) -> &'a T {
    &arr[rng.gen_range(0..arr.len())]
}

#[inline]
pub fn rand_range<R: Rng + ?Sized>(rng: &mut R, a: f64, b: f64) -> f64 {
    a + rng.r#gen::<f64>() * (b - a)
}

pub fn curve(pts: &[[f64; 2]], x: f64) -> f64 {
    if x <= pts[0][0] {
        return pts[0][1];
    }
    for i in 1..pts.len() {
        if x <= pts[i][0] {
            let a = pts[i - 1];
            let b = pts[i];
            return lerp(a[1], b[1], (x - a[0]) / (b[0] - a[0]));
        }
    }
    pts[pts.len() - 1][1]
}

pub const THROTTLE_CURVE: [[f64; 2]; 4] = [
    [0.0, 16.0],
    [14.0, 1.6],
    [14.1, 0.0],
    [30.0, 0.0],
];

pub const STEER_CURVE: [[f64; 2]; 6] = [
    [0.0, 0.69],
    [5.0, 0.398],
    [10.0, 0.235],
    [15.0, 0.1375],
    [17.5, 0.11],
    [23.0, 0.088],
];

pub const EXTRA_IMPULSE_CURVE: [[f64; 2]; 4] = [
    [0.0, 0.65],
    [5.0, 0.65],
    [23.0, 0.55],
    [46.0, 0.30],
];

pub const BUMP_CURVE: [[f64; 2]; 4] = [
    [0.0, 0.0],
    [14.0, 11.0],
    [22.0, 15.3],
    [46.0, 18.0],
];

pub const BUMP_UP_CURVE: [[f64; 2]; 4] = [
    [0.0, 0.0],
    [14.0, 2.0],
    [22.0, 3.2],
    [46.0, 4.0],
];

/// Three.js-compatible helpers on `glam::DVec3`.
pub trait ThreeVec3Ext {
    /// Matches `THREE.Vector3.normalize()`: returns `DVec3::ZERO` when length is 0.
    fn three_normalize(self) -> DVec3;
    /// Matches `THREE.Vector3.setLength(len)`: returns `DVec3::ZERO` on zero vector.
    fn with_length(self, len: f64) -> DVec3;
    /// Matches `THREE.Vector3.angleTo(v)`: returns `FRAC_PI_2` if either vector is zero,
    /// and clamps cosine to `[-1.0, 1.0]` before `acos`.
    fn three_angle_to(self, rhs: DVec3) -> f64;
    /// Matches `THREE.Vector3.clampLength(min, max)`.
    fn three_clamp_length(self, min: f64, max: f64) -> DVec3;
}

impl ThreeVec3Ext for DVec3 {
    #[inline]
    fn three_normalize(self) -> DVec3 {
        let len = self.length();
        if len == 0.0 {
            DVec3::ZERO
        } else {
            self / len
        }
    }

    #[inline]
    fn with_length(self, len: f64) -> DVec3 {
        self.three_normalize() * len
    }

    #[inline]
    fn three_angle_to(self, rhs: DVec3) -> f64 {
        let denom = (self.length_squared() * rhs.length_squared()).sqrt();
        if denom == 0.0 {
            return FRAC_PI_2;
        }
        (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos()
    }

    #[inline]
    fn three_clamp_length(self, min: f64, max: f64) -> DVec3 {
        let len = self.length();
        if len == 0.0 {
            DVec3::ZERO
        } else {
            (self / len) * len.clamp(min, max)
        }
    }
}

/// Three.js-compatible helpers on `glam::DQuat`.
pub trait ThreeQuatExt {
    /// Matches `THREE.Quaternion.normalize()`: returns `DQuat::IDENTITY` if length is 0.
    fn normalize_or_identity(self) -> DQuat;
    /// Matches `quatIntegrate(q, w, dt)` in `03_physics.js`.
    fn integrate(self, omega: DVec3, dt: f64) -> DQuat;
}

impl ThreeQuatExt for DQuat {
    #[inline]
    fn normalize_or_identity(self) -> DQuat {
        let len = self.length();
        if len == 0.0 {
            DQuat::IDENTITY
        } else {
            let inv = 1.0 / len;
            DQuat::from_xyzw(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
        }
    }

    #[inline]
    fn integrate(self, w: DVec3, dt: f64) -> DQuat {
        let h = 0.5 * dt;
        DQuat::from_xyzw(
            self.x + h * (w.x * self.w + w.y * self.z - w.z * self.y),
            self.y + h * (w.y * self.w + w.z * self.x - w.x * self.z),
            self.z + h * (w.z * self.w + w.x * self.y - w.y * self.x),
            self.w + h * (-w.x * self.x - w.y * self.y - w.z * self.z),
        )
        .normalize_or_identity()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Basis {
    pub f: DVec3, // local +X (forward)
    pub u: DVec3, // local +Y (up)
    pub r: DVec3, // local +Z (right)
}

impl Default for Basis {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Basis {
    pub const IDENTITY: Self = Self {
        f: DVec3::X,
        u: DVec3::Y,
        r: DVec3::Z,
    };

    #[inline]
    pub fn from_quat(q: DQuat) -> Self {
        let (x2, y2, z2) = (q.x + q.x, q.y + q.y, q.z + q.z);
        let (xx, xy, xz) = (q.x * x2, q.x * y2, q.x * z2);
        let (yy, yz, zz) = (q.y * y2, q.y * z2, q.z * z2);
        let (wx, wy, wz) = (q.w * x2, q.w * y2, q.w * z2);
        Self {
            f: DVec3::new(1.0 - (yy + zz), xy + wz, xz - wy),
            u: DVec3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
            r: DVec3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
        }
    }

    #[inline]
    pub fn to_local(self, v: DVec3) -> DVec3 {
        DVec3::new(self.f.dot(v), self.u.dot(v), self.r.dot(v))
    }

    #[inline]
    pub fn to_world(self, l: DVec3) -> DVec3 {
        DVec3::new(
            self.f.x * l.x + self.u.x * l.y + self.r.x * l.z,
            self.f.y * l.x + self.u.y * l.y + self.r.y * l.z,
            self.f.z * l.x + self.u.z * l.y + self.r.z * l.z,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysConfig {
    pub dt: f64,
    pub g: f64,
    pub car_mass: f64,
    pub ball_mass: f64,
    pub ball_r: f64,
    pub car_max: f64,
    pub supersonic: f64,
    pub ball_max: f64,
    pub car_max_ang: f64,
    pub ball_max_ang: f64,
    pub boost_acc: f64,
    pub boost_use: f64,
    pub air_throttle: f64,
    pub brake: f64,
    pub coast: f64,
    pub jump_imp: f64,
    pub jump_hold: f64,
    pub jump_hold_t: f64,
    pub dodge_window: f64,
    pub flip_time: f64,
    pub flip_rate: f64,
    pub susp_rest: f64,
    pub susp_max: f64,
    pub susp_k: f64,
    pub susp_c: f64,
    pub susp_grace: f64,
    pub sticky: f64,
    pub ball_drag: f64,
    pub ball_rest: f64,
    pub ball_fric: f64,
    pub ball_magnus: f64,
    pub t_roll: f64,
    pub d_roll: f64,
    pub t_pitch: f64,
    pub d_pitch: f64,
    pub t_yaw: f64,
    pub d_yaw: f64,
}

impl Default for PhysConfig {
    fn default() -> Self {
        Self {
            dt: 1.0 / 120.0,
            g: 6.5,
            car_mass: 180.0,
            ball_mass: 30.0,
            ball_r: 0.9275,
            car_max: 23.0,
            supersonic: 22.0,
            ball_max: 60.0,
            car_max_ang: 5.5,
            ball_max_ang: 6.0,
            boost_acc: 9.9167,
            boost_use: 33.3,
            air_throttle: 0.6667,
            brake: 35.0,
            coast: 5.25,
            jump_imp: 2.92,
            jump_hold: 14.58,
            jump_hold_t: 0.2,
            dodge_window: 1.25,
            flip_time: 0.65,
            flip_rate: 5.5,
            susp_rest: 0.18,
            susp_max: 0.25,
            susp_k: 55.0,
            susp_c: 7.0,
            susp_grace: 0.05,
            sticky: 3.25,
            ball_drag: 0.0305,
            ball_rest: 0.6,
            ball_fric: 0.35,
            ball_magnus: 0.011,
            t_roll: 36.08,
            d_roll: 4.47,
            t_pitch: 12.15,
            d_pitch: 2.80,
            t_yaw: 8.92,
            d_yaw: 1.89,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hitbox {
    pub hx: f64,
    pub hy: f64,
    pub hz: f64,
}

pub const HITBOXES: [Hitbox; 5] = [
    Hitbox { hx: 0.59, hy: 0.18, hz: 0.42 },
    Hitbox { hx: 0.64, hy: 0.15, hz: 0.42 },
    Hitbox { hx: 0.66, hy: 0.145, hz: 0.40 },
    Hitbox { hx: 0.60, hy: 0.21, hz: 0.44 },
    Hitbox { hx: 0.61, hy: 0.165, hz: 0.42 },
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Controls {
    pub throttle: f64,
    pub steer: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub roll: f64,
    pub jump: bool,
    pub boost: bool,
    pub handbrake: bool,
    pub air_roll_free: bool,
    pub item: bool,
}

#[inline]
pub fn make_controls() -> Controls {
    Controls::default()
}
