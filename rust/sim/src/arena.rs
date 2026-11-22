use glam::DVec3;

use crate::math_ext::{PI, TAU, clamp, lerp};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeType {
    Rect,
    Octagon,
    Hexagon,
    Round,
    Figure8,
    Standalone,
    SplitGoal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalMode {
    Standard,
    Standalone,
    Split,
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeSpec {
    pub shape_type: ShapeType,
    pub w: f64,
    pub l: f64,
    pub rc: f64,
    pub ch: Option<f64>,
    pub a: Option<f64>,
    pub n: Option<usize>,
    pub zc: Option<f64>,
    pub rl: Option<f64>,
    pub kf: Option<f64>,
    pub goal_z: Option<f64>,
    pub ch_x: Option<f64>,
    pub ch_z: Option<f64>,
    pub side_z: Option<f64>,
}

impl ShapeSpec {
    const fn rect(w: f64, l: f64, rc: f64) -> Self {
        Self {
            shape_type: ShapeType::Rect,
            w,
            l,
            rc,
            ch: None,
            a: None,
            n: None,
            zc: None,
            rl: None,
            kf: None,
            goal_z: None,
            ch_x: None,
            ch_z: None,
            side_z: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaDef {
    pub id: &'static str,
    pub name: &'static str,
    pub tag: &'static str,
    pub weird: bool,
    pub w: f64,
    pub l: f64,
    pub h: f64,
    pub rc: f64,
    pub rf: f64,
    pub gw: f64,
    pub gh: f64,
    pub gd: f64,
    pub shape: Option<ShapeSpec>,
}

pub const ARENAS: [ArenaDef; 15] = [
    ArenaDef {
        id: "neon",
        name: "NEON NEXUS",
        tag: "NIGHT CITY",
        weird: false,
        w: 40.0,
        l: 51.0,
        h: 19.0,
        rc: 10.0,
        rf: 3.6,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: None,
    },
    ArenaDef {
        id: "stadium",
        name: "CHAMPIONS FIELD",
        tag: "OUTDOOR STADIUM",
        weird: false,
        w: 41.0,
        l: 52.0,
        h: 20.0,
        rc: 11.5,
        rf: 3.2,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: None,
    },
    ArenaDef {
        id: "foundry",
        name: "FOUNDRY 9",
        tag: "INDUSTRIAL",
        weird: false,
        w: 36.0,
        l: 54.0,
        h: 18.0,
        rc: 7.0,
        rf: 4.2,
        gw: 8.6,
        gh: 6.4,
        gd: 8.4,
        shape: None,
    },
    ArenaDef {
        id: "mesa",
        name: "SUNSCORCH MESA",
        tag: "DESERT",
        weird: false,
        w: 44.0,
        l: 50.0,
        h: 19.0,
        rc: 16.0,
        rf: 3.8,
        gw: 9.2,
        gh: 6.4,
        gd: 8.8,
        shape: None,
    },
    ArenaDef {
        id: "orbit",
        name: "STARLIGHT ORBIT",
        tag: "COSMIC",
        weird: false,
        w: 40.0,
        l: 52.0,
        h: 24.0,
        rc: 13.0,
        rf: 6.5,
        gw: 8.9,
        gh: 6.6,
        gd: 8.8,
        shape: None,
    },
    ArenaDef {
        id: "frost",
        name: "FROSTBITE CITADEL",
        tag: "ARCTIC · OCTAGON",
        weird: false,
        w: 42.0,
        l: 52.0,
        h: 20.0,
        rc: 4.5,
        rf: 3.6,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Octagon,
            w: 42.0,
            l: 52.0,
            rc: 4.5,
            ch: Some(15.0),
            ..ShapeSpec::rect(42.0, 52.0, 4.5)
        }),
    },
    ArenaDef {
        id: "abyss",
        name: "ABYSSAL DOME",
        tag: "DEEP SEA · ROUND",
        weird: false,
        w: 47.0,
        l: 50.0,
        h: 21.0,
        rc: 7.0,
        rf: 4.2,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Round,
            w: 47.0,
            l: 50.0,
            rc: 7.0,
            a: Some(16.0),
            n: Some(9),
            ..ShapeSpec::rect(47.0, 50.0, 7.0)
        }),
    },
    ArenaDef {
        id: "ruins",
        name: "VERDANT RUINS",
        tag: "JUNGLE · HEXAGON",
        weird: false,
        w: 46.0,
        l: 52.0,
        h: 19.0,
        rc: 7.0,
        rf: 3.6,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Hexagon,
            w: 46.0,
            l: 52.0,
            rc: 7.0,
            a: Some(21.0),
            ..ShapeSpec::rect(46.0, 52.0, 7.0)
        }),
    },
    ArenaDef {
        id: "rift",
        name: "CINDER RIFT",
        tag: "VOLCANIC · CAPSULE",
        weird: false,
        w: 35.0,
        l: 60.0,
        h: 19.0,
        rc: 22.0,
        rf: 3.6,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec::rect(35.0, 60.0, 22.0)),
    },
    ArenaDef {
        id: "sakura",
        name: "SAKURA SHRINE",
        tag: "SPRING · TEMPLE",
        weird: false,
        w: 41.0,
        l: 51.0,
        h: 20.0,
        rc: 11.0,
        rf: 3.5,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: None,
    },
    ArenaDef {
        id: "skyline",
        name: "SKYLINE CITADEL",
        tag: "CLOUDS · FLOATING",
        weird: false,
        w: 43.0,
        l: 51.0,
        h: 22.0,
        rc: 5.5,
        rf: 3.8,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Octagon,
            w: 43.0,
            l: 51.0,
            rc: 5.5,
            ch: Some(13.0),
            ..ShapeSpec::rect(43.0, 51.0, 5.5)
        }),
    },
    ArenaDef {
        id: "synthwave",
        name: "SYNTHWAVE HORIZON",
        tag: "RETRO · OUTRUN",
        weird: false,
        w: 45.0,
        l: 52.0,
        h: 20.0,
        rc: 7.5,
        rf: 3.6,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Hexagon,
            w: 45.0,
            l: 52.0,
            rc: 7.5,
            a: Some(20.0),
            ..ShapeSpec::rect(45.0, 52.0, 7.5)
        }),
    },
    ArenaDef {
        id: "figure8",
        name: "HYPERLOOP 8",
        tag: "LABS · FIGURE-8",
        weird: true,
        w: 39.0,
        l: 62.0,
        h: 21.0,
        rc: 8.0,
        rf: 3.6,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Figure8,
            w: 39.0,
            l: 62.0,
            rc: 8.0,
            zc: Some(36.0),
            rl: Some(39.0),
            kf: Some(10.0),
            ..ShapeSpec::rect(39.0, 62.0, 8.0)
        }),
    },
    ArenaDef {
        id: "standalone",
        name: "THROWBACK CORE",
        tag: "LABS · FORWARD GOAL",
        weird: true,
        w: 42.0,
        l: 61.0,
        h: 20.0,
        rc: 14.0,
        rf: 3.6,
        gw: 8.9,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Standalone,
            w: 42.0,
            l: 61.0,
            rc: 14.0,
            goal_z: Some(39.0),
            ..ShapeSpec::rect(42.0, 61.0, 14.0)
        }),
    },
    ArenaDef {
        id: "splitgoal",
        name: "TWIN VORTEX",
        tag: "LABS · DUAL GOALS",
        weird: true,
        w: 46.0,
        l: 58.0,
        h: 20.0,
        rc: 4.5,
        rf: 3.4,
        gw: 6.8,
        gh: 6.4,
        gd: 8.8,
        shape: Some(ShapeSpec {
            shape_type: ShapeType::SplitGoal,
            w: 46.0,
            l: 58.0,
            rc: 4.5,
            ch_x: Some(31.0),
            ch_z: Some(44.0),
            side_z: Some(22.0),
            ..ShapeSpec::rect(46.0, 58.0, 4.5)
        }),
    },
];

#[derive(Clone, Copy, Debug)]
pub struct LayoutDef {
    pub id: &'static str,
    pub name: &'static str,
    pub shape: Option<ShapeSpec>,
}

pub const LAYOUTS: [LayoutDef; 6] = [
    LayoutDef {
        id: "default",
        name: "ARENA DEFAULT",
        shape: None,
    },
    LayoutDef {
        id: "standard",
        name: "STANDARD",
        shape: Some(ShapeSpec::rect(40.0, 51.0, 10.0)),
    },
    LayoutDef {
        id: "octagon",
        name: "OCTAGON",
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Octagon,
            w: 42.0,
            l: 52.0,
            rc: 4.5,
            ch: Some(15.0),
            ..ShapeSpec::rect(42.0, 52.0, 4.5)
        }),
    },
    LayoutDef {
        id: "hexagon",
        name: "HEXAGON",
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Hexagon,
            w: 46.0,
            l: 52.0,
            rc: 7.0,
            a: Some(21.0),
            ..ShapeSpec::rect(46.0, 52.0, 7.0)
        }),
    },
    LayoutDef {
        id: "round",
        name: "ROUND",
        shape: Some(ShapeSpec {
            shape_type: ShapeType::Round,
            w: 47.0,
            l: 50.0,
            rc: 7.0,
            a: Some(16.0),
            n: Some(9),
            ..ShapeSpec::rect(47.0, 50.0, 7.0)
        }),
    },
    LayoutDef {
        id: "capsule",
        name: "CAPSULE",
        shape: Some(ShapeSpec::rect(35.0, 60.0, 22.0)),
    },
];

#[derive(Clone, Debug)]
pub struct ArenaOpts {
    pub layout: String,
    pub scale: f64,
}

impl Default for ArenaOpts {
    fn default() -> Self {
        Self {
            layout: "default".to_string(),
            scale: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fig8Params {
    pub zc: f64,
    pub rl: f64,
    pub kf: f64,
    pub rc: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SplitGoalParams {
    pub gx: f64,
    pub gz: f64,
    pub gtx: f64,
    pub gtz: f64,
    pub gnx: f64,
    pub gnz: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge2D {
    pub ox: f64,
    pub oz: f64,
    pub h: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Seg2D {
    pub x: f64,
    pub z: f64,
    pub dx: f64,
    pub dz: f64,
    pub il: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PathPoint {
    pub x: f64,
    pub z: f64,
    pub nx: f64,
    pub nz: f64,
    pub s: f64,
}

#[derive(Clone, Debug)]
pub struct WallPath {
    pub pts: Vec<PathPoint>,
    pub length: f64,
}

fn shape_outline(sp: &ShapeSpec) -> Vec<[f64; 2]> {
    let w = sp.w;
    let l = sp.l;
    match sp.shape_type {
        ShapeType::Octagon => {
            let ch = sp.ch.unwrap_or(15.0);
            vec![[0.0, l], [w - ch, l], [w, l - ch], [w, 0.0]]
        }
        ShapeType::Hexagon => {
            let a = sp.a.unwrap_or(21.0);
            vec![[0.0, l], [a, l], [w, 0.0]]
        }
        ShapeType::SplitGoal => {
            let ch_x = sp.ch_x.unwrap_or(31.0);
            let ch_z = sp.ch_z.unwrap_or(44.0);
            let side_z = sp.side_z.unwrap_or(22.0);
            vec![[0.0, l], [ch_x, ch_z], [w, side_z], [w, 0.0]]
        }
        ShapeType::Round => {
            let a = sp.a.unwrap_or(16.0);
            let p0 = (a / w).acos();
            let lr = l / p0.sin();
            let n = sp.n.unwrap_or(8);
            let mut pts = Vec::with_capacity(n + 2);
            pts.push([0.0, l]);
            for i in 0..=n {
                let p = p0 * (1.0 - (i as f64) / (n as f64));
                pts.push([w * p.cos(), lr * p.sin()]);
            }
            pts
        }
        _ => vec![[0.0, l], [w, l], [w, 0.0]],
    }
}

#[inline]
pub fn sd_box(px: f64, py: f64, pz: f64, hx: f64, hy: f64, hz: f64) -> f64 {
    let qx = px.abs() - hx;
    let qy = py.abs() - hy;
    let qz = pz.abs() - hz;
    let ox = if qx > 0.0 { qx } else { 0.0 };
    let oy = if qy > 0.0 { qy } else { 0.0 };
    let oz = if qz > 0.0 { qz } else { 0.0 };
    let m = if qx > qy {
        if qx > qz { qx } else { qz }
    } else if qy > qz {
        qy
    } else {
        qz
    };
    (ox * ox + oy * oy + oz * oz).sqrt() + if m < 0.0 { m } else { 0.0 }
}

#[derive(Clone, Debug)]
pub struct ArenaShape {
    pub shape_type: ShapeType,
    pub layout: String,
    pub goal_mode: GoalMode,
    pub w: f64,
    pub l: f64,
    pub rc: f64,
    pub h: f64,
    pub rf: f64,
    pub goal_taper: f64,
    pub gw: f64,
    pub gh: f64,
    pub gd: f64,
    pub fig8: Option<Fig8Params>,
    pub goal_z: f64,
    pub gt: f64,
    pub split_goal: Option<SplitGoalParams>,
    pub outline: Vec<[f64; 2]>,
    pub edges: Vec<Edge2D>,
    pub inset: Vec<[f64; 2]>,
    pub segs: Vec<Seg2D>,
}

impl ArenaShape {
    pub fn new(d: &ArenaDef, opts: &ArenaOpts) -> Self {
        let lay = if !opts.layout.is_empty() && opts.layout != "default" {
            LAYOUTS.iter().find(|l| l.id == opts.layout)
        } else {
            None
        };
        let mut sp = d.shape.unwrap_or_else(|| ShapeSpec::rect(d.w, d.l, d.rc));
        if let Some(l_def) = lay
            && let Some(l_sp) = l_def.shape
        {
            sp = l_sp;
        }
        let k = if opts.scale > 0.0 { opts.scale } else { 1.0 };
        let goal_mode = match sp.shape_type {
            ShapeType::Standalone => GoalMode::Standalone,
            ShapeType::SplitGoal => GoalMode::Split,
            _ => GoalMode::Standard,
        };
        let w = sp.w * k;
        let l = sp.l * k;
        let rc = sp.rc * k;
        let h = d.h;
        let rf = d.rf;
        let goal_taper = rf * 1.8;
        let gw = d.gw * if k > 1.0 { 1.06 } else { 1.0 };
        let gh = d.gh;
        let gd = d.gd;

        let fig8 = if sp.shape_type == ShapeType::Figure8 {
            Some(Fig8Params {
                zc: sp.zc.unwrap_or(36.0) * k,
                rl: sp.rl.unwrap_or(39.0) * k,
                kf: sp.kf.unwrap_or(10.0) * k,
                rc,
            })
        } else {
            None
        };

        let (goal_z, gt) = if goal_mode == GoalMode::Standalone {
            (sp.goal_z.unwrap_or(39.0) * k, 0.75)
        } else {
            (0.0, 0.0)
        };

        let scaled_sp = ShapeSpec {
            shape_type: sp.shape_type,
            w,
            l,
            rc,
            ch: sp.ch.map(|v| v * k),
            a: sp.a.map(|v| v * k),
            n: sp.n,
            zc: sp.zc.map(|v| v * k),
            rl: sp.rl.map(|v| v * k),
            kf: sp.kf.map(|v| v * k),
            goal_z: sp.goal_z.map(|v| v * k),
            ch_x: sp.ch_x.map(|v| v * k),
            ch_z: sp.ch_z.map(|v| v * k),
            side_z: sp.side_z.map(|v| v * k),
        };
        let outline = shape_outline(&scaled_sp);

        let split_goal = if goal_mode == GoalMode::Split {
            let a = outline[0];
            let b = outline[1];
            let dx = b[0] - a[0];
            let dz = b[1] - a[1];
            let len = dx.hypot(dz);
            let len = if len == 0.0 { 1.0 } else { len };
            Some(SplitGoalParams {
                gx: (a[0] + b[0]) * 0.5,
                gz: (a[1] + b[1]) * 0.5,
                gtx: dx / len,
                gtz: dz / len,
                gnx: -dz / len,
                gnz: dx / len,
            })
        } else {
            None
        };

        let mut shape = Self {
            shape_type: sp.shape_type,
            layout: lay.map(|x| x.id).unwrap_or("default").to_string(),
            goal_mode,
            w,
            l,
            rc,
            h,
            rf,
            goal_taper,
            gw,
            gh,
            gd,
            fig8,
            goal_z,
            gt,
            split_goal,
            outline,
            edges: Vec::new(),
            inset: Vec::new(),
            segs: Vec::new(),
        };
        shape.build_inset();
        shape
    }

    pub fn goal_line_z(&self) -> f64 {
        match self.goal_mode {
            GoalMode::Standalone => self.goal_z,
            GoalMode::Split => self.split_goal.map(|s| s.gz).unwrap_or(self.l),
            GoalMode::Standard => self.l,
        }
    }

    pub fn kickoff_pos(&self, spot: [f64; 2], team: u8) -> (f64, f64) {
        let sx = self.w / 40.96;
        let sz = (if self.goal_mode == GoalMode::Standalone {
            self.goal_z
        } else {
            self.l
        }) / 51.2;
        let mut x = spot[0] * sx;
        let mut z = spot[1] * sz;
        if team == 1 {
            x = -x;
            z = -z;
        }
        let (vx, mut vz) = self.push_inside_xz(x, z, 4.0);
        if self.goal_mode == GoalMode::Standalone && vz.abs() > self.goal_z - 3.0 {
            vz = (if vz < 0.0 { -1.0 } else { 1.0 }) * (self.goal_z - 3.0);
        }
        (vx, vz)
    }

    fn build_inset(&mut self) {
        let v = &self.outline;
        let rc = self.rc;
        let mut edges = Vec::with_capacity(v.len().saturating_sub(1));
        for i in 0..v.len().saturating_sub(1) {
            let dx = v[i + 1][0] - v[i][0];
            let dz = v[i + 1][1] - v[i][1];
            let l = dx.hypot(dz);
            let l = if l == 0.0 { 1.0 } else { l };
            let ox = -dz / l;
            let oz = dx / l;
            edges.push(Edge2D {
                ox,
                oz,
                h: ox * v[i][0] + oz * v[i][1] - rc,
            });
        }
        let mut inset = Vec::with_capacity(edges.len() + 1);
        inset.push([0.0, edges[0].h / edges[0].oz]);
        for i in 1..edges.len() {
            let a = edges[i - 1];
            let b = edges[i];
            let det = a.ox * b.oz - a.oz * b.ox;
            inset.push([
                (a.h * b.oz - a.oz * b.h) / det,
                (a.ox * b.h - a.h * b.ox) / det,
            ]);
        }
        let e_last = edges[edges.len() - 1];
        inset.push([e_last.h / e_last.ox, 0.0]);

        let mut segs = Vec::with_capacity(inset.len().saturating_sub(1));
        for i in 0..inset.len().saturating_sub(1) {
            let dx = inset[i + 1][0] - inset[i][0];
            let dz = inset[i + 1][1] - inset[i][1];
            segs.push(Seg2D {
                x: inset[i][0],
                z: inset[i][1],
                dx,
                dz,
                il: 1.0 / (dx * dx + dz * dz).max(1e-9),
            });
        }

        self.edges = edges;
        self.inset = inset;
        self.segs = segs;
    }

    /// 2D distance from (ax, az) [first quadrant] to the wall, positive inside.
    #[inline]
    pub fn d2(&self, ax: f64, az: f64) -> f64 {
        if let Some(f) = self.fig8 {
            let d_pos = ax.hypot(az - f.zc) - f.rl;
            let d_neg = ax.hypot(az + f.zc) - f.rl;
            let h1 = (f.kf - (d_pos - d_neg).abs()).max(0.0) / f.kf;
            let d_lobes = d_pos.min(d_neg) - h1 * h1 * f.kf * 0.25;
            let d_end = az - self.l;
            let h2 = (f.rc - (d_lobes - d_end).abs()).max(0.0) / f.rc;
            let d_out = d_lobes.max(d_end) + h2 * h2 * f.rc * 0.25;
            return -d_out;
        }
        let mut mx = -1e9_f64;
        for e in &self.edges {
            let val = e.ox * ax + e.oz * az - e.h;
            if val > mx {
                mx = val;
            }
        }
        if mx <= 0.0 {
            return self.rc - mx;
        }
        let mut best = 1e18_f64;
        for s in &self.segs {
            let px = ax - s.x;
            let pz = az - s.z;
            let t = clamp((px * s.dx + pz * s.dz) * s.il, 0.0, 1.0);
            let qx = px - s.dx * t;
            let qz = pz - s.dz * t;
            let dd = qx * qx + qz * qz;
            if dd < best {
                best = dd;
            }
        }
        self.rc - best.sqrt()
    }

    /// Full closed CCW inset polygon.
    pub fn inset_loop(&self) -> Vec<[f64; 2]> {
        let q = &self.inset;
        let mut out: Vec<[f64; 2]> = Vec::new();
        let push = |out: &mut Vec<[f64; 2]>, x: f64, z: f64| {
            if let Some(last) = out.last() {
                if (last[0] - x).hypot(last[1] - z) > 1e-6 {
                    out.push([x, z]);
                }
            } else {
                out.push([x, z]);
            }
        };
        for i in (0..q.len()).rev() {
            push(&mut out, q[i][0], q[i][1]);
        }
        for pt in q {
            push(&mut out, -pt[0], pt[1]);
        }
        for i in (0..q.len()).rev() {
            push(&mut out, -q[i][0], -q[i][1]);
        }
        for pt in q {
            push(&mut out, pt[0], -pt[1]);
        }
        if out.len() >= 2
            && (out[0][0] - out[out.len() - 1][0]).hypot(out[0][1] - out[out.len() - 1][1]) < 1e-6
        {
            out.pop();
        }
        let mut res = Vec::new();
        let n = out.len();
        for i in 0..n {
            let p = out[(i + n - 1) % n];
            let v = out[i];
            let nx = out[(i + 1) % n];
            let cr = (v[0] - p[0]) * (nx[1] - v[1]) - (v[1] - p[1]) * (nx[0] - v[0]);
            if cr.abs() > 1e-7 {
                res.push(v);
            }
        }
        res
    }

    /// Wall outline path offset outward by `offset` (0 = wall base).
    pub fn path(&self, step: f64, offset: f64) -> WallPath {
        if self.fig8.is_some() {
            let m = ((160.0 / step.max(0.4)).ceil() as usize).max(48);
            let max_r = self.w + self.l + offset.abs() + 30.0;
            let mut q = Vec::with_capacity(m + 1);
            for i in 0..=m {
                let a = (i as f64 / m as f64) * (PI * 0.5);
                let (ca, sa) = (a.cos(), a.sin());
                let mut lo = 0.0;
                let mut hi = max_r;
                for _ in 0..24 {
                    let mid = (lo + hi) * 0.5;
                    if self.d2(mid * ca, mid * sa) + offset > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                q.push([lo * ca, lo * sa]);
            }
            let mut raw = Vec::new();
            for pt in &q {
                raw.push([pt[0], pt[1]]);
            }
            for i in (0..q.len().saturating_sub(1)).rev() {
                raw.push([-q[i][0], q[i][1]]);
            }
            for pt in q.iter().skip(1) {
                raw.push([-pt[0], -pt[1]]);
            }
            for i in (1..q.len().saturating_sub(1)).rev() {
                raw.push([q[i][0], -q[i][1]]);
            }
            let mut pts = Vec::with_capacity(raw.len() + 1);
            let mut s = 0.0;
            for i in 0..raw.len() {
                let [x, z] = raw[i];
                if i > 0 {
                    s += (x - raw[i - 1][0]).hypot(z - raw[i - 1][1]);
                }
                let (gx, gz) = self.grad_2d(x, z);
                pts.push(PathPoint {
                    x,
                    z,
                    nx: -gx,
                    nz: -gz,
                    s,
                });
            }
            if !raw.is_empty() {
                let last = raw[raw.len() - 1];
                s += (raw[0][0] - last[0]).hypot(raw[0][1] - last[1]);
                let mut first = pts[0];
                first.s = s;
                pts.push(first);
            }
            return WallPath { pts, length: s };
        }

        let p = self.inset_loop();
        let n = p.len();
        let r = (self.rc + offset).max(0.3);
        let mut nrm = Vec::with_capacity(n);
        for i in 0..n {
            let a = p[i];
            let b = p[(i + 1) % n];
            let dx = b[0] - a[0];
            let dz = b[1] - a[1];
            let l = dx.hypot(dz);
            let l = if l == 0.0 { 1.0 } else { l };
            nrm.push([dz / l, -dx / l]);
        }
        let mut pts = Vec::new();
        let mut s = 0.0;
        for i in 0..n {
            let v = p[i];
            let np = nrm[(i + n - 1) % n];
            let nc = nrm[i];
            let a0 = np[1].atan2(np[0]);
            let mut a1 = nc[1].atan2(nc[0]);
            while a1 < a0 - 1e-9 {
                a1 += TAU;
            }
            let al = r * (a1 - a0);
            if al > 1e-5 {
                let k = ((al / step).ceil() as usize).max(2);
                for j in 0..k {
                    let a = lerp(a0, a1, j as f64 / k as f64);
                    pts.push(PathPoint {
                        x: v[0] + a.cos() * r,
                        z: v[1] + a.sin() * r,
                        nx: a.cos(),
                        nz: a.sin(),
                        s: s + al * j as f64 / k as f64,
                    });
                }
                s += al;
            }
            let w = p[(i + 1) % n];
            let (ax, az) = (v[0] + nc[0] * r, v[1] + nc[1] * r);
            let (bx, bz) = (w[0] + nc[0] * r, w[1] + nc[1] * r);
            let len = (bx - ax).hypot(bz - az);
            let k = ((len / step).ceil() as usize).max(1);
            for j in 0..k {
                let t = j as f64 / k as f64;
                pts.push(PathPoint {
                    x: lerp(ax, bx, t),
                    z: lerp(az, bz, t),
                    nx: nc[0],
                    nz: nc[1],
                    s: s + len * t,
                });
            }
            s += len;
        }
        if !pts.is_empty() {
            let mut first = pts[0];
            first.s = s;
            pts.push(first);
        }
        WallPath { pts, length: s }
    }

    #[inline]
    pub fn grad_2d(&self, x: f64, z: f64) -> (f64, f64) {
        let h = 0.05;
        let gx = self.wall_dist_2d(x + h, z) - self.wall_dist_2d(x - h, z);
        let gz = self.wall_dist_2d(x, z + h) - self.wall_dist_2d(x, z - h);
        let l = gx.hypot(gz);
        let l = if l == 0.0 { 1.0 } else { l };
        (gx / l, gz / l)
    }

    pub fn push_inside_xz(&self, mut x: f64, mut z: f64, margin: f64) -> (f64, f64) {
        for _ in 0..6 {
            let d = self.wall_dist_2d(x, z);
            if d >= margin {
                break;
            }
            let (gx, gz) = self.grad_2d(x, z);
            x += gx * (margin - d + 0.01);
            z += gz * (margin - d + 0.01);
        }
        (x, z)
    }

    #[inline]
    pub fn push_inside(&self, mut v: DVec3, margin: f64) -> DVec3 {
        let (x, z) = self.push_inside_xz(v.x, v.z, margin);
        v.x = x;
        v.z = z;
        v
    }

    pub fn half_width_at(&self, z: f64) -> f64 {
        let mut lo = 0.0;
        let mut hi = self.w + 2.0;
        let az = z.abs();
        for _ in 0..18 {
            let m = (lo + hi) * 0.5;
            if self.d2(m, az) > 0.0 {
                lo = m;
            } else {
                hi = m;
            }
        }
        lo
    }

    #[inline]
    pub fn dist(&self, x: f64, y: f64, z: f64) -> f64 {
        let l = self.l;
        let rf = self.rf;
        let gw = self.gw;
        let gh = self.gh;
        let ax = x.abs();
        let az = z.abs();

        if self.goal_mode == GoalMode::Standalone {
            let mut d = self.shell_dist(ax, y, az);
            let gz = self.goal_z;
            let gd = self.gd;
            let gt = self.gt;
            let margin = d.max(0.0);
            if az > gz - margin
                && az < gz + gd + gt + margin
                && ax < gw + gt + margin
                && y < gh + gt + margin
            {
                let d_side = sd_box(
                    ax - (gw + gt * 0.5),
                    y - (gh + gt) * 0.5,
                    az - (gz + (gd + gt) * 0.5),
                    gt * 0.5,
                    (gh + gt) * 0.5,
                    (gd + gt) * 0.5,
                );
                let d_back = sd_box(
                    ax,
                    y - (gh + gt) * 0.5,
                    az - (gz + gd + gt * 0.5),
                    gw + gt,
                    (gh + gt) * 0.5,
                    gt * 0.5,
                );
                let d_roof = sd_box(
                    ax,
                    y - (gh + gt * 0.5),
                    az - (gz + (gd + gt) * 0.5),
                    gw + gt,
                    gt * 0.5,
                    (gd + gt) * 0.5,
                );
                if d_side < d {
                    d = d_side;
                }
                if d_back < d {
                    d = d_back;
                }
                if d_roof < d {
                    d = d_roof;
                }
            }
            return d;
        }

        if let Some(sg) = self.split_goal {
            let u = (ax - sg.gx) * sg.gtx + (az - sg.gz) * sg.gtz;
            let v = (ax - sg.gx) * sg.gnx + (az - sg.gz) * sg.gnz;
            let au = u.abs();
            if v > -(rf + 1.0) && au < gw + 2.0 && y < gh + 2.0 {
                if v > 0.0 {
                    let ex = au - gw;
                    let ey = y - gh;
                    if ex <= 0.0 && ey <= 0.0 {
                        let mut d = -ex;
                        if -ey < d {
                            d = -ey;
                        }
                        if y < d {
                            d = y;
                        }
                        let back = self.gd - v;
                        if back < d {
                            d = back;
                        }
                        return d;
                    }
                    let ox = if ex > 0.0 { ex } else { 0.0 };
                    let oy = if ey > 0.0 { ey } else { 0.0 };
                    let d_out = (ox * ox + oy * oy).sqrt();
                    return -(if d_out < v { d_out } else { v });
                }
                if au < gw && y < gh {
                    let dv = -v;
                    let lx = gw - au;
                    let ly = gh - y;
                    let dr = if lx < ly { lx } else { ly };
                    let rim = (dv * dv + dr * dr).sqrt();
                    return if y < rim { y } else { rim };
                }
            }
            return self.shell_dist(ax, y, az);
        }

        if az > l - rf - 1.0 && ax < gw + 2.0 && y < gh + 2.0 {
            if az > l {
                let ex = ax - gw;
                let ey = y - gh;
                if ex <= 0.0 && ey <= 0.0 {
                    let mut d = -ex;
                    if -ey < d {
                        d = -ey;
                    }
                    if y < d {
                        d = y;
                    }
                    let back = l + self.gd - az;
                    if back < d {
                        d = back;
                    }
                    return d;
                }
                let ox = if ex > 0.0 { ex } else { 0.0 };
                let oy = if ey > 0.0 { ey } else { 0.0 };
                let d_out = (ox * ox + oy * oy).sqrt();
                let dz = az - l;
                return -(if d_out < dz { d_out } else { dz });
            }
            if ax < gw && y < gh {
                let dz = l - az;
                let lx = gw - ax;
                let ly = gh - y;
                let dr = if lx < ly { lx } else { ly };
                let rim = (dz * dz + dr * dr).sqrt();
                return if y < rim { y } else { rim };
            }
        }
        self.shell_dist(ax, y, az)
    }

    #[inline]
    pub fn floor_fillet_r(&self, ax: f64, az: f64) -> f64 {
        if self.goal_mode == GoalMode::Standalone {
            return self.rf;
        }
        if let Some(sg) = self.split_goal {
            let v = (ax - sg.gx) * sg.gnx + (az - sg.gz) * sg.gnz;
            if v < -(self.rf + 4.0) {
                return self.rf;
            }
            let u = ((ax - sg.gx) * sg.gtx + (az - sg.gz) * sg.gtz).abs();
            let t = clamp((u - self.gw) / self.goal_taper, 0.0, 1.0);
            return self.rf * t * t * (3.0 - 2.0 * t);
        }
        let t = clamp((ax - self.gw) / self.goal_taper, 0.0, 1.0);
        self.rf * t * t * (3.0 - 2.0 * t)
    }

    #[inline]
    pub fn floor_fillet_dr(&self, ax: f64, az: f64) -> f64 {
        if self.goal_mode == GoalMode::Standalone {
            return 0.0;
        }
        if let Some(sg) = self.split_goal {
            let v = (ax - sg.gx) * sg.gnx + (az - sg.gz) * sg.gnz;
            if v < -(self.rf + 4.0) {
                return 0.0;
            }
            let u = ((ax - sg.gx) * sg.gtx + (az - sg.gz) * sg.gtz).abs();
            let t = (u - self.gw) / self.goal_taper;
            if t <= 0.0 || t >= 1.0 {
                return 0.0;
            }
            return (self.rf / self.goal_taper) * 6.0 * t * (1.0 - t);
        }
        let t = (ax - self.gw) / self.goal_taper;
        if t <= 0.0 || t >= 1.0 {
            return 0.0;
        }
        (self.rf / self.goal_taper) * 6.0 * t * (1.0 - t)
    }

    #[inline]
    pub fn shell_dist(&self, ax: f64, y: f64, az: f64) -> f64 {
        let rf_ceil = self.rf;
        let h = self.h;
        let d2 = self.d2(ax, az);
        let rf = self.floor_fillet_r(ax, az);
        if rf > 1e-4 && d2 < rf && y < rf {
            let a = rf - d2;
            let b = rf - y;
            return rf - (a * a + b * b).sqrt();
        }
        if d2 < rf_ceil && h - y < rf_ceil {
            let a = rf_ceil - d2;
            let b = rf_ceil - (h - y);
            return rf_ceil - (a * a + b * b).sqrt();
        }
        let dy = if y < h - y { y } else { h - y };
        if d2 < dy { d2 } else { dy }
    }

    #[inline]
    pub fn normal(&self, x: f64, y: f64, z: f64) -> DVec3 {
        let h = 0.01;
        let nx = self.dist(x + h, y, z) - self.dist(x - h, y, z);
        let ny = self.dist(x, y + h, z) - self.dist(x, y - h, z);
        let nz = self.dist(x, y, z + h) - self.dist(x, y, z - h);
        let l = (nx * nx + ny * ny + nz * nz).sqrt();
        let l = if l == 0.0 { 1.0 } else { l };
        DVec3::new(nx / l, ny / l, nz / l)
    }

    #[inline]
    pub fn raycast(&self, o: DVec3, d: DVec3, max_t: f64) -> f64 {
        let mut t = 0.0;
        for _ in 0..24 {
            let px = o.x + d.x * t;
            let py = o.y + d.y * t;
            let pz = o.z + d.z * t;
            let s = self.dist(px, py, pz);
            if s < 0.004 {
                let r = t + if s < 0.0 { s } else { 0.0 };
                return if r > 0.0 { r } else { 0.0 };
            }
            t += s * 0.95;
            if t > max_t {
                return -1.0;
            }
        }
        if t <= max_t {
            let s = self.dist(o.x + d.x * t, o.y + d.y * t, o.z + d.z * t);
            if s < 0.01 {
                let r = t + if s < 0.0 { s } else { 0.0 };
                return if r > 0.0 { r } else { 0.0 };
            }
        }
        -1.0
    }

    #[inline]
    pub fn wall_dist_2d(&self, x: f64, z: f64) -> f64 {
        self.d2(x.abs(), z.abs())
    }

    #[inline]
    pub fn in_goal(&self, p: DVec3, r: f64) -> i32 {
        if self.goal_mode == GoalMode::Standalone {
            let az = p.z.abs();
            if p.x.abs() < self.gw
                && p.y < self.gh
                && az > self.goal_z + r
                && az < self.goal_z + self.gd
            {
                return if p.z < 0.0 { 0 } else { 1 };
            }
            return -1;
        }
        if let Some(sg) = self.split_goal {
            let ax = p.x.abs();
            let az = p.z.abs();
            let u = (ax - sg.gx) * sg.gtx + (az - sg.gz) * sg.gtz;
            let v = (ax - sg.gx) * sg.gnx + (az - sg.gz) * sg.gnz;
            if u.abs() < self.gw && p.y < self.gh && v > r {
                return if p.z < 0.0 { 0 } else { 1 };
            }
            return -1;
        }
        if p.x.abs() < self.gw && p.y < self.gh && p.z.abs() > self.l + r {
            return if p.z < 0.0 { 0 } else { 1 };
        }
        -1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pad {
    pub pos: DVec3,
    pub big: bool,
    pub active: bool,
    pub timer: f64,
    pub radius: f64,
    pub respawn: f64,
    pub amount: f64,
    pub anim: f64,
}

pub type BoostPad = Pad;

impl Pad {
    pub fn new(x: f64, z: f64, big: bool) -> Self {
        Self {
            pos: DVec3::new(x, 0.0, z),
            big,
            active: true,
            timer: 0.0,
            radius: if big { 2.08 } else { 1.44 },
            respawn: if big { 10.0 } else { 4.0 },
            amount: if big { 100.0 } else { 12.0 },
            anim: 0.0,
        }
    }
}

pub fn make_boost_pads(shape: &ArenaShape) -> Vec<Pad> {
    let sx = (shape.w - 2.0) / 40.96;
    let sz = (if shape.goal_mode == GoalMode::Standalone {
        shape.goal_z - 1.5
    } else {
        shape.l - 1.5
    }) / 51.2;

    const BIG: [[f64; 2]; 6] = [
        [-35.84, 0.0],
        [35.84, 0.0],
        [-30.72, 40.96],
        [30.72, 40.96],
        [-30.72, -40.96],
        [30.72, -40.96],
    ];
    const SMALL: [[f64; 2]; 28] = [
        [0.0, -42.4],
        [-17.92, -41.84],
        [17.92, -41.84],
        [-9.4, -33.08],
        [9.4, -33.08],
        [0.0, -28.16],
        [-35.84, -24.84],
        [35.84, -24.84],
        [-17.88, -23.0],
        [17.88, -23.0],
        [-20.48, -10.36],
        [0.0, -10.24],
        [20.48, -10.36],
        [-10.24, 0.0],
        [10.24, 0.0],
        [-20.48, 10.36],
        [0.0, 10.24],
        [20.48, 10.36],
        [-17.88, 23.0],
        [17.88, 23.0],
        [-35.84, 24.84],
        [35.84, 24.84],
        [0.0, 28.16],
        [-9.4, 33.08],
        [9.4, 33.08],
        [-17.92, 41.84],
        [17.92, 41.84],
        [0.0, 42.4],
    ];

    let clear_goal = |(x, mut z): (f64, f64)| -> (f64, f64) {
        if shape.goal_mode != GoalMode::Standalone {
            return (x, z);
        }
        let ax = x.abs();
        let az = z.abs();
        if ax < shape.gw + shape.gt + 2.2
            && az > shape.goal_z - 2.0
            && az < shape.goal_z + shape.gd + shape.gt + 2.2
        {
            z = (if z < 0.0 { -1.0 } else { 1.0 }) * (shape.goal_z - 3.0);
        }
        (x, z)
    };

    let mut pads = Vec::with_capacity(40);
    for [x, z] in BIG {
        let (vx, vz) = clear_goal(shape.push_inside_xz(x * sx, z * sz, shape.rf + 2.4));
        pads.push(Pad::new(vx, vz, true));
    }
    for [x, z] in SMALL {
        let (vx, vz) = clear_goal(shape.push_inside_xz(x * sx, z * sz, shape.rf + 1.6));
        pads.push(Pad::new(vx, vz, false));
    }
    if shape.goal_mode == GoalMode::Standalone {
        let bz = (shape.goal_z + shape.gd + shape.gt + shape.l) * 0.5;
        for side in [-1.0_f64, 1.0] {
            pads.push(Pad::new(0.0, side * bz, false));
            pads.push(Pad::new(-11.0, side * bz, false));
            pads.push(Pad::new(11.0, side * bz, false));
        }
    }

    // Keep pads off each other after being pushed in from cut corners
    let len = pads.len();
    for i in 0..len {
        for j in (i + 1)..len {
            let dx = pads[j].pos.x - pads[i].pos.x;
            let dz = pads[j].pos.z - pads[i].pos.z;
            let d = dx.hypot(dz);
            let need = pads[i].radius + pads[j].radius + 0.6;
            if d < need && d > 1e-3 {
                let k = (need - d) / d * 0.5;
                if !pads[i].big {
                    pads[i].pos.x -= dx * k;
                    pads[i].pos.z -= dz * k;
                }
                if !pads[j].big {
                    pads[j].pos.x += dx * k;
                    pads[j].pos.z += dz * k;
                }
            }
        }
    }
    pads
}

pub const KICKOFF_SPOTS: [[f64; 2]; 5] = [
    [-20.48, -25.6],
    [20.48, -25.6],
    [-2.56, -38.4],
    [2.56, -38.4],
    [0.0, -46.08],
];

const KICKOFF_SETS_1: &[&[usize]] = &[&[0], &[1], &[2], &[3], &[4]];
const KICKOFF_SETS_2: &[&[usize]] = &[&[0, 1], &[2, 3], &[0, 4], &[1, 4], &[2, 4]];
const KICKOFF_SETS_3: &[&[usize]] = &[&[0, 1, 4], &[2, 3, 4], &[0, 1, 2], &[0, 3, 4]];
const KICKOFF_SETS_4: &[&[usize]] = &[&[0, 1, 2, 4], &[0, 1, 3, 4], &[0, 1, 2, 3]];
const KICKOFF_SETS_5: &[&[usize]] = &[&[0, 1, 2, 3, 4]];

pub const KICKOFF_SETS: [&[&[usize]]; 5] = [
    KICKOFF_SETS_1,
    KICKOFF_SETS_2,
    KICKOFF_SETS_3,
    KICKOFF_SETS_4,
    KICKOFF_SETS_5,
];

pub fn kickoff_sets_for(team_size: usize) -> &'static [&'static [usize]] {
    let idx = team_size.clamp(1, 5) - 1;
    KICKOFF_SETS[idx]
}
