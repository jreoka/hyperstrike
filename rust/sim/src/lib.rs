pub mod ai;
pub mod anticheat;
pub mod arena;
pub mod math_ext;
pub mod netcode;
pub mod physics;
pub mod policy;
pub mod powers;
pub mod strike;
pub mod world;

pub use anticheat::{AcContext, AcModel, AcSummary, AcTracker, FEATURE_NAMES as AC_FEATURE_NAMES, ShadowBot};
pub use ai::{BOT_NAMES, BotAi, BotMode, DIFFICULTY, Difficulty, PERSONALITIES, Personality, Role, SupportSlot, TeamBrain};
pub use arena::{
    ARENAS, ArenaDef, ArenaOpts, ArenaShape, BoostPad, GoalMode, KICKOFF_SETS, KICKOFF_SPOTS,
    LAYOUTS, LayoutDef, Pad, ShapeSpec, ShapeType, kickoff_sets_for, make_boost_pads, sd_box,
};
pub use math_ext::{
    BUMP_CURVE, BUMP_UP_CURVE, Basis, Controls, EXTRA_IMPULSE_CURVE, HITBOXES, Hitbox, PI,
    PhysConfig, STEER_CURVE, TAU, THROTTLE_CURVE, ThreeQuatExt, ThreeVec3Ext, clamp, curve, damp,
    lerp, make_controls, math_sign, pick, rand_range, sign, sstep, yaw_toward,
};
pub use netcode::{
    DecodedSnapshot, NET_BALL_N, NET_CAR_N, NET_MAGIC, NET_NO_CD, NET_STATES, NET_TPI, NET_VERSION,
    SnapshotInfo, net_decode_snapshot, net_encode_snapshot, net_pack_controls, net_patch_snapshot,
    net_read_ball, net_read_car, net_unpack_controls, net_write_ball, net_write_car,
};
pub use physics::{
    Ball, BallPredictor, BallState, BounceEvent, Car, CarBallHit, CarCollisionResult, CarEventKind,
    CarStats, PredSlice, TouchRecord, Wheel, ball_integrate, collide_car_ball, collide_cars,
    obb_overlap,
};
pub use policy::{
    ACT_BRANCHES, ACT_HEADS, EMBEDDED_POLICY_BYTES, OBS_DIM, POLICY_MAGIC, POLICY_TICK_SKIP,
    SavantPolicy, TOTAL_LOGITS, decode_action, embedded_savant_policy, encode_action, extract_obs,
    policy_supports_world,
};
pub use powers::{
    POWERS, PowerDef, PowerEvent, PowerShot, PowerSystem, PowerTarget, PowerTargetKind,
    SWAP_CHARGE, ShotPhase, ShotTarget, Tornado, power_by_id,
};
pub use world::{Threat, World, WorldEvent};

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;

    #[test]
    fn sim_js_smoke_test() {
        let mut w = World::with_seed(ARENAS[0], &ArenaOpts::default(), 42);
        let c1 = w.add_car(Car::new(0, 0, "T1", 0));
        let c2 = w.add_car(Car::new(1, 1, "T2", 0));
        w.cars[c1].place(-1.2, 0.35, 0.0, 0.0);
        w.cars[c1].vel = DVec3::new(20.0, 0.0, 0.0);
        w.cars[c2].place(1.2, 0.35, 0.0, PI);
        w.cars[c2].vel = DVec3::new(-20.0, 0.0, 0.0);
        w.cars[c1].bot = Some(BotAi::new_with_rng(1, 0, &mut w.rng));
        w.cars[c2].bot = Some(BotAi::new_with_rng(1, 1, &mut w.rng));

        let dt = w.phys.dt;
        for _ in 0..240 {
            w.step(dt);
        }
        assert_eq!(w.tick, 240);
        for c in &w.cars {
            assert!(c.pos.is_finite());
            assert!(c.vel.is_finite());
            assert!(c.quat.is_finite());
        }
        assert!(w.ball.pos.is_finite());
        assert!(w.ball.vel.is_finite());
    }

    #[test]
    fn all_arenas_and_layouts_step_cleanly() {
        for (ai, arena) in ARENAS.iter().enumerate() {
            let mut w = World::with_seed(*arena, &ArenaOpts::default(), ai as u64 + 1);
            let c1 = w.add_car(Car::new(0, 0, "B1", ai % HITBOXES.len()));
            let c2 = w.add_car(Car::new(1, 1, "B2", (ai + 1) % HITBOXES.len()));
            let (x1, z1) = w.shape.kickoff_pos(KICKOFF_SPOTS[0], 0);
            let (x2, z2) = w.shape.kickoff_pos(KICKOFF_SPOTS[0], 1);
            w.cars[c1].place(x1, 0.35, z1, yaw_toward(x1, z1, 0.0, 0.0));
            w.cars[c2].place(x2, 0.35, z2, yaw_toward(x2, z2, 0.0, 0.0));
            w.cars[c1].bot = Some(BotAi::new_with_rng(2, 0, &mut w.rng));
            w.cars[c2].bot = Some(BotAi::new_with_rng(2, 1, &mut w.rng));
            w.brain.choose_kickers(&w.cars, &w.ball);
            w.kickoff = true;
            let dt = w.phys.dt;
            for _ in 0..60 {
                w.step(dt);
            }
            assert!(w.ball.pos.is_finite(), "arena {} ball NaN", arena.id);
            assert!(w.cars[0].pos.is_finite(), "arena {} car0 NaN", arena.id);
            assert!(!w.pads.is_empty(), "arena {} has pads", arena.id);
            let path = w.shape.path(2.0, 0.0);
            assert!(path.length > 50.0, "arena {} path length", arena.id);
        }

        for lay in LAYOUTS.iter() {
            let opts = ArenaOpts {
                layout: lay.id.to_string(),
                scale: 1.12,
            };
            let mut w = World::with_seed(ARENAS[0], &opts, 99);
            let dt = w.phys.dt;
            for _ in 0..30 {
                w.step(dt);
            }
            assert!(w.ball.pos.is_finite(), "layout {} ball NaN", lay.id);
        }
    }

    #[test]
    fn mayhem_power_system_steps_and_triggers() {
        let mut w = World::with_seed(ARENAS[0], &ArenaOpts::default(), 777);
        w.powers = Some(PowerSystem::new(2.0));
        let c1 = w.add_car(Car::new(0, 0, "P1", 0));
        let c2 = w.add_car(Car::new(1, 1, "P2", 1));
        w.cars[c1].place(-5.0, 0.35, -10.0, yaw_toward(-5.0, -10.0, 0.0, 0.0));
        w.cars[c2].place(5.0, 0.35, 10.0, yaw_toward(5.0, 10.0, 0.0, 0.0));
        w.cars[c1].bot = Some(BotAi::new_with_rng(2, 1, &mut w.rng));
        w.cars[c2].bot = Some(BotAi::new_with_rng(2, 2, &mut w.rng));

        let dt = w.phys.dt;
        let mut saw_power_event = false;
        for _ in 0..600 {
            w.step(dt);
            if w.events.iter().any(|e| matches!(e, WorldEvent::Power(_))) {
                saw_power_event = true;
            }
        }
        assert!(saw_power_event, "expected at least one power-up event");
        assert!(w.ball.pos.is_finite());
        for c in &w.cars {
            assert!(c.pos.is_finite());
            assert!(c.vel.is_finite());
        }
    }

    #[test]
    fn binary_snapshot_roundtrip() {
        let mut w = World::with_seed(ARENAS[0], &ArenaOpts::default(), 123);
        let c1 = w.add_car(Car::new(0, 0, "A", 0));
        let c2 = w.add_car(Car::new(1, 1, "B", 2));
        w.cars[c1].place(-10.0, 0.35, -20.0, 0.5);
        w.cars[c1].vel = DVec3::new(12.5, 0.1, -3.25);
        w.cars[c1].boost = 77.5;
        w.cars[c1].supersonic = true;
        w.cars[c2].place(8.0, 0.35, 15.0, -1.2);
        w.pads[0].active = false;
        w.tick = 42;
        w.frozen = true;

        let ctls = [
            net_pack_controls(&Controls {
                throttle: 0.75,
                steer: -0.25,
                jump: true,
                boost: true,
                ..Controls::default()
            }),
            net_pack_controls(&Controls::default()),
        ];

        let mut buf = net_encode_snapshot(
            &w,
            &SnapshotInfo {
                state: "play",
                overtime: true,
                cd: Some(120),
                time_left: 245.5,
                ot_time: 12.25,
                score: [2, 1],
                ctls: &ctls,
                meta: Some(r#"{"cs":[[10,1,0,0,1,0],[0,0,0,0,0,0]]}"#),
            },
        );
        net_patch_snapshot(&mut buf, 99, 3.0, 4);

        // Reference bytes produced by JS `sim.js` (`netEncodeSnapshot` + `netPatchSnapshot`)
        // on the exact same World/Car/Ball/Pad state.
        let js_hex = "480202032a000000630000000000404004007800008075430000444102010222feffffff0300000000000000a4706d3f000000000000000000000000000000000000000000000000000000000000000000000000000000000000803f0000803f0000000000000000000080bf000000000000000000000000000080bf00000040000020c13333b33e0000a0c100004841cdcccc3d000050c00000000077577d3e00000000a50a783f00000000000000000000000000009b420000000000008040000000000000803f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000504296430b3e96430b3e96430b3e96430b3e0001f1470000403f000080be00000000000000000000000000004040000000413333b33e0000704100000000000000000000000000000080698c10bf000000803249533f000000000000000000000000000004420000000000008040000000000000803f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000504296430b3e96430b3e96430b3e96430b3e0001f047000000000000000000000000000000000000000000000000250000007b226373223a5b5b31302c312c302c302c312c305d2c5b302c302c302c302c302c305d5d7d";
        let js_bytes: Vec<u8> = (0..js_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&js_hex[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(buf.len(), js_bytes.len());
        // Header + pad bitmask (bytes 0..40) and trailing meta (bytes 504..) match byte-for-byte.
        assert_eq!(&buf[..40], &js_bytes[..40]);
        let f_end = 40 + (NET_BALL_N + 2 * NET_CAR_N) * 4;
        assert_eq!(&buf[f_end..], &js_bytes[f_end..]);
        // Every f32 payload slot matches identically (normalizing IEEE -0.0 == +0.0 from Three.js axis-angle).
        for off in (40..f_end).step_by(4) {
            let a = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            let b = f32::from_le_bytes(js_bytes[off..off + 4].try_into().unwrap());
            assert_eq!(a, b, "f32 mismatch at byte offset {off}: rust={a} js={b}");
        }

        let dec = net_decode_snapshot(&buf).expect("decode snapshot");
        assert_eq!(dec.state, "play");
        assert!(dec.frozen);
        assert!(dec.overtime);
        assert_eq!(dec.tick, 42);
        assert_eq!(dec.ack, 99);
        assert_eq!(dec.depth, 3.0);
        assert_eq!(dec.starves, 4);
        assert_eq!(dec.cd, 120);
        assert_eq!(dec.time_left, 245.5);
        assert_eq!(dec.ot_time, 12.25);
        assert_eq!(dec.score, [2, 1]);
        assert_eq!(dec.n_cars, 2);
        assert_eq!(dec.n_pads, w.pads.len());
        assert_eq!(dec.pads[0] & 1, 0);
        assert_eq!(dec.pads[0] & 2, 2);
        assert!(dec.meta.is_some());

        let mut c_out = Car::new(0, 0, "A", 0);
        let mut ctl_out = Controls::default();
        net_read_car(&dec.f, NET_BALL_N, &mut c_out, dec.tick as u64, Some(&mut ctl_out));
        assert!((c_out.pos.x - (-10.0)).abs() < 1e-4);
        assert!((c_out.boost - 77.5).abs() < 1e-4);
        assert!(c_out.supersonic);
        assert!(ctl_out.jump);
        assert!(ctl_out.boost);
        assert!((ctl_out.throttle - 0.75).abs() < 1e-4);
    }

    #[test]
    fn zero_vector_three_helpers_never_nan() {
        assert_eq!(DVec3::ZERO.three_normalize(), DVec3::ZERO);
        assert_eq!(DVec3::ZERO.with_length(5.0), DVec3::ZERO);
        assert!((DVec3::X.three_angle_to(DVec3::ZERO) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(
            glam::DQuat::from_xyzw(0.0, 0.0, 0.0, 0.0).normalize_or_identity(),
            glam::DQuat::IDENTITY
        );
    }

    #[test]
    fn all_arenas_layouts_and_scales_geometry_and_spawns_valid() {
        for arena in ARENAS.iter() {
            for lay in LAYOUTS.iter() {
                for scale in [1.0, 1.12] {
                    let opts = ArenaOpts {
                        layout: lay.id.to_string(),
                        scale,
                    };
                    let shape = ArenaShape::new(arena, &opts);

                    // 1. Kickoff spots strictly inside arena and non-overlapping per team
                    for team in [0_u8, 1_u8] {
                        let mut pts: Vec<(f64, f64)> = Vec::new();
                        for spot in KICKOFF_SPOTS {
                            let (x, z) = shape.kickoff_pos(spot, team);
                            assert!(x.is_finite() && z.is_finite());
                            let d = shape.dist(x, 0.35, z);
                            assert!(
                                d >= 0.30,
                                "kickoff spot inside wall on {} / {} / {}: d={}",
                                arena.id,
                                lay.id,
                                scale,
                                d
                            );
                            for &(px, pz) in &pts {
                                let sep = (x - px).hypot(z - pz);
                                assert!(
                                    sep > 1.5,
                                    "overlapping kickoff spots on {} / {} / {}: sep={}",
                                    arena.id,
                                    lay.id,
                                    scale,
                                    sep
                                );
                            }
                            pts.push((x, z));
                        }
                    }

                    // 2. All boost pads inside arena and clear of standalone goal walls
                    let pads = make_boost_pads(&shape);
                    assert!(!pads.is_empty());
                    for p in &pads {
                        assert!(p.pos.is_finite());
                        let d = shape.dist(p.pos.x, 0.5, p.pos.z);
                        assert!(
                            d > 0.4,
                            "pad inside wall on {} / {} / {}: pos={:?} d={}",
                            arena.id,
                            lay.id,
                            scale,
                            p.pos,
                            d
                        );
                    }

                    // 3. SDF & normals finite and unit-length across 3D sample grid
                    for ix in -2..=2 {
                        for iy in [0.5, 5.0, shape.h - 0.5] {
                            for iz in -2..=2 {
                                let x = ix as f64 * shape.w * 0.4;
                                let z = iz as f64 * shape.l * 0.4;
                                let d = shape.dist(x, iy, z);
                                let n = shape.normal(x, iy, z);
                                assert!(d.is_finite(), "non-finite dist on {}", arena.id);
                                assert!(n.is_finite(), "non-finite normal on {}", arena.id);
                                let nl = n.length();
                                assert!(
                                    (nl - 1.0).abs() < 1e-6,
                                    "non-unit normal on {}: len={}",
                                    arena.id,
                                    nl
                                );
                            }
                        }
                    }

                    // 4. Goal interiors are open (dist > 0) and trigger in_goal
                    let r = PhysConfig::default().ball_r;
                    let goal_probe = match shape.goal_mode {
                        GoalMode::Standalone => DVec3::new(0.0, 2.0, shape.goal_z + r + 1.5),
                        GoalMode::Split => {
                            let sg = shape.split_goal.unwrap();
                            DVec3::new(
                                sg.gx + sg.gnx * (r + 1.5),
                                2.0,
                                sg.gz + sg.gnz * (r + 1.5),
                            )
                        }
                        GoalMode::Standard => DVec3::new(0.0, 2.0, shape.l + r + 1.5),
                    };
                    assert!(
                        shape.dist(goal_probe.x, goal_probe.y, goal_probe.z) > r,
                        "goal interior blocked on {} / {}",
                        arena.id,
                        lay.id
                    );
                    assert_eq!(shape.in_goal(goal_probe, r), 1);
                    assert_eq!(
                        shape.in_goal(DVec3::new(goal_probe.x, goal_probe.y, -goal_probe.z), r),
                        0
                    );
                }
            }
        }
    }

    #[test]
    fn raycast_no_phantom_hits_on_grazing_rays_and_standalone_sdf_continuous() {
        let shape = ArenaShape::new(&ARENAS[0], &ArenaOpts::default());
        // Horizontal ray 0.015m above floor must NOT report a phantom hit
        let grazing = shape.raycast(DVec3::new(0.0, 0.015, 0.0), DVec3::X, 0.47);
        assert!(
            grazing < 0.0,
            "grazing parallel ray produced phantom hit t={grazing}"
        );
        // Downward ray from y=0.31 should hit floor at t≈0.31
        let hit = shape.raycast(DVec3::new(0.0, 0.31, 0.0), DVec3::NEG_Y, 0.47);
        assert!((hit - 0.31).abs() < 0.01, "expected floor hit ~0.31, got {hit}");

        // Standalone goal SDF must be continuous across the 4m distance boundary
        let standalone_def = ARENAS.iter().find(|a| a.id == "standalone").unwrap();
        let sa = ArenaShape::new(standalone_def, &ArenaOpts::default());
        let d_near = sa.dist(0.0, 6.0, sa.goal_z - 3.99);
        let d_far = sa.dist(0.0, 6.0, sa.goal_z - 4.01);
        assert!(
            (d_near - d_far).abs() < 0.05,
            "standalone SDF discontinuity: d_near={d_near} d_far={d_far}"
        );
    }

    #[test]
    fn goal_pocket_sdf_measures_back_wall_behind_net() {
        // Solid just behind a goal's back wall and above the crossbar must report its true depth
        // (through the back wall), not a shallow hit with an upward normal that pushes deeper.
        for arena in ARENAS.iter() {
            let shape = ArenaShape::new(arena, &ArenaOpts::default());
            let (base, nrm, tan) = match shape.goal_mode {
                GoalMode::Standalone => continue,
                GoalMode::Standard => (
                    DVec3::new(0.0, 0.0, shape.l + shape.gd),
                    DVec3::Z,
                    DVec3::X,
                ),
                GoalMode::Split => {
                    let sg = shape.split_goal.unwrap();
                    (
                        DVec3::new(sg.gx + sg.gnx * shape.gd, 0.0, sg.gz + sg.gnz * shape.gd),
                        DVec3::new(sg.gnx, 0.0, sg.gnz),
                        DVec3::new(sg.gtx, 0.0, sg.gtz),
                    )
                }
            };
            let behind = 0.45;
            let p = base + nrm * behind + tan * (shape.gw * 0.7) + DVec3::Y * (shape.gh + 0.02);
            let d = shape.dist(p.x, p.y, p.z);
            assert!(
                d <= -behind + 1e-6,
                "{}: point {behind}m behind back wall reports d={d}",
                arena.id
            );
            // Normal must point back toward the goal interior, not up into more solid.
            let n = shape.normal(p.x, p.y, p.z);
            assert!(n.dot(nrm) < -0.5, "{}: bad normal {n:?}", arena.id);
        }
    }

    #[test]
    fn ball_physics_rest_rolling_magnus_and_fastball_no_tunneling() {
        let shape = ArenaShape::new(&ARENAS[0], &ArenaOpts::default());
        let phys = PhysConfig::default();

        // 1. Resting stability
        let mut ball = Ball::new(&phys);
        for _ in 0..120 {
            ball.step(phys.dt, &shape, &phys);
        }
        assert!(ball.on_ground);
        assert!((ball.pos.y - phys.ball_r).abs() < 1e-4);
        assert!(ball.vel.length() < 1e-4);

        // 2. Sliding ball acquires pure rolling spin (ang_vel.z ≈ -vel.x / r) below ball_max_ang,
        //    and clamps cleanly to ball_max_ang at high speeds.
        let mut roll_ball = Ball::new(&phys);
        roll_ball.vel = DVec3::new(4.0, 0.0, 0.0);
        for _ in 0..60 {
            roll_ball.step(phys.dt, &shape, &phys);
        }
        let expected_wz = -roll_ball.vel.x / phys.ball_r;
        assert!(
            (roll_ball.ang_vel.z - expected_wz).abs() < 0.25,
            "expected rolling wz≈{expected_wz}, got {}",
            roll_ball.ang_vel.z
        );
        let mut fast_roll = Ball::new(&phys);
        fast_roll.vel = DVec3::new(15.0, 0.0, 0.0);
        for _ in 0..60 {
            fast_roll.step(phys.dt, &shape, &phys);
        }
        assert!(
            fast_roll.ang_vel.length() <= phys.ball_max_ang + 1e-6,
            "high-speed rolling spin exceeded ball_max_ang: {}",
            fast_roll.ang_vel.length()
        );

        // 3. Magnus curve: topspin (-Z) dips faster than zero spin, backspin (+Z) lifts
        let mut b_zero = Ball::new(&phys);
        let mut b_top = Ball::new(&phys);
        let mut b_back = Ball::new(&phys);
        for b in [&mut b_zero, &mut b_top, &mut b_back] {
            b.pos = DVec3::new(0.0, 8.0, 0.0);
            b.vel = DVec3::new(25.0, 0.0, 0.0);
        }
        b_top.ang_vel = DVec3::new(0.0, 0.0, -5.0);
        b_back.ang_vel = DVec3::new(0.0, 0.0, 5.0);
        for _ in 0..40 {
            b_zero.step(phys.dt, &shape, &phys);
            b_top.step(phys.dt, &shape, &phys);
            b_back.step(phys.dt, &shape, &phys);
        }
        assert!(b_top.pos.y < b_zero.pos.y - 0.05, "topspin should dip");
        assert!(b_back.pos.y > b_zero.pos.y + 0.05, "backspin should lift");

        // 4. Fastball (78 m/s) aimed at standalone goal 0.75m back wall must bounce, not tunnel
        let standalone_def = ARENAS.iter().find(|a| a.id == "standalone").unwrap();
        let sa = ArenaShape::new(standalone_def, &ArenaOpts::default());
        let mut fast_phys = phys;
        fast_phys.ball_max = 78.0;
        let outer_back_z = sa.goal_z + sa.gd + sa.gt;
        let mut fb = Ball::new(&fast_phys);
        fb.pos = DVec3::new(0.0, 3.0, outer_back_z + fast_phys.ball_r + 1.5);
        fb.vel = DVec3::new(0.0, 0.0, -78.0);
        let mut pred = BallPredictor::new();
        pred.update(&fb, &sa, &fast_phys);
        assert_eq!(
            pred.goal_team, -1,
            "fastball from behind standalone goal tunneled through back wall in BallPredictor!"
        );
        let mut bounced_positive = false;
        for _ in 0..30 {
            fb.step(fast_phys.dt, &sa, &fast_phys);
            assert_eq!(
                sa.in_goal(fb.pos, fast_phys.ball_r),
                -1,
                "fastball tunneled through standalone back wall at pos={:?}",
                fb.pos
            );
            assert!(
                fb.pos.z >= outer_back_z,
                "fastball penetrated past outer back wall z={outer_back_z}: pos={:?}",
                fb.pos
            );
            if fb.vel.z > 0.0 {
                bounced_positive = true;
            }
        }
        assert!(bounced_positive, "ball should have bounced off standalone back wall");
    }

    #[test]
    fn car_physics_equilibrium_dodge_and_turtle_recovery_wall_guard() {
        let shape = ArenaShape::new(&ARENAS[0], &ArenaOpts::default());
        let phys = PhysConfig::default();

        // 1. Resting equilibrium
        let mut car = Car::new(0, 0, "C", 0);
        car.place(0.0, 0.35, 0.0, 0.0);
        for _ in 0..120 {
            car.step(phys.dt, &shape, &phys);
        }
        assert_eq!(car.contacts, 4);
        assert!(car.on_ground);
        assert!(car.vel.length() < 0.05);
        assert!((car.pos.y - 0.35).abs() < 0.05);

        // 2. Upside-down car on floor recovers cleanly onto wheels when pressing jump, without crazy height
        let mut turtle = Car::new(1, 0, "T", 0);
        turtle.place(0.0, 0.25, 0.0, 0.0);
        turtle.quat = glam::DQuat::from_rotation_x(PI);
        turtle.contacts = 0;
        turtle.on_ground = false;
        turtle.body_contact = true;
        turtle.controls.jump = true;
        turtle.step(phys.dt, &shape, &phys);
        assert!(turtle.vel.y > 2.5, "upside-down turtle should hop upward");
        assert!(turtle.ang_vel.length() > 3.0, "turtle should spin upright");
        turtle.controls.jump = false;
        let mut max_y = turtle.pos.y;
        for _ in 0..160 {
            turtle.step(phys.dt, &shape, &phys);
            max_y = max_y.max(turtle.pos.y);
        }
        assert!(
            max_y < 1.6,
            "turtle jump recovery should stay modest in height, got max_y={max_y}"
        );
        assert!(
            turtle.basis.u.y > 0.85 && turtle.contacts >= 3,
            "turtle should land upright on wheels: u.y={} contacts={}",
            turtle.basis.u.y,
            turtle.contacts
        );

        // 2b. Intentionally upside-down car (no jump/roll) stays upside down and is not forced upright
        let mut intentional_turtle = Car::new(4, 0, "IT", 0);
        intentional_turtle.place(0.0, 0.25, 0.0, 0.0);
        intentional_turtle.quat = glam::DQuat::from_rotation_x(PI);
        intentional_turtle.basis = Basis::from_quat(intentional_turtle.quat);
        intentional_turtle.contacts = 0;
        intentional_turtle.on_ground = false;
        for _ in 0..120 {
            intentional_turtle.step(phys.dt, &shape, &phys);
        }
        assert!(
            intentional_turtle.basis.u.y < -0.8,
            "intentionally upside-down car should stay upside down without recovery input: u.y={}",
            intentional_turtle.basis.u.y
        );

        // 3. Upright car grazing a vertical wall mid-air must NOT register phantom wheel contacts
        //    or trigger infinite turtle climb; pressing jump with forward pitch should dodge cleanly.
        let mut wall_car = Car::new(2, 0, "W", 0);
        wall_car.place(shape.w - 0.5, 8.0, 0.0, 0.0);
        wall_car.contacts = 0;
        wall_car.on_ground = false;
        wall_car.body_contact = true;
        wall_car.has_flip = true;
        wall_car.jumped = true;
        wall_car.time_since_jump = 0.2;
        wall_car.controls.jump = true;
        wall_car.controls.pitch = 1.0;
        wall_car.step(phys.dt, &shape, &phys);
        assert_eq!(
            wall_car.contacts, 0,
            "upright car grazing vertical wall must not get phantom wheel contacts"
        );
        assert!(!wall_car.has_flip, "mid-air wall graze should dodge and consume flip");
        assert!(wall_car.flipping, "mid-air wall graze should enter dodge flip");

        // 4. Driving smoothly up a curved arena wall still transitions onto the vertical wall
        let mut climb = Car::new(3, 0, "Climb", 0);
        climb.place(shape.w - 12.0, 0.35, 0.0, 0.0);
        climb.controls.throttle = 1.0;
        for _ in 0..240 {
            climb.step(phys.dt, &shape, &phys);
        }
        assert!(
            climb.pos.y > 4.0 && climb.contacts >= 3,
            "car should smoothly drive onto vertical arena wall: pos={:?} contacts={}",
            climb.pos,
            climb.contacts
        );
    }

    #[test]
    fn collide_car_ball_interior_overlap_never_explodes_and_pinch_bounded() {
        let shape = ArenaShape::new(&ARENAS[0], &ArenaOpts::default());
        let phys = PhysConfig::default();

        // Exact center overlap (dist == 0) previously exploded normal by 100,000x
        let mut car = Car::new(0, 0, "A", 0);
        car.place(0.0, 1.0, 0.0, 0.0);
        car.vel = DVec3::new(10.0, 0.0, 0.0);
        let mut ball = Ball::new(&phys);
        ball.pos = car.pos;
        ball.vel = DVec3::ZERO;

        let hit = collide_car_ball(&mut car, 0, &mut ball, 10, Some(&shape), &phys);
        assert!(hit.is_some());
        assert!(ball.pos.is_finite() && car.pos.is_finite());
        assert!(ball.vel.is_finite() && car.vel.is_finite());
        assert!(
            ball.pos.length() < 5.0,
            "ball teleported on interior overlap: {:?}",
            ball.pos
        );
        assert!(
            car.pos.length() < 5.0,
            "car teleported on interior overlap: {:?}",
            car.pos
        );
        assert!(ball.vel.length() <= phys.ball_max + 1e-6);
        assert!(ball.ang_vel.length() <= phys.ball_max_ang + 1e-6);
        assert!(car.ang_vel.length() <= phys.car_max_ang + 1e-6);
    }

    #[test]
    fn collide_cars_demos_bumps_symmetry_and_stationary_guard() {
        let phys = PhysConfig::default();

        // 1. Dead-on supersonic demo (Orange car index 1 hitting stationary Blue car index 0 facing it!)
        let mut stationary = Car::new(0, 0, "Blue", 0);
        stationary.place(0.0, 0.35, 0.0, 0.0); // facing +X
        stationary.vel = DVec3::ZERO;
        let mut supersonic = Car::new(1, 1, "Orange", 0);
        supersonic.place(0.9, 0.35, 0.0, PI); // facing -X directly at stationary car
        supersonic.vel = DVec3::new(-22.5, 0.0, 0.0);
        supersonic.supersonic = true;

        let res = collide_cars(&mut stationary, 0, &mut supersonic, 1, 5.0, true, &phys);
        match res {
            Some(CarCollisionResult::Demo {
                attacker_idx,
                victim_idx,
                overdrive,
            }) => {
                assert_eq!(attacker_idx, 1, "supersonic Orange car must be attacker");
                assert_eq!(victim_idx, 0, "stationary Blue car must be victim");
                assert!(!overdrive);
            }
            other => panic!("expected Demo by Orange on Blue, got {other:?}"),
        }

        // 2. Symmetric head-on supersonic collision -> DoubleDemo
        let mut a = Car::new(0, 0, "Blue", 0);
        a.place(-0.45, 0.35, 0.0, 0.0);
        a.vel = DVec3::new(22.5, 0.0, 0.0);
        a.supersonic = true;
        let mut b = Car::new(1, 1, "Orange", 0);
        b.place(0.45, 0.35, 0.0, PI);
        b.vel = DVec3::new(-22.5, 0.0, 0.0);
        b.supersonic = true;
        let res = collide_cars(&mut a, 0, &mut b, 1, 5.0, true, &phys);
        assert!(
            matches!(
                res,
                Some(CarCollisionResult::DoubleDemo {
                    a_idx: 0,
                    b_idx: 1,
                    overdrive: false
                })
            ),
            "expected DoubleDemo on head-on supersonic, got {res:?}"
        );

        // 3. Dead-on non-supersonic bump: moving car bumps stationary facing car (not vice versa!)
        let mut still = Car::new(0, 0, "Still", 0);
        still.place(0.0, 0.35, 0.0, 0.0);
        still.vel = DVec3::ZERO;
        let mut rammer = Car::new(1, 1, "Rammer", 0);
        rammer.place(0.9, 0.35, 0.0, PI);
        rammer.vel = DVec3::new(-14.0, 0.0, 0.0);
        let res = collide_cars(&mut still, 0, &mut rammer, 1, 5.0, true, &phys);
        match res {
            Some(CarCollisionResult::Bump {
                attacker_idx,
                victim_idx,
                strength,
            }) => {
                assert_eq!(attacker_idx, 1, "moving car must be bump attacker");
                assert_eq!(victim_idx, 0, "stationary car must be bump victim");
                assert!(strength >= 13.0);
                assert!(still.vel.x < -10.0, "victim should be launched in -X: {:?}", still.vel);
                assert!(still.vel.y > 0.5, "victim should pop upward: {:?}", still.vel);
            }
            other => panic!("expected Bump by moving car, got {other:?}"),
        }
    }

    #[test]
    fn net_read_ball_clears_last_touch_when_negative() {
        let mut w = World::with_seed(ARENAS[0], &ArenaOpts::default(), 7);
        w.add_car(Car::new(0, 0, "A", 0));
        w.ball.last_touch = None;
        let mut f = vec![0.0_f32; NET_BALL_N];
        net_write_ball(&mut f, 0, &w.ball);
        let mut b_out = Ball::new(&w.phys);
        b_out.last_touch = Some(0);
        net_read_ball(&f, 0, &mut b_out, 1);
        assert_eq!(b_out.last_touch, None, "rollback snapshot -1 must clear last_touch");
    }

    #[test]
    fn powers_edge_cases_spikes_overlap_and_demo_release() {
        let mut w = World::with_seed(ARENAS[0], &ArenaOpts::default(), 99);
        let mut ps = PowerSystem::new(10.0);
        let c1 = w.add_car(Car::new(0, 0, "A", 0));
        let c2 = w.add_car(Car::new(1, 1, "B", 0));
        w.cars[c1].place(0.0, 1.0, 0.0, 0.0);
        w.cars[c1].spike_t = 5.0;
        w.ball.pos = w.cars[c1].pos; // exact overlap

        let hit = CarBallHit {
            car_idx: c1,
            strength: 5.0,
            x: 0.0,
            y: 1.0,
            z: 0.0,
            ball_speed: 0.0,
            prev_speed: 0.0,
            shot: false,
            save: false,
            epic: false,
        };
        ps.on_ball_contact(c1, &hit, false, &mut w);
        assert_eq!(w.ball.stuck_to, Some(c1));
        assert!(
            w.ball.stuck_local.length() >= w.cars[c1].half.x + 0.74,
            "spikes attach on zero offset must push ball outside car: {:?}",
            w.ball.stuck_local
        );

        // Demolishing carrier releases ball cleanly
        w.powers = Some(ps);
        w.demolish(c1, c2, false);
        assert_eq!(w.ball.stuck_to, None);
    }

    #[test]
    fn savant_standalone_goal_escape_does_not_wedge() {
        // Throwback Core: SAVANT car behind the goal must route cleanly around the box to midfield
        let standalone = ARENAS.iter().find(|a| a.id == "standalone").unwrap();
        let mut w = World::with_seed(*standalone, &ArenaOpts::default(), 123);
        let ci = w.add_car(Car::new(0, 0, "Savant", 0));
        w.cars[ci].bot = Some(BotAi::new_with_rng(5, 0, &mut w.rng));
        w.cars[ci].place(0.0, 0.35, 54.0, -PI * 0.5); // directly behind +Z goal facing -Z back wall
        w.cars[ci].boost = 50.0;
        w.ball.reset(0.0, w.phys.ball_r, -10.0); // opposite half -> same_end == false!
        w.kickoff = false;
        let dt = w.phys.dt;
        for _ in 0..480 {
            w.step(dt);
        }
        assert!(
            w.cars[ci].pos.z < 30.0,
            "SAVANT got stuck behind Throwback Core goal: pos={:?}",
            w.cars[ci].pos
        );
    }

    #[test]
    #[ignore = "multi-match benchmark; run with `cargo test --release -- --ignored --nocapture`"]
    fn bench_savant_experimental_arenas_and_mayhem() {
        let run_matches = |arena_id: &str, mayhem: bool, team_size: usize, matches: usize| -> (u32, u32) {
            let def = *ARENAS.iter().find(|a| a.id == arena_id).unwrap();
            let mut g_sav = 0u32;
            let mut g_leg = 0u32;
            for m in 0..matches {
                let mut w = World::with_seed(def, &ArenaOpts::default(), 500 + m as u64);
                if mayhem {
                    w.powers = Some(PowerSystem::new(8.0));
                }
                let mut id = 0;
                for k in 0..team_size {
                    let c0 = w.add_car(Car::new(id, 0, format!("S{k}"), 0));
                    w.cars[c0].bot = Some(BotAi::new_with_rng(5, k % PERSONALITIES.len(), &mut w.rng));
                    id += 1;
                    let c1 = w.add_car(Car::new(id, 1, format!("L{k}"), 0));
                    w.cars[c1].bot = Some(BotAi::new_with_rng(4, k % PERSONALITIES.len(), &mut w.rng));
                    id += 1;
                }
                let reset_ko = |w: &mut World| {
                    let r = w.phys.ball_r;
                    w.ball.reset(0.0, r, 0.0);
                    w.ball.frozen = false;
                    let sets = kickoff_sets_for(team_size);
                    let ord: Vec<usize> = sets[0].to_vec();
                    for team in [0u8, 1u8] {
                        let idxs: Vec<usize> = (0..w.cars.len()).filter(|&i| w.cars[i].team == team).collect();
                        for (i, ci) in idxs.into_iter().enumerate() {
                            let (x, z) = w.shape.kickoff_pos(KICKOFF_SPOTS[ord[i % ord.len()]], team);
                            let yaw = yaw_toward(x, z, 0.0, 0.0);
                            w.cars[ci].place(x, 0.35, z, yaw);
                            w.cars[ci].boost = 33.0;
                            w.cars[ci].demolished = false;
                            if let Some(b) = w.cars[ci].bot.as_mut() {
                                b.reset();
                            }
                        }
                    }
                    w.reset_pads();
                    w.kickoff = true;
                    w.brain.choose_kickers(&w.cars, &w.ball);
                };
                reset_ko(&mut w);
                let dt = w.phys.dt;
                let mut since_ko = 0.0;
                for _ in 0..(120 * 120) {
                    w.step(dt);
                    since_ko += dt;
                    if w.kickoff && (w.ball.last_touch.is_some() || since_ko > 5.0) {
                        w.kickoff = false;
                    }
                    let g = w.shape.in_goal(w.ball.pos, w.phys.ball_r);
                    if g >= 0 {
                        if g == 1 {
                            g_sav += 1;
                        } else {
                            g_leg += 1;
                        }
                        reset_ko(&mut w);
                        since_ko = 0.0;
                    }
                }
            }
            (g_sav, g_leg)
        };

        let (s_tb, l_tb) = run_matches("standalone", false, 1, 3);
        let (s_tv, l_tv) = run_matches("splitgoal", false, 1, 3);
        let (s_f8, l_f8) = run_matches("figure8", false, 1, 2);
        let (s_my, l_my) = run_matches("neon", true, 2, 2);
        println!("Throwback Core 1v1 SAVANT vs LEGEND: {s_tb}-{l_tb}");
        println!("Twin Vortex    1v1 SAVANT vs LEGEND: {s_tv}-{l_tv}");
        println!("Hyperloop 8    1v1 SAVANT vs LEGEND: {s_f8}-{l_f8}");
        println!("Mayhem Neon    2v2 SAVANT vs LEGEND: {s_my}-{l_my}");
        assert!(s_tb > l_tb, "SAVANT should beat LEGEND on Throwback Core: {s_tb}-{l_tb}");
        assert!(s_tv > l_tv, "SAVANT should beat LEGEND on Twin Vortex: {s_tv}-{l_tv}");
        assert!(s_f8 > l_f8, "SAVANT should beat LEGEND on Hyperloop 8: {s_f8}-{l_f8}");
        assert!(s_my > l_my, "SAVANT should beat LEGEND in Mayhem 2v2: {s_my}-{l_my}");
    }
}
