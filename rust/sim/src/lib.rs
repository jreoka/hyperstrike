pub mod ai;
pub mod arena;
pub mod math_ext;
pub mod netcode;
pub mod physics;
pub mod powers;
pub mod world;

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
}
