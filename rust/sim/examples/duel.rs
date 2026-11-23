//! Headless bot-vs-bot harness for tuning AI strength.
//! Usage: cargo run --release --example duel -- <diffA> <diffB> [teamSize] [matches] [secondsPerMatch] [arenaIdx]
//! Team A plays as team 0 in even matches and team 1 in odd matches.
use sim::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let b: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let matches: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
    let secs: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(300.0);
    let arena: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);

    let handles: Vec<_> = (0..matches)
        .map(|m| {
            std::thread::spawn(move || {
                let swap = m % 2 == 1;
                let (d0, d1) = if swap { (b, a) } else { (a, b) };
                let score = play(d0, d1, n, secs, arena, 1000 + m as u64);
                if swap { [score[1], score[0]] } else { score }
            })
        })
        .collect();
    let mut tot = [0, 0];
    let (mut wa, mut wb) = (0, 0);
    for h in handles {
        let s = h.join().unwrap();
        tot[0] += s[0];
        tot[1] += s[1];
        if s[0] > s[1] {
            wa += 1
        } else if s[1] > s[0] {
            wb += 1
        }
        println!("A {} - {} B", s[0], s[1]);
    }
    let g = STATS.lock().unwrap();
    {
        let hq = HQ.lock().unwrap();
        for l in 0..2 {
            let n = hq[l].len().max(1) as f64;
            let mq: f64 = hq[l].iter().map(|x| x.0).sum::<f64>() / n;
            let ms: f64 = hq[l].iter().map(|x| x.1).sum::<f64>() / n;
            let good = hq[l].iter().filter(|x| x.0 > 0.7).count() as f64 / n;
            let bad = hq[l].iter().filter(|x| x.0 < -0.3).count() as f64 / n;
            println!(
                "{} hitDir {:.3} speed {:.1} good {:.2} bad {:.2}",
                if l == 0 { "A" } else { "B" },
                mq,
                ms,
                good,
                bad
            );
        }
    }
    {
        let kz = KZ.lock().unwrap();
        if !kz.is_empty() {
            let m: f64 = kz.iter().sum::<f64>() / kz.len() as f64;
            let pos = kz.iter().filter(|z| **z > 0.0).count();
            println!(
                "kick n={} meanZ(A-attack)={:.2} ballInBhalf={:.3}",
                kz.len(),
                m,
                pos as f64 / kz.len() as f64
            );
        }
    }
    println!(
        "A hits {} shots {} owngoals {} kickoffGoals {} airHits {}",
        g[0][0], g[0][1], g[0][2], g[0][3], g[0][5]
    );
    println!(
        "B hits {} shots {} owngoals {} kickoffGoals {} airHits {}",
        g[1][0], g[1][1], g[1][2], g[1][3], g[1][5]
    );
    println!(
        "TOTAL diff{} {} - {} diff{}  (wins {}-{})",
        a, tot[0], tot[1], b, wa, wb
    );
}

fn kickoff(w: &mut World, n: usize) {
    let r = w.phys.ball_r;
    w.ball.reset(0.0, r, 0.0);
    w.ball.frozen = false;
    let sets = kickoff_sets_for(n);
    let order: Vec<usize> = pick(sets, &mut w.rng).to_vec();
    for team in [0u8, 1u8] {
        let idx: Vec<usize> = (0..w.cars.len())
            .filter(|&i| w.cars[i].team == team)
            .collect();
        for (i, ci) in idx.into_iter().enumerate() {
            let spot = KICKOFF_SPOTS[order[i % order.len()]];
            let (x, z) = w.shape.kickoff_pos(spot, team);
            let yaw = yaw_toward(x, z, 0.0, 0.0);
            let car = &mut w.cars[ci];
            car.place(x, 0.35, z, yaw);
            car.boost = 33.0;
            car.demolished = false;
            if let Some(bot) = car.bot.as_mut() {
                bot.reset();
            }
        }
    }
    w.reset_pads();
    w.kickoff = true;
    w.brain.choose_kickers(&w.cars, &w.ball);
}

pub static KZ: std::sync::Mutex<Vec<f64>> = std::sync::Mutex::new(Vec::new());
pub static HQ: std::sync::Mutex<[Vec<(f64, f64)>; 2]> =
    std::sync::Mutex::new([Vec::new(), Vec::new()]);
pub static STATS: std::sync::Mutex<[[u64; 6]; 2]> = std::sync::Mutex::new([[0; 6]; 2]);
// per team-letter (0=A,1=B): hits, shots, own_goals_conceded_by_own_touch, kickoff_goals_scored, goals_scored_from_ownhalf_touch, airborne_hits
fn play(d0: usize, d1: usize, n: usize, secs: f64, arena: usize, seed: u64) -> [i32; 2] {
    let a_team: u8 = if seed % 2 == 0 { 0 } else { 1 };
    let letter = |team: u8| if team == a_team { 0usize } else { 1usize };
    let mut st = [[0u64; 6]; 2];
    let kick_mode = std::env::var("KICK").is_ok();
    let kick_t: f64 = std::env::var("KICK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6.0);
    let mut w = World::with_seed(ARENAS[arena], &ArenaOpts::default(), seed);
    let mut id = 0;
    for team in [0u8, 1u8] {
        for k in 0..n {
            let ci = w.add_car(Car::new(id, team, format!("B{id}"), 0));
            let d = if team == 0 { d0 } else { d1 };
            let mut bot = BotAi::new_with_rng(d, k % PERSONALITIES.len(), &mut w.rng);
            let is_a = (team == 0) == (seed % 2 == 0);
            apply_env(&mut bot, if is_a { "A" } else { "B" });
            w.cars[ci].bot = Some(bot);
            id += 1;
        }
    }
    kickoff(&mut w, n);
    let dt = w.phys.dt;
    let mut score = [0, 0];
    let mut t = 0.0;
    let mut since = 0.0;
    while t < secs {
        w.step(dt);
        if std::env::var("TRACE").is_ok() && t < 3.0 && (w.tick % 12 == 0) {
            for c in &w.cars {
                print!(
                    "t{} z={:6.2} x={:5.2} y={:4.2} v={:5.2} b={:4.1} g={} | ",
                    c.team,
                    c.pos.z,
                    c.pos.x,
                    c.pos.y,
                    c.speed(),
                    c.boost,
                    c.on_ground as u8
                );
            }
            println!(
                "ball z={:5.2} y={:4.2} v={:5.2}",
                w.ball.pos.z,
                w.ball.pos.y,
                w.ball.vel.length()
            );
        }
        for e in &w.events {
            if let WorldEvent::Hit(h) = e {
                let car = &w.cars[h.car_idx];
                let l = letter(car.team);
                st[l][0] += 1;
                if h.shot {
                    st[l][1] += 1;
                }
                if !car.on_ground {
                    st[l][5] += 1;
                }
                let ad = if car.team == 0 { 1.0 } else { -1.0 };
                let gl = w.shape.goal_line_z();
                let to_goal = (glam::DVec3::new(0.0, 0.0, ad * gl) - w.ball.pos).normalize();
                let v = w.ball.vel;
                let q = v.dot(to_goal) / v.length().max(0.1);
                HQ.lock().unwrap()[l].push((q, v.length()));
            }
        }
        t += dt;
        since += dt;
        if kick_mode && since > kick_t {
            let adz = if a_team == 0 { 1.0 } else { -1.0 };
            KZ.lock().unwrap().push(w.ball.pos.z * adz);
            kickoff(&mut w, n);
            since = 0.0;
        }
        if w.kickoff && (w.ball.last_touch.is_some() || since > 6.0) {
            w.kickoff = false;
        }
        let gt = w.shape.in_goal(w.ball.pos, w.phys.ball_r);
        if gt >= 0 {
            score[1 - gt as usize] += 1;
            let scorer = 1 - gt as u8;
            if let Some(lt) = w.ball.last_touch {
                if w.cars[lt].team != scorer {
                    st[letter(w.cars[lt].team)][2] += 1;
                }
            }
            if since < 8.0 {
                st[letter(scorer)][3] += 1;
            }
            kickoff(&mut w, n);
            since = 0.0;
        }
    }
    let mut g = STATS.lock().unwrap();
    for l in 0..2 {
        for k in 0..6 {
            g[l][k] += st[l][k];
        }
    }
    score
}

fn apply_env(bot: &mut BotAi, who: &str) {
    if let Ok(v) = std::env::var(format!("{who}_IDX")) {
        bot.diff_index = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var(format!("{who}_D")) {
        for kv in v.split(',') {
            let mut it = kv.split('=');
            let k = it.next().unwrap();
            let x: f64 = it.next().unwrap().parse().unwrap();
            match k {
                "react" => bot.d.react = x,
                "aim" => bot.d.aim = x,
                "speed" => bot.d.speed = x,
                "aerialH" => bot.d.aerial_h = x,
                "err" => bot.d.err = x,
                "fix" => bot.d.fix = x,
                "plan" => bot.d.plan = x,
                _ => panic!("bad key {k}"),
            }
        }
    }
}
