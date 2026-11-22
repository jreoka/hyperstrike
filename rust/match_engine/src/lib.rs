pub mod pool;
pub mod server_match;

pub use pool::{SimPool, TokioMatchHandle, detect_worker_count};
pub use server_match::{MatchState, ServerMatch, parse_packed_controls, unpack_controls};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use serde_json::{Value, json};
    use server_core::db::sanitize_car;
    use server_core::protocol::{
        MatchBackend, MatchCreateOptions, MatchEventSink, MatchPlayerSpec, MatchStatsEntry,
        SnapshotPayload, default_lobby_settings,
    };
    use sim::{NET_MAGIC, NET_VERSION};

    use super::*;

    struct TestSink {
        broadcasts: Mutex<Vec<Value>>,
        player_snaps: Mutex<Vec<(String, Vec<u8>)>>,
        spec_snaps: Mutex<Vec<Vec<u8>>>,
    }

    impl TestSink {
        fn new() -> Self {
            Self {
                broadcasts: Mutex::new(Vec::new()),
                player_snaps: Mutex::new(Vec::new()),
                spec_snaps: Mutex::new(Vec::new()),
            }
        }
    }

    impl MatchEventSink for TestSink {
        fn broadcast(&self, msg: Value) {
            self.broadcasts.lock().unwrap().push(msg);
        }

        fn send_user(&self, _user_id: &str, _msg: Value) {}

        fn send_raw(&self, user_id: &str, payload: SnapshotPayload) {
            if let SnapshotPayload::Binary(b) = payload {
                self.player_snaps
                    .lock()
                    .unwrap()
                    .push((user_id.to_string(), b));
            }
        }

        fn send_spectators(&self, payload: SnapshotPayload) {
            if let SnapshotPayload::Binary(b) = payload {
                self.spec_snaps.lock().unwrap().push(b);
            }
        }

        fn on_goal(&self, _info: Value) {}
        fn on_end(&self, _score: [i32; 2], _stats: HashMap<String, MatchStatsEntry>) {}
        fn on_afk_kick(&self, _user_id: &str) {}
        fn on_error(&self, _message: String) {}
    }

    #[tokio::test]
    async fn sim_pool_runs_match_and_streams_binary_snapshots() {
        let pool = SimPool::new(2);
        assert_eq!(pool.stats().ready_workers, 2);
        assert_eq!(pool.stats().active_matches, 0);

        let sink = Arc::new(TestSink::new());
        let mut settings = default_lobby_settings("P1");
        settings.team_size = 1;
        settings.bot_fill = true;

        let handle = pool
            .create_match(MatchCreateOptions {
                settings,
                teammate_diff: None,
                afk_enabled: false,
                afk_warn_s: None,
                afk_kick_s: None,
                players: vec![MatchPlayerSpec {
                    user_id: "u1".to_string(),
                    username: "Player1".to_string(),
                    team: 0,
                    car: sanitize_car(None),
                }],
                callbacks: sink.clone(),
            })
            .await
            .expect("create match");

        assert_eq!(pool.stats().active_matches, 1);
        assert_eq!(handle.car_info().as_array().map(|a| a.len()), Some(2));

        handle.start_when_loaded(vec!["u1".to_string()]);
        handle.mark_loaded("u1");

        // Push a few valid inputs and let the 60Hz snapshot loop run briefly.
        assert!(handle.push_input("u1", 1, &[json!(1.0), json!(0.0), json!(0.0), json!(0.0), json!(0.0), json!(2)]));
        assert!(!handle.push_input("u1", 2, &[json!(99.0), json!(0.0), json!(0.0), json!(0.0), json!(0.0), json!(0)]));

        tokio::time::sleep(Duration::from_millis(80)).await;
        handle.stop();
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert_eq!(pool.stats().active_matches, 0);
        let broadcasts = sink.broadcasts.lock().unwrap();
        assert!(
            broadcasts.iter().any(|m| m["type"] == "match_kickoff"),
            "expected match_kickoff broadcast"
        );
        let snaps = sink.player_snaps.lock().unwrap();
        assert!(!snaps.is_empty(), "expected binary snapshots");
        let (uid, bytes) = &snaps[0];
        assert_eq!(uid, "u1");
        assert_eq!(bytes[0], NET_MAGIC);
        assert_eq!(bytes[1], NET_VERSION);
    }
}
