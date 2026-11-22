import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MAX_CARS = 10
OTHER_CARS = MAX_CARS - 1
NUM_PADS_OBS = 6
SELF_DIM = 24
BALL_DIM = 16
OTHER_DIM = 18
PAD_DIM = 4
ACTION_DIM = 8

# Factored raw control bins: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
ACTION_BIN_VALUES = [
    np.array([-1.0, 0.0, 1.0], dtype=np.float32),          # throttle
    np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),  # steer
    np.array([-1.0, 0.0, 1.0], dtype=np.float32),          # pitch
    np.array([-1.0, 0.0, 1.0], dtype=np.float32),          # yaw
    np.array([-1.0, 0.0, 1.0], dtype=np.float32),          # roll
    np.array([0.0, 1.0], dtype=np.float32),                # jump
    np.array([0.0, 1.0], dtype=np.float32),                # boost
    np.array([0.0, 1.0], dtype=np.float32),                # handbrake
]
ACTION_N_BINS = [len(v) for v in ACTION_BIN_VALUES]


class EnvConfig(ctypes.Structure):
    _fields_ = [
        ("num_envs", ctypes.c_uint32),
        ("min_team_size", ctypes.c_uint32),
        ("max_team_size", ctypes.c_uint32),
        ("ticks_per_step", ctypes.c_uint32),
        ("max_episode_steps", ctypes.c_uint32),
        ("bot_opponent_prob", ctypes.c_float),
        ("bot_diff", ctypes.c_uint32),
        ("num_threads", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("w_goal", ctypes.c_float),
        ("w_touch", ctypes.c_float),
        ("w_shot", ctypes.c_float),
        ("w_save", ctypes.c_float),
        ("w_demo", ctypes.c_float),
        ("w_ball_to_goal", ctypes.c_float),
        ("w_player_to_ball", ctypes.c_float),
        ("w_boost_pickup", ctypes.c_float),
    ]


class StepStats(ctypes.Structure):
    _fields_ = [
        ("goals_blue", ctypes.c_uint32),
        ("goals_orange", ctypes.c_uint32),
        ("episodes_done", ctypes.c_uint32),
        ("touches", ctypes.c_uint32),
        ("bot_games_won", ctypes.c_uint32),
        ("bot_games_lost", ctypes.c_uint32),
    ]


def _default_lib_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    crate = root / "rust" / "rl_env"
    if sys.platform == "win32":
        name = "rl_env.dll"
    elif sys.platform == "darwin":
        name = "librl_env.dylib"
    else:
        name = "librl_env.so"
    return crate / "target" / "release" / name


def load_rl_env_lib(auto_build: bool = True) -> ctypes.CDLL:
    lib_path = _default_lib_path()
    if not lib_path.exists():
        if not auto_build:
            raise FileNotFoundError(f"Rust rl_env library not found at {lib_path}")
        crate_toml = lib_path.parents[2] / "Cargo.toml"
        print(f"[rl_env] Building native Rust simulator ({crate_toml})...")
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(crate_toml)],
            check=True,
        )
    lib = ctypes.CDLL(str(lib_path))

    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    void_p = ctypes.c_void_p

    lib.rl_env_create.argtypes = [EnvConfig]
    lib.rl_env_create.restype = void_p

    lib.rl_env_destroy.argtypes = [void_p]
    lib.rl_env_destroy.restype = None

    lib.rl_env_set_bot_curriculum.argtypes = [void_p, ctypes.c_float, ctypes.c_uint32]
    lib.rl_env_set_bot_curriculum.restype = None

    lib.rl_env_reset_all.argtypes = [
        void_p, f32_p, f32_p, f32_p, f32_p, f32_p, f32_p, i32_p
    ]
    lib.rl_env_reset_all.restype = None

    lib.rl_env_step.argtypes = [
        void_p, f32_p, f32_p, f32_p, f32_p, f32_p, f32_p, f32_p, i32_p,
        f32_p, u8_p, ctypes.POINTER(StepStats)
    ]
    lib.rl_env_step.restype = None

    lib.rl_env_expert_actions.argtypes = [void_p, f32_p]
    lib.rl_env_expert_actions.restype = None
    return lib


def discrete_to_controls(action_indices: np.ndarray) -> np.ndarray:
    """Convert [..., 8] integer bin indices into [..., 8] raw float32 controls."""
    out = np.empty(action_indices.shape, dtype=np.float32)
    for k, vals in enumerate(ACTION_BIN_VALUES):
        out[..., k] = vals[action_indices[..., k]]
    return out


def controls_to_discrete(raw_controls: np.ndarray) -> np.ndarray:
    """Quantize [..., 8] raw float32 controls into nearest [..., 8] int64 bin indices (for BC)."""
    out = np.empty(raw_controls.shape, dtype=np.int64)
    for k, vals in enumerate(ACTION_BIN_VALUES):
        diff = np.abs(raw_controls[..., k : k + 1] - vals)
        out[..., k] = np.argmin(diff, axis=-1)
    return out


class HyperstrikeVecEnv:
    """Multi-threaded vectorized Hyperstrike environment backed by exact Rust physics."""

    def __init__(
        self,
        num_envs: int = 64,
        min_team_size: int = 1,
        max_team_size: int = 5,
        ticks_per_step: int = 8,
        max_episode_steps: int = 900,
        bot_opponent_prob: float = 0.25,
        bot_diff: int = 4,
        num_threads: int | None = None,
        seed: int = 42,
        w_goal: float = 10.0,
        w_touch: float = 0.25,
        w_shot: float = 0.6,
        w_save: float = 1.0,
        w_demo: float = 0.8,
        w_ball_to_goal: float = 0.35,
        w_player_to_ball: float = 0.08,
        w_boost_pickup: float = 0.12,
    ):
        self.lib = load_rl_env_lib(auto_build=True)
        self.num_envs = num_envs
        threads = num_threads if num_threads is not None else min(os.cpu_count() or 8, 32)
        self.cfg = EnvConfig(
            num_envs=num_envs,
            min_team_size=min_team_size,
            max_team_size=max_team_size,
            ticks_per_step=ticks_per_step,
            max_episode_steps=max_episode_steps,
            bot_opponent_prob=bot_opponent_prob,
            bot_diff=bot_diff,
            num_threads=threads,
            seed=seed,
            w_goal=w_goal,
            w_touch=w_touch,
            w_shot=w_shot,
            w_save=w_save,
            w_demo=w_demo,
            w_ball_to_goal=w_ball_to_goal,
            w_player_to_ball=w_player_to_ball,
            w_boost_pickup=w_boost_pickup,
        )
        self._ptr = self.lib.rl_env_create(self.cfg)

        # Pre-allocated contiguous zero-copy buffers
        self.self_obs = np.zeros((num_envs, MAX_CARS, SELF_DIM), dtype=np.float32)
        self.ball_obs = np.zeros((num_envs, MAX_CARS, BALL_DIM), dtype=np.float32)
        self.other_obs = np.zeros((num_envs, MAX_CARS, OTHER_CARS, OTHER_DIM), dtype=np.float32)
        self.other_mask = np.zeros((num_envs, MAX_CARS, OTHER_CARS), dtype=np.float32)
        self.pads_obs = np.zeros((num_envs, MAX_CARS, NUM_PADS_OBS, PAD_DIM), dtype=np.float32)
        self.active_mask = np.zeros((num_envs, MAX_CARS), dtype=np.float32)
        self.team_ids = np.zeros((num_envs, MAX_CARS), dtype=np.int32)
        self.rewards = np.zeros((num_envs, MAX_CARS), dtype=np.float32)
        self.dones = np.zeros((num_envs,), dtype=np.uint8)
        self._expert_buf = np.zeros((num_envs, MAX_CARS, ACTION_DIM), dtype=np.float32)

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        return {
            "self": self.self_obs,
            "ball": self.ball_obs,
            "other": self.other_obs,
            "other_mask": self.other_mask,
            "pads": self.pads_obs,
            "active_mask": self.active_mask,
            "team_ids": self.team_ids,
        }

    def set_bot_curriculum(self, bot_opponent_prob: float, bot_diff: int) -> None:
        self.lib.rl_env_set_bot_curriculum(
            self._ptr, ctypes.c_float(bot_opponent_prob), ctypes.c_uint32(bot_diff)
        )

    def reset(self) -> Dict[str, np.ndarray]:
        self.lib.rl_env_reset_all(
            self._ptr,
            self.self_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.ball_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.other_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.other_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.pads_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.active_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.team_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return self._obs_dict()

    def step(self, raw_controls: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, StepStats]:
        actions = np.ascontiguousarray(raw_controls, dtype=np.float32)
        stats = StepStats()
        self.lib.rl_env_step(
            self._ptr,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.self_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.ball_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.other_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.other_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.pads_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.active_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.team_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(stats),
        )
        return self._obs_dict(), self.rewards, self.dones, stats

    def get_expert_actions(self) -> np.ndarray:
        """Query the built-in CHAMP bot for raw controls on all cars (for BC warmstart)."""
        self.lib.rl_env_expert_actions(
            self._ptr,
            self._expert_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return self._expert_buf

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self.lib.rl_env_destroy(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()
