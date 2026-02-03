"""FurnitureBench 仿真环境 gRPC 服务端（无 proto，使用 bytes 序列化）。"""

from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence

import grpc
from concurrent import futures
import numpy as np
import torch

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
from furniture_bench.envs.observation import DEFAULT_STATE_OBS, DEFAULT_VISUAL_OBS
from furniture_bench.grpc.grpc_codec import dumps, loads


_DEFAULT_MAX_MSG = 256 * 1024 * 1024


def _is_image_key(key: str) -> bool:
    key = key.lower()
    return (
        key.startswith("color")
        or key.startswith("depth")
        or "image" in key
        or "camera" in key
    )


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten_obs_batch(
    obs: Dict[str, Any],
    state_keys: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """把 obs dict 展平成 [num_envs, state_dim] 的 float32。"""

    if state_keys is None:
        keys = [k for k in sorted(obs.keys()) if not _is_image_key(k)]
    else:
        keys = list(state_keys)

    parts: List[np.ndarray] = []
    for key in keys:
        if key not in obs:
            continue
        val = obs[key]
        if isinstance(val, dict):
            for sub_key in sorted(val.keys()):
                arr = _to_numpy(val[sub_key]).reshape(len(val[sub_key]), -1)
                parts.append(arr.astype(np.float32, copy=False))
        else:
            arr = _to_numpy(val).reshape(_to_numpy(val).shape[0], -1)
            parts.append(arr.astype(np.float32, copy=False))

    if not parts:
        num_envs = 0
        if obs:
            sample = next(iter(obs.values()))
            num_envs = int(_to_numpy(sample).shape[0])
        return np.zeros((num_envs, 0), dtype=np.float32)

    return np.concatenate(parts, axis=-1)


class _EnvService:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()

        obs_keys = cfg.obs_keys
        if not obs_keys:
            obs_keys = (
                DEFAULT_VISUAL_OBS if cfg.obs_mode == "image" else DEFAULT_STATE_OBS
            )

        self.env = FurnitureRLSimEnv(
            cfg.randomness,
            randomize_obstacle=cfg.randomize_obstacle,
            furniture=cfg.furniture,
            num_envs=cfg.num_envs,
            obs_keys=obs_keys,
            concat_robot_state=False,
            headless=cfg.headless,
            compute_device_id=cfg.compute_device_id,
            graphics_device_id=cfg.graphics_device_id,
            init_assembled=cfg.init_assembled,
            np_step_out=False,
            channel_first=False,
            action_type=cfg.action_type,
            act_rot_repr=cfg.act_rot_repr,
            ctrl_mode=cfg.ctrl_mode,
            record=False,
            max_env_steps=cfg.max_env_steps,
            parts_poses_in_robot_frame=cfg.parts_poses_in_robot_frame,
        )

        self.num_envs = int(cfg.num_envs)
        self.state_keys = cfg.state_keys
        self.main_image_key = cfg.main_image_key
        self.extra_image_key = cfg.extra_image_key
        self.include_extra_view = cfg.include_extra_view

        self._last_obs = None

    def _build_obs_payload(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        state = _flatten_obs_batch(obs, self.state_keys)
        payload["states"] = state.astype(np.float32, copy=False)

        if self.cfg.obs_mode == "image":
            main = obs.get(self.main_image_key)
            if main is None:
                # 回退到任意图像 key
                candidates = [k for k in obs.keys() if _is_image_key(k)]
                if candidates:
                    main = obs[candidates[0]]
            if main is not None:
                main_np = _to_numpy(main).astype(np.uint8, copy=False)
                payload["main_images"] = main_np

            if self.include_extra_view and self.extra_image_key in obs:
                extra_np = _to_numpy(obs[self.extra_image_key]).astype(
                    np.uint8, copy=False
                )
                payload["extra_view_images"] = extra_np[:, None, ...]

        return payload

    def get_spaces(self) -> Dict[str, Any]:
        action_shape = tuple(self.env.action_space.shape)
        return {
            "num_envs": self.num_envs,
            "action_shape": action_shape,
            "obs_mode": self.cfg.obs_mode,
        }

    def reset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            env_idxs = payload.get("env_idx")
            if env_idxs is not None:
                env_idxs = torch.as_tensor(env_idxs, dtype=torch.int32, device=self.env.device)
                obs = self.env.reset(env_idxs=env_idxs)
            else:
                obs = self.env.reset()
            self._last_obs = obs
            return {"obs": self._build_obs_payload(obs), "info": {}}

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        actions = payload["actions"]
        with self._lock:
            actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.env.device)
            obs, reward, done, info = self.env.step(actions_t)
            self._last_obs = obs
        reward_np = _to_numpy(reward).reshape(self.num_envs, -1)
        done_np = _to_numpy(done).reshape(self.num_envs, -1).astype(bool)
        return {
            "obs": self._build_obs_payload(obs),
            "reward": reward_np[:, 0],
            "terminated": done_np[:, 0],
            "truncated": np.zeros((self.num_envs,), dtype=bool),
            "info": info if isinstance(info, dict) else {},
        }

    def render(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._last_obs is None:
                obs = self.env.reset()
                self._last_obs = obs
            obs = self._last_obs
        img = obs.get(self.main_image_key)
        if img is None:
            candidates = [k for k in obs.keys() if _is_image_key(k)]
            if candidates:
                img = obs[candidates[0]]
        if img is None:
            return {"image": None}
        return {"image": _to_numpy(img)}

    def close(self) -> Dict[str, Any]:
        with self._lock:
            if hasattr(self.env, "__del__"):
                try:
                    self.env.__del__()
                except Exception:
                    pass
        return {}


def _make_handler(service: _EnvService) -> grpc.GenericRpcHandler:
    def _unary_unary(fn):
        return grpc.unary_unary_rpc_method_handler(
            fn,
            request_deserializer=lambda b: loads(b),
            response_serializer=lambda obj: dumps(obj),
        )

    methods = {
        "GetSpaces": _unary_unary(lambda req, ctx: service.get_spaces()),
        "Reset": _unary_unary(lambda req, ctx: service.reset(req)),
        "Step": _unary_unary(lambda req, ctx: service.step(req)),
        "Render": _unary_unary(lambda req, ctx: service.render(req)),
        "Close": _unary_unary(lambda req, ctx: service.close()),
    }
    return grpc.method_handlers_generic_handler("furniturebench.EnvService", methods)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)

    parser.add_argument("--furniture", type=str, default="square_table")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--init-assembled", action="store_true")
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument("--randomize-obstacle", action="store_true")
    parser.add_argument("--obs-mode", type=str, default="state", choices=["state", "image"])
    parser.add_argument("--obs-keys", nargs="*", default=None)
    parser.add_argument("--state-keys", nargs="*", default=None)
    parser.add_argument("--main-image-key", type=str, default="color_image2")
    parser.add_argument("--extra-image-key", type=str, default="color_image1")
    parser.add_argument("--include-extra-view", action="store_true")
    parser.add_argument("--parts-poses-in-robot-frame", action="store_true")

    parser.add_argument("--act-rot-repr", type=str, default="rot_6d")
    parser.add_argument("--action-type", type=str, default="delta")
    parser.add_argument("--ctrl-mode", type=str, default="diffik")
    parser.add_argument("--max-env-steps", type=int, default=3000)

    return parser.parse_args()


def main() -> None:
    cfg = _parse_args()
    service = _EnvService(cfg)

    server = grpc.server(
        thread_pool=futures.ThreadPoolExecutor(max_workers=8),
        options=[
            ("grpc.max_send_message_length", _DEFAULT_MAX_MSG),
            ("grpc.max_receive_message_length", _DEFAULT_MAX_MSG),
        ],
    )
    server.add_generic_rpc_handlers((_make_handler(service),))
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
