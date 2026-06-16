import os
import glob
import pickle
from typing import Dict, List, Sequence, Optional

import numpy as np
import tensorflow as tf


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return obs.reshape(-1).astype(np.float32)


class TeacherTaskWindowDataset:
    """
    Dataset de janelas temporais para treinar o teacher task encoder.

    Cada amostra representa uma janela de trajetória:
      - states  : [L, state_dim]
      - actions : [L, action_dim]
      - rewards : [L, 1]
      - mask    : [L]
      - task_id : escalar
    """

    def __init__(
        self,
        dataset_dirs: Sequence[str],
        window_len: int = 20,
        stride: int = 10,
        state_key: str = "obs",
        action_source: str = "teacher_mean_action",   # ou "act"
        reward_key: str = "rew",
        flatten_state: bool = True,
        pad_short_windows: bool = True,
        balance_tasks: bool = False,
    ):
        self.dataset_dirs = list(dataset_dirs)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.state_key = str(state_key)
        self.action_source = str(action_source)
        self.reward_key = str(reward_key)
        self.flatten_state = bool(flatten_state)
        self.pad_short_windows = bool(pad_short_windows)
        self.balance_tasks = bool(balance_tasks)

        self.samples: List[Dict] = []
        self.task_to_id: Dict[str, int] = {}

        self._load_all()

        if self.balance_tasks:
            self._rebalance_tasks()

    def _rollout_files(self) -> List[str]:
        files: List[str] = []
        for d in self.dataset_dirs:
            files.extend(sorted(glob.glob(os.path.join(d, "*.pkl"))))
        return files

    def _get_task_id(self, task_name: str) -> int:
        if task_name not in self.task_to_id:
            self.task_to_id[task_name] = len(self.task_to_id)
        return self.task_to_id[task_name]

    def _extract_state(self, step: Dict) -> np.ndarray:
        state = step[self.state_key]
        state = np.asarray(state, dtype=np.float32)
        if self.flatten_state:
            state = _flatten_obs(state)
        return state

    def _extract_action(self, step: Dict) -> np.ndarray:
        if self.action_source == "teacher_mean_action" and step.get("teacher_mean_action", None) is not None:
            return np.asarray(step["teacher_mean_action"], dtype=np.float32)

        if self.action_source == "act" and step.get("act", None) is not None:
            return np.asarray(step["act"], dtype=np.float32)

        # fallback seguro
        if step.get("teacher_mean_action", None) is not None:
            return np.asarray(step["teacher_mean_action"], dtype=np.float32)
        if step.get("act", None) is not None:
            return np.asarray(step["act"], dtype=np.float32)

        raise KeyError("No valid action found in step (expected teacher_mean_action or act).")

    def _extract_reward(self, step: Dict) -> np.ndarray:
        if self.reward_key not in step:
            raise KeyError(f"Reward key '{self.reward_key}' not found in step.")
        r = np.asarray(step[self.reward_key], dtype=np.float32).reshape(1)
        return r

    def _build_window(self, steps: List[Dict], start_idx: int, task_id: int, task_name: str, episode_id: str):
        state_list = []
        action_list = []
        reward_list = []
        mask_list = []

        end_idx = start_idx + self.window_len
        n_steps = len(steps)

        for t in range(start_idx, min(end_idx, n_steps)):
            step = steps[t]
            state_list.append(self._extract_state(step))
            action_list.append(self._extract_action(step))
            reward_list.append(self._extract_reward(step))
            mask_list.append(1.0)

        valid_len = len(state_list)

        if valid_len == 0:
            return None

        if valid_len < self.window_len:
            if not self.pad_short_windows:
                return None

            state_dim = state_list[0].shape[-1]
            action_dim = action_list[0].shape[-1]

            for _ in range(self.window_len - valid_len):
                state_list.append(np.zeros((state_dim,), dtype=np.float32))
                action_list.append(np.zeros((action_dim,), dtype=np.float32))
                reward_list.append(np.zeros((1,), dtype=np.float32))
                mask_list.append(0.0)

        states = np.stack(state_list, axis=0).astype(np.float32)    # [L, Ds]
        actions = np.stack(action_list, axis=0).astype(np.float32)  # [L, Da]
        rewards = np.stack(reward_list, axis=0).astype(np.float32)  # [L, 1]
        mask = np.asarray(mask_list, dtype=np.float32)               # [L]

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "mask": mask,
            "task_id": np.int32(task_id),
            "task_name": task_name,
            "episode_id": episode_id,
            "valid_len": np.int32(valid_len),
        }

    def _load_all(self):
        files = self._rollout_files()
        if len(files) == 0:
            raise FileNotFoundError(f"No .pkl files found in: {self.dataset_dirs}")

        total_windows = 0

        for path in files:
            with open(path, "rb") as f:
                traj = pickle.load(f)

            task_name = str(traj.get("task_name", "unknown"))
            task_id = self._get_task_id(task_name)
            episode_id = str(traj.get("episode_id", os.path.basename(path)))

            steps = traj.get("steps", [])
            if len(steps) == 0:
                continue

            if self.pad_short_windows:
                start_positions = list(range(0, len(steps), self.stride))
            else:
                max_start = len(steps) - self.window_len + 1
                if max_start <= 0:
                    start_positions = []
                else:
                    start_positions = list(range(0, max_start, self.stride))

            for start_idx in start_positions:
                sample = self._build_window(
                    steps=steps,
                    start_idx=start_idx,
                    task_id=task_id,
                    task_name=task_name,
                    episode_id=episode_id,
                )
                if sample is not None:
                    self.samples.append(sample)
                    total_windows += 1

        print(f"Loaded {total_windows} windows from {len(files)} rollout files")
        print(f"Task mapping: {self.task_to_id}")

    def _rebalance_tasks(self):
        by_task: Dict[int, List[Dict]] = {}
        for s in self.samples:
            tid = int(s["task_id"])
            by_task.setdefault(tid, []).append(s)

        min_count = min(len(v) for v in by_task.values())
        balanced = []
        rng = np.random.default_rng(42)

        for tid, items in by_task.items():
            idx = rng.permutation(len(items))[:min_count]
            for i in idx:
                balanced.append(items[i])

        rng.shuffle(balanced)
        self.samples = balanced

        print(f"Balanced tasks to {min_count} windows each")

    def __len__(self):
        return len(self.samples)

    def get_specimen(self) -> Dict:
        return self.samples[0]

    def get_task_counts(self) -> Dict[str, int]:
        counts = {}
        for s in self.samples:
            name = s["task_name"]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def make_tf_dataset(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        sample0 = self.samples[0]

        output_signature = {
            "states": tf.TensorSpec(shape=sample0["states"].shape, dtype=tf.float32),
            "actions": tf.TensorSpec(shape=sample0["actions"].shape, dtype=tf.float32),
            "rewards": tf.TensorSpec(shape=sample0["rewards"].shape, dtype=tf.float32),
            "mask": tf.TensorSpec(shape=sample0["mask"].shape, dtype=tf.float32),
            "task_id": tf.TensorSpec(shape=(), dtype=tf.int32),
            "valid_len": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        def gen():
            for s in self.samples:
                yield {
                    "states": s["states"],
                    "actions": s["actions"],
                    "rewards": s["rewards"],
                    "mask": s["mask"],
                    "task_id": s["task_id"],
                    "valid_len": s["valid_len"],
                }

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        if shuffle:
            ds = ds.shuffle(min(shuffle_buffer, len(self.samples)))
        if repeat:
            ds = ds.repeat()

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    

# dataset = TeacherTaskWindowDataset(
#     dataset_dirs=[
#         "../../export_rollouts/datasets/lane_keeping_eval",
#         "../../export_rollouts/datasets/change_lane_eval",
#     ],
#     window_len=20,
#     stride=10,
#     action_source="teacher_mean_action",
#     reward_key="rew",
#     balance_tasks=True, # False
# )