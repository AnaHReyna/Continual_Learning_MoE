import os
import glob
import pickle
from typing import Dict, List, Sequence, Optional, Tuple

import numpy as np
import tensorflow as tf




def load_teacher_task_prototypes(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    prototypes = data["prototypes"].astype(np.float32)   # [K, D]
    task_ids = data["task_ids"].astype(np.int32)
    task_names = data["task_names"]

    task_name_to_proto = {}
    task_id_to_proto = {}

    for i in range(len(task_ids)):
        tid = int(task_ids[i])
        tname = str(task_names[i])
        proto = prototypes[i]

        task_name_to_proto[tname] = proto
        task_id_to_proto[tid] = proto

    emb_dim = int(prototypes.shape[-1])
    return task_name_to_proto, task_id_to_proto, emb_dim


def infer_time_mask(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 3:
        raise ValueError(f"Expected obs with ndim=3, got shape={obs.shape}")
    ego = obs[0]
    mask = (np.abs(ego[:, 0]) > 1e-8).astype(np.float32)
    if mask.sum() == 0:
        mask[-1] = 1.0
    return mask


def _step_info(step: Dict) -> Dict:
    info = step.get("info", None)
    return info if isinstance(info, dict) else {}


def get_preferred_steer(step: Dict,
                        control_steer_threshold: float = 0.08,
                        action_steer_threshold: float = 0.15,
                        steer_idx: int = 1) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    
    info = _step_info(step)

    if "control_steer" in info and info["control_steer"] is not None:
        try:
            return float(info["control_steer"]), float(control_steer_threshold), "control_steer"
        except Exception:
            pass

    if "rl_steer" in info and info["rl_steer"] is not None:
        try:
            return float(info["rl_steer"]), float(action_steer_threshold), "rl_steer"
        except Exception:
            pass

    if step.get("teacher_mean_action", None) is not None:
        try:
            return float(step["teacher_mean_action"][steer_idx]), float(action_steer_threshold), "teacher_mean_action"
        except Exception:
            pass

    if step.get("act", None) is not None:
        try:
            return float(step["act"][steer_idx]), float(action_steer_threshold), "act"
        except Exception:
            pass

    return None, None, None


def compute_mode_label_from_step(step: Dict,
                                 control_steer_threshold: float = 0.08,
                                 action_steer_threshold: float = 0.15,
                                 steer_idx: int = 1) -> Tuple[np.int32, str]:
    
    steer, threshold, _ = get_preferred_steer(step,
                                              control_steer_threshold=control_steer_threshold,
                                              action_steer_threshold=action_steer_threshold,
                                              steer_idx=steer_idx,
                                            )

    if steer is None or threshold is None:
        return np.int32(0), "straight"

    return (np.int32(1), "curve") if abs(steer) >= threshold else (np.int32(0), "straight")


def get_preferred_speed_cmd(step: Dict,
                            speed_idx: int = 0) -> Tuple[Optional[float], Optional[str]]:
    """
    Retorna a velocidade comandada pelo teacher.
    Prioridade:
      1) teacher_mean_action[0]
      2) act[0]
    """
    if step.get("teacher_mean_action", None) is not None:
        try:
            return float(step["teacher_mean_action"][speed_idx]), "teacher_mean_action"
        except Exception:
            pass

    if step.get("act", None) is not None:
        try:
            return float(step["act"][speed_idx]), "act"
        except Exception:
            pass

    return None, None


def compute_interaction_label_from_step(step: Dict,
                                        task_name: str,
                                        speed_idx: int = 0,
                                        cautious_speed_thresh: float = 25.0,
                                        yield_speed_thresh: float = 8.0
                                        ) -> Tuple[np.int32, str]:
    """
    interaction_label:
      0 = free_drive
      1 = cautious
      2 = yield_stop

    Rules:
    - outside of pedestrian zones -> free_drive
    - in pedestrian zones:
        speed_cmd <= yield_speed_thresh      -> yield_stop
        speed_cmd <= cautious_speed_thresh   -> cautious
        speed_cmd > cautious_speed_thresh    -> free_drive
    """

    if str(task_name) != "pedestrian":
        return np.int32(0), "free_drive"

    speed_cmd, source = get_preferred_speed_cmd(step, speed_idx=speed_idx)

    if speed_cmd is None:
        return np.int32(0), "free_drive"

    if speed_cmd <= yield_speed_thresh:
        return np.int32(2), "yield_stop"
    elif speed_cmd <= cautious_speed_thresh:
        return np.int32(1), "cautious"
    else:
        return np.int32(0), "free_drive"


class DistillTransitionDataset:
    def __init__(self,
                 dataset_dirs: Sequence[str],
                 target_mode: str = "mean_action",
                 use_vision: bool = False,
                 require_map: bool = False,
                 mode_control_steer_threshold: float = 0.08,
                 mode_action_steer_threshold: float = 0.15,
                 mode_steer_idx: int = 1,
                 interaction_speed_idx: int = 0,
                 interaction_cautious_speed_thresh: float = 25.0,
                 interaction_yield_speed_thresh: float = 8.0,
                 teacher_task_prototypes_path: Optional[str] = None,
                 ):
            
        self.dataset_dirs = list(dataset_dirs)
        self.target_mode = target_mode
        self.use_vision = bool(use_vision)
        self.require_map = bool(require_map)

        self.mode_control_steer_threshold = float(mode_control_steer_threshold)
        self.mode_action_steer_threshold = float(mode_action_steer_threshold)
        self.mode_steer_idx = int(mode_steer_idx)

        self.teacher_task_prototypes_path = teacher_task_prototypes_path
        self.teacher_task_name_to_proto = None
        self.teacher_task_id_to_proto = None
        self.teacher_task_emb_dim = 0

        if self.teacher_task_prototypes_path is not None:
            (self.teacher_task_name_to_proto,
            self.teacher_task_id_to_proto,
            self.teacher_task_emb_dim) = load_teacher_task_prototypes(self.teacher_task_prototypes_path)


        self.interaction_speed_idx = int(interaction_speed_idx)
        self.interaction_cautious_speed_thresh = float(interaction_cautious_speed_thresh)
        self.interaction_yield_speed_thresh = float(interaction_yield_speed_thresh)

        self.samples: List[Dict] = []
        self.task_to_id: Dict[str, int] = {}
        self._load_all()

    def _rollout_files(self) -> List[str]:
        files: List[str] = []
        for d in self.dataset_dirs:
            files.extend(sorted(glob.glob(os.path.join(d, "*.pkl"))))
        return files
    

    def _choose_target(self, step: Dict) -> np.ndarray:
        if self.target_mode == "action":
            key = "act"
        elif self.target_mode == "mean_action":
            key = "teacher_mean_action" if step.get("teacher_mean_action", None) is not None else "act"
        elif self.target_mode == "raw_mean":
            key = "teacher_mean_raw" if step.get("teacher_mean_raw", None) is not None else "act"
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")
        return np.asarray(step[key], dtype=np.float32)
    

    def _load_all(self):
        files = self._rollout_files()
        if len(files) == 0:
            raise FileNotFoundError(f"No .pkl files found in: {self.dataset_dirs}")

        total = 0
        total_curve = 0
        total_straight = 0

        for path in files:
            with open(path, "rb") as f:
                traj = pickle.load(f)

            task_name = str(traj.get("task_name", "unknown"))            
            if task_name not in self.task_to_id:
                self.task_to_id[task_name] = len(self.task_to_id)
            task_id = self.task_to_id[task_name]


            teacher_task_embedding = None
            if self.teacher_task_name_to_proto is not None:
                if task_name in self.teacher_task_name_to_proto:
                    teacher_task_embedding = self.teacher_task_name_to_proto[task_name].astype(np.float32)
                else:
                    teacher_task_embedding = np.zeros((self.teacher_task_emb_dim,), dtype=np.float32)



            for step in traj.get("steps", []):
                obs = np.asarray(step["obs"], dtype=np.float32)
                mask = infer_time_mask(obs)

                map_state = step.get("map_state", None)
                if map_state is not None:
                    map_state = np.asarray(map_state, dtype=np.float32)
                elif self.require_map:
                    raise ValueError(f"map_state missing in {path}")

                vision = step.get("vision", None)
                if vision is not None:
                    vision = np.asarray(vision, dtype=np.float32)
                elif self.use_vision:
                    vision = np.zeros((280,), dtype=np.float32)

                teacher_log_std = step.get("teacher_log_std", None)
                if teacher_log_std is not None:
                    teacher_log_std = np.asarray(teacher_log_std, dtype=np.float32)

                teacher_mean_raw = step.get("teacher_mean_raw", None)
                if teacher_mean_raw is not None:
                    teacher_mean_raw = np.asarray(teacher_mean_raw, dtype=np.float32)

                teacher_mean_action = step.get("teacher_mean_action", None)
                if teacher_mean_action is not None:
                    teacher_mean_action = np.asarray(teacher_mean_action, dtype=np.float32)

                mode_label, mode_name = compute_mode_label_from_step(step,
                                                                     control_steer_threshold=self.mode_control_steer_threshold,
                                                                     action_steer_threshold=self.mode_action_steer_threshold,
                                                                     steer_idx=self.mode_steer_idx,
                                                                    )
                
                interaction_label, interaction_name = compute_interaction_label_from_step(step,
                                                                                          task_name=task_name,
                                                                                          speed_idx=self.interaction_speed_idx,
                                                                                          cautious_speed_thresh=self.interaction_cautious_speed_thresh,
                                                                                          yield_speed_thresh=self.interaction_yield_speed_thresh,
                                                                                         )
                

                
                if int(mode_label) == 1:
                    total_curve += 1
                else:
                    total_straight += 1


                self.samples.append({"obs": obs,
                                     "mask": mask,
                                     "map_state": map_state,
                                     "vision": vision,
                                     "target": self._choose_target(step),
                                     "act": np.asarray(step["act"], dtype=np.float32),
                                     "teacher_mean_raw": teacher_mean_raw,
                                     "teacher_log_std": teacher_log_std,
                                     "teacher_mean_action": teacher_mean_action,
                                     "task_id": np.int32(task_id),
                                     "task_name": task_name,
                                     "mode_label": np.int32(mode_label),
                                     "mode_name": mode_name,
                                     "interaction_label": np.int32(interaction_label),
                                     "interaction_name": interaction_name,
                                     "done": np.float32(step.get("done", False)),
                                     "teacher_task_embedding": teacher_task_embedding,
                                    }
                                   )
                
                total += 1

        print(f"Loaded {total} transitions from {len(self.dataset_dirs)} dataset directories")
        print(f"Task mapping: {self.task_to_id}")
        print(f"Mode counts: straight={total_straight}, curve={total_curve}")



    def __len__(self):
        return len(self.samples)
    

    def get_specimen(self) -> Dict:
        return self.samples[0]
    

    def make_tf_dataset(self,
                        batch_size: int = 64,
                        shuffle: bool = True,
                        shuffle_buffer: int = 10000,
                        repeat: bool = False,
                       ) -> tf.data.Dataset:
        
        sample0 = self.samples[0]

        output_signature = {"obs": tf.TensorSpec(shape=sample0["obs"].shape, dtype=tf.float32),
                            "mask": tf.TensorSpec(shape=sample0["mask"].shape, dtype=tf.float32),
                            "map_state": tf.TensorSpec(shape=sample0["map_state"].shape if sample0["map_state"] is not None else (0,), dtype=tf.float32,),
                            "vision": tf.TensorSpec(shape=sample0["vision"].shape if sample0["vision"] is not None else (0,), dtype=tf.float32,),
                            "target": tf.TensorSpec(shape=sample0["target"].shape, dtype=tf.float32),
                            "act": tf.TensorSpec(shape=sample0["act"].shape, dtype=tf.float32),
                            "teacher_mean_raw": tf.TensorSpec(shape=sample0["teacher_mean_raw"].shape if sample0["teacher_mean_raw"] is not None else (0,), dtype=tf.float32,),
                            "teacher_log_std": tf.TensorSpec(shape=sample0["teacher_log_std"].shape if sample0["teacher_log_std"] is not None else (0,), dtype=tf.float32,),
                            "teacher_mean_action": tf.TensorSpec(shape=sample0["teacher_mean_action"].shape if sample0["teacher_mean_action"] is not None else (0,), dtype=tf.float32,),
                            "task_id": tf.TensorSpec(shape=(), dtype=tf.int32),
                            "mode_label": tf.TensorSpec(shape=(), dtype=tf.int32),
                            "interaction_label": tf.TensorSpec(shape=(), dtype=tf.int32),
                            "teacher_task_embedding": tf.TensorSpec(shape=sample0["teacher_task_embedding"].shape if sample0["teacher_task_embedding"] is not None else (0,), dtype=tf.float32,),
                            "done": tf.TensorSpec(shape=(), dtype=tf.float32),
                            }
        

        def gen():
            for s in self.samples:
                yield {"obs": s["obs"],
                       "mask": s["mask"],
                       "map_state": s["map_state"] if s["map_state"] is not None else np.zeros((0,), dtype=np.float32),
                       "vision": s["vision"] if s["vision"] is not None else np.zeros((0,), dtype=np.float32),
                       "target": s["target"],
                       "act": s["act"],
                       "teacher_mean_raw": s["teacher_mean_raw"] if s["teacher_mean_raw"] is not None else np.zeros((0,), dtype=np.float32),
                       "teacher_log_std": s["teacher_log_std"] if s["teacher_log_std"] is not None else np.zeros((0,), dtype=np.float32),
                       "teacher_mean_action": s["teacher_mean_action"] if s["teacher_mean_action"] is not None else np.zeros((0,), dtype=np.float32),
                       "task_id": s["task_id"],
                       "mode_label": s["mode_label"],
                       "interaction_label": s["interaction_label"],
                       "teacher_task_embedding": s["teacher_task_embedding"] if s["teacher_task_embedding"] is not None else np.zeros((0,), dtype=np.float32),
                       "done": s["done"],
                       }

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(min(shuffle_buffer, len(self.samples)))
        if repeat:
            ds = ds.repeat()
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)