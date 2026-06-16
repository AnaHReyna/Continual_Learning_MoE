import os
import glob
import pickle
import random
import shutil


def build_success_replay_subset(src_dir,
                                dst_dir,
                                fraction=0.10,
                                seed=42,
                                require_success=True):
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, "*.pkl")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found in: {src_dir}")

    candidates = []

    for path in files:
        try:
            with open(path, "rb") as f:
                traj = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to open {path}: {e}")
            continue

        success = bool(traj.get("success", False))

        if require_success and not success:
            continue

        candidates.append(path)

    if len(candidates) == 0:
        raise ValueError("No episode with success=True found.")

    k = max(1, int(round(len(files) * fraction)))
    k = min(k, len(candidates))

    rng = random.Random(seed)
    selected = rng.sample(candidates, k)

    for path in selected:
        fname = os.path.basename(path)
        shutil.copy2(path, os.path.join(dst_dir, fname))

    print(f"Total number of files in the original dataset: {len(files)}")
    print(f"Candidates with success=True: {len(candidates)}")
    print(f"Selected for replay: {k}")
    print(f"Replay saved to: {dst_dir}")


if __name__ == "__main__":
    build_success_replay_subset(
        src_dir="../export_rollouts/datasets/change_lane_eval_antigo",
        dst_dir="../export_rollouts/datasets/change_lane_replay_10pct_success",
        fraction=0.10,
        seed=42,
        require_success=True
    )
