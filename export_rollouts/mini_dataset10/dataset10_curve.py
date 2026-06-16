import os
import glob
import pickle
import random
import shutil
import argparse


def get_steer_from_step(step, steer_idx=1):
    action = None

    if "teacher_mean_action" in step and step["teacher_mean_action"] is not None:
        action = step["teacher_mean_action"]
    elif "act" in step and step["act"] is not None:
        action = step["act"]

    if action is None:
        return None

    try:
        return float(action[steer_idx])
    except Exception:
        return None


def is_curve_episode(traj,
                     steer_idx=1,
                     steer_threshold=0.15,
                     min_curve_steps=3):
    steps = traj.get("steps", [])
    if len(steps) == 0:
        return False

    curve_count = 0
    for step in steps:
        steer = get_steer_from_step(step, steer_idx=steer_idx)
        if steer is None:
            continue
        if abs(steer) >= steer_threshold:
            curve_count += 1
            if curve_count >= min_curve_steps:
                return True

    return False


def build_success_replay_subset_curve_aware(src_dir,
                                            dst_dir,
                                            fraction=0.10,
                                            curve_ratio=0.60,
                                            seed=42,
                                            require_success=True,
                                            steer_idx=1,
                                            steer_threshold=0.15,
                                            min_curve_steps=3,
                                            clear_dst=True):
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, "*.pkl")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found in: {src_dir}")

    curve_candidates = []
    straight_candidates = []
    failed_or_filtered = 0

    for path in files:
        try:
            with open(path, "rb") as f:
                traj = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to open {path}: {e}")
            continue

        success = bool(traj.get("success", False))
        if require_success and not success:
            failed_or_filtered += 1
            continue

        if is_curve_episode(traj,
                            steer_idx=steer_idx,
                            steer_threshold=steer_threshold,
                            min_curve_steps=min_curve_steps):
            curve_candidates.append(path)
        else:
            straight_candidates.append(path)

    total_success = len(curve_candidates) + len(straight_candidates)
    if total_success == 0:
        raise ValueError("No successful episodes available after filtering.")

    k = max(1, int(round(len(files) * fraction)))
    k = min(k, total_success)

    rng = random.Random(seed)

    target_curve = int(round(k * curve_ratio))
    target_straight = k - target_curve

    take_curve = min(target_curve, len(curve_candidates))
    take_straight = min(target_straight, len(straight_candidates))

    selected = []

    if take_curve > 0:
        selected.extend(rng.sample(curve_candidates, take_curve))
    if take_straight > 0:
        selected.extend(rng.sample(straight_candidates, take_straight))

    remaining_needed = k - len(selected)
    if remaining_needed > 0:
        remaining_pool = list(set(curve_candidates + straight_candidates) - set(selected))
        if len(remaining_pool) < remaining_needed:
            raise ValueError("Not enough remaining successful episodes to complete subset.")
        selected.extend(rng.sample(remaining_pool, remaining_needed))

    if clear_dst:
        for old_file in glob.glob(os.path.join(dst_dir, "*.pkl")):
            os.remove(old_file)

    for path in selected:
        fname = os.path.basename(path)
        shutil.copy2(path, os.path.join(dst_dir, fname))

    selected_curve = 0
    selected_straight = 0
    for path in selected:
        with open(path, "rb") as f:
            traj = pickle.load(f)
        if is_curve_episode(traj,
                            steer_idx=steer_idx,
                            steer_threshold=steer_threshold,
                            min_curve_steps=min_curve_steps):
            selected_curve += 1
        else:
            selected_straight += 1

    print("========== REPLAY SUBSET SUMMARY ==========")
    print(f"Source dir:                 {src_dir}")
    print(f"Destination dir:            {dst_dir}")
    print(f"Total files in source:      {len(files)}")
    print(f"Filtered out (non-success): {failed_or_filtered}")
    print(f"Successful curve episodes:  {len(curve_candidates)}")
    print(f"Successful straight epis.:  {len(straight_candidates)}")
    print(f"Requested fraction:         {fraction}")
    print(f"Requested replay size:      {k}")
    print(f"Requested curve ratio:      {curve_ratio}")
    print(f"Selected curve episodes:    {selected_curve}")
    print(f"Selected straight episodes: {selected_straight}")
    print(f"Steer threshold:            {steer_threshold}")
    print(f"Min curve steps:            {min_curve_steps}")
    print("==========================================")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str,
                        default="../export_rollouts/datasets/lane_keeping")
    parser.add_argument("--dst-dir", type=str,
                        default="../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--curve-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-success", action="store_true", default=True)
    parser.add_argument("--steer-idx", type=int, default=1)
    parser.add_argument("--steer-threshold", type=float, default=0.15)
    parser.add_argument("--min-curve-steps", type=int, default=3)
    parser.add_argument("--no-clear-dst", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_success_replay_subset_curve_aware(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        fraction=args.fraction,
        curve_ratio=args.curve_ratio,
        seed=args.seed,
        require_success=args.require_success,
        steer_idx=args.steer_idx,
        steer_threshold=args.steer_threshold,
        min_curve_steps=args.min_curve_steps,
        clear_dst=not args.no_clear_dst,
    )


# python dataset10_curve.py \
#   --src-dir ../export_rollouts/datasets/lane_keeping \
#   --dst-dir ../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve \
#   --fraction 0.10 \
#   --curve-ratio 0.60 \
#   --steer-idx 1 \
#   --steer-threshold 0.15 \
#   --min-curve-steps 3