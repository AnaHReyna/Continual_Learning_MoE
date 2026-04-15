# tasks/lane_keeping.py

class LaneKeepingTask:
    name = "lane_keeping"

    def configure_env(self, cfg):
        cfg.num_npc_vehicles = 0
        return cfg

    def on_reset(self, env):
        # Pure lane keeping: no special scenario.
        # optionally reset NPCs
        pass

    def after_tick(self, env):
        # no extra scenery
        pass

    def compute_reward_done(self, env, info):
        finish = info["route_finish"]
        collision = info["collision"]
        off_route = info["off_route"]
        stuck = info["stuck"]

        if finish or collision or off_route or stuck:
            done = True
        else:
            done = False

        if finish:
            done_reason = "finish"
        elif collision:
            done_reason = "collision"
        elif off_route:
            done_reason = "off_route"
        elif stuck:
            done_reason = "stuck"
        else:
            done_reason = None

        reward = 0.0

        # progress
        reward += 3.0 * info["progress_delta"]
        reward += 0.03 * info["dist_delta"]

        # stay in the lane and aligned
        reward -= 0.04 * abs(info["lateral_error"])
        reward -= 0.015 * abs(info["heading_error"])

        # improve centering
        reward += 0.08 * info["lateral_improvement"]

        # smoothness
        reward -= 0.02 * abs(info["control_steer"])
        reward -= 0.06 * info["steer_delta"]

        # speed
        if 6.0 <= info["speed_kmh"] <= env.cfg.target_speed_kmh:
            reward += 0.01
        elif info["speed_kmh"] < 2.0:
            reward -= 0.01

        # terminal bonuses / penalties
        if finish:
            reward += 1.0
        if collision:
            reward -= 2.0
        if off_route:
            reward -= 1.0
        if stuck:
            reward -= 0.5

        task_info = {"finish": finish,
                     "done_reason": done_reason,
                     "task_name": self.name,
                    }
        
        return reward, done, task_info