from collections import deque


class LaneKeepingTask:
    name = "lane_keeping"

    def __init__(self,
                curriculum_level=0,
                auto_curriculum=True,
                max_level=3,
                window_size=30,
                promote_threshold=0.65,
                demote_threshold=0.35,   # opcional
                cooldown_episodes=10,
                allow_demotion=False,
                ):
        
        self.curriculum_level = curriculum_level
        self.auto_curriculum = auto_curriculum
        self.max_level = max_level
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.cooldown_episodes = cooldown_episodes
        self.allow_demotion = allow_demotion

        self.recent_success = deque(maxlen=window_size)
        self.episodes_since_change = 0


    def _apply_curriculum(self, env):
        """
        Define the difficulty of the current episode.
        """
        if self.curriculum_level == 0:
            env.cfg.num_npc_vehicles = 0
            env.cfg.target_speed_kmh = 25.0

        elif self.curriculum_level == 1:
            env.cfg.num_npc_vehicles = 1
            env.cfg.target_speed_kmh = 25.0


        elif self.curriculum_level == 2:
            env.cfg.num_npc_vehicles = 2
            env.cfg.target_speed_kmh = 30.0


        else:
            env.cfg.num_npc_vehicles = 6
            env.cfg.target_speed_kmh = 50.0


        print(f"[TASK] level={self.curriculum_level} "
                f"num_npc={env.cfg.num_npc_vehicles} "
                f"target_speed={env.cfg.target_speed_kmh}"
            )


    def configure_env(self, cfg):
        """
        Optional: initial setup before the first reset.
        """
        # remains consistent with the current curriculum.
        if self.curriculum_level == 0:
            cfg.num_npc_vehicles = 0
            cfg.target_speed_kmh = 20.0
        elif self.curriculum_level == 1:
            cfg.num_npc_vehicles = 2
            cfg.target_speed_kmh = 30.0
        elif self.curriculum_level == 2:
            cfg.num_npc_vehicles = 4
            cfg.target_speed_kmh = 40.0
        else:
            cfg.num_npc_vehicles = 6
            cfg.target_speed_kmh = 50.0
        return cfg

    def on_reset(self, env):
        """
        Adjust the task context before the episode begins.
        """
        self._apply_curriculum(env)

    def after_tick(self, env):
        # no extra scenario in lane keeping
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

        # maintain lane and alignment
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


        # safety with nearest traffic vehicle
        # nearest_dist = info.get("nearest_vehicle_dist", float("inf"))
        # nearest_ttc = info.get("nearest_vehicle_ttc", float("inf"))


        lead_dist = info.get("lead_vehicle_dist", float("inf"))
        lead_ttc = info.get("lead_vehicle_ttc", float("inf"))


        if lead_dist < 8.0:
            reward -= 0.006 * (8.0 - lead_dist)

        if lead_dist < 4.0:
            reward -= 0.02 * (4.0 - lead_dist)

        if lead_ttc < 3.0:
            reward -= 0.015 * (3.0 - lead_ttc)

        if lead_ttc < 1.5:
            reward -= 0.05 * (1.5 - lead_ttc)

        # penalty for dangerous proximity
        # if nearest_dist < 10.0:
            # reward -= 0.01 * (10.0 - nearest_dist)

        # Heavier penalty if you are too close.
        # if nearest_dist < 5.0:
            # reward -= 0.03 * (5.0 - nearest_dist)

        # penalty for low TTC
        # if nearest_ttc < 4.0:
            # reward -= 0.02 * (4.0 - nearest_ttc)

        # if nearest_ttc < 2.0:
            # reward -= 0.08 * (2.0 - nearest_ttc)


        # terminal
        if finish:
            reward += 1.0
        if collision:
            reward -= 3.0
        if off_route:
            reward -= 1.0
        if stuck:
            reward -= 0.5

        task_info = {"finish": finish,
                     "done_reason": done_reason,
                     "task_name": self.name,
                     "curriculum_level": self.curriculum_level,
                    }

        return reward, done, task_info

    def record_episode_result(self, success: bool):
        """
        Call at the end of each episode.
        """
        if not self.auto_curriculum:
            return


        if success:
            self.recent_success.append(1.0)
        else:
            self.recent_success.append(0.0)


        self.episodes_since_change += 1

        if len(self.recent_success) < self.window_size:
            return

        if self.episodes_since_change < self.cooldown_episodes:
            return

        success_rate = sum(self.recent_success) / len(self.recent_success)

        # level up
        if success_rate >= self.promote_threshold and self.curriculum_level < self.max_level:
            self.curriculum_level += 1
            self.episodes_since_change = 0
            self.recent_success.clear()
            print(f"[Curriculum] [Curriculum] has been upgraded to level {self.curriculum_level}")
            return

        # Go down level (optional)
        if self.allow_demotion and success_rate <= self.demote_threshold and self.curriculum_level > 0:
            self.curriculum_level -= 1
            self.episodes_since_change = 0
            self.recent_success.clear()
            print(f"[Curriculum] dropped to level {self.curriculum_level}")



# task = LaneKeepingTask(curriculum_level=0,
#                        auto_curriculum=True,
#                        max_level=3,
#                        window_size=30,
#                        promote_threshold=0.85,
#                        cooldown_episodes=10,
#                        allow_demotion=False,
#                       )


# task = LaneKeepingTask(curriculum_level=0,
#                        auto_curriculum=True,
#                        max_level=3,
#                        window_size=5,
#                        promote_threshold=0.05,
#                        cooldown_episodes=1,
#                        allow_demotion=False,
#                       )