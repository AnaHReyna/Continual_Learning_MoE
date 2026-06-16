# tasks/pedestrian.py

from collections import deque
from multiprocessing.util import info
import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing


class PedestrianScenarioConfig:
    def __init__(self, trigger_point, route, target_speed_kmh):

        self.trigger_points = [trigger_point]
        self.route = route
        self.target_speed_kmh = target_speed_kmh
        self.name = "PedestrianCrossing"
        self.town = None
        self.weather = carla.WeatherParameters().ClearNoon
        self.friction = None


class PedestrianTask:
    name = "pedestrian"

    def __init__(self,
                 curriculum_level=0,
                 auto_curriculum=True, # for train
                 # auto_curriculum=False, # for eval
                 max_level=3,
                 window_size=50,
                 promote_threshold=0.75,
                 demote_threshold=0.45,
                 cooldown_episodes=25,
                 allow_demotion=True,
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

        self.trigger_range = (0.05, 0.15)

    
    def _apply_curriculum(self, env):
        """
        Here the curriculum controls:
        - background traffic
        - target speed
        - approximate position of the scenario trigger along the route
        """
    
        if self.curriculum_level == 0:
            env.cfg.num_npc_vehicles = 0
            env.cfg.target_speed_kmh = 25.0
            self.trigger_range = (0.05, 0.12)

        elif self.curriculum_level == 1:
            env.cfg.num_npc_vehicles = 1
            env.cfg.target_speed_kmh = 30.0
            self.trigger_range = (0.05, 0.13)

        elif self.curriculum_level == 2:
            env.cfg.num_npc_vehicles = 2
            env.cfg.target_speed_kmh = 40.0
            self.trigger_range = (0.06, 0.15)

        else:
            env.cfg.num_npc_vehicles = 1
            env.cfg.target_speed_kmh = 30.0
            # env.cfg.target_speed_kmh = 30.0
            self.trigger_range = (0.08, 0.18)

        print(f"[TASK] level={self.curriculum_level} "
              f"num_npc={env.cfg.num_npc_vehicles} "
              f"target_speed={env.cfg.target_speed_kmh}"
              f"trigger_range={self.trigger_range}"
            )


    def configure_env(self, cfg):
        if self.curriculum_level == 0:
            cfg.num_npc_vehicles = 0
            cfg.target_speed_kmh = 20.0
        elif self.curriculum_level == 1:
            cfg.num_npc_vehicles = 1
            cfg.target_speed_kmh = 30.0
        elif self.curriculum_level == 2:
            cfg.num_npc_vehicles = 3
            cfg.target_speed_kmh = 40.0
        else:
            cfg.num_npc_vehicles = 4
            cfg.target_speed_kmh = 50.0
        return cfg


    # def on_reset(self, env):
    #     self._apply_curriculum(env)

    #     CarlaDataProvider.set_client(env.client)
    #     CarlaDataProvider.set_world(env.world)
    #     CarlaDataProvider.set_traffic_manager_port(env.cfg.traffic_manager_port)
    #     CarlaDataProvider.register_actor(env.ego)

    #     # fraction = env._rng.uniform(0.05, 0.15)
    #     fraction = env._rng.uniform(*self.trigger_range)
    #     idx = int(fraction * len(env.route_waypoints))
    #     idx = max(1, min(idx, len(env.route_waypoints) - 1))
    #     trigger_point = env.route_waypoints[idx]

    #     config = PedestrianScenarioConfig(trigger_point=trigger_point, route=env.route_dense)

    #     env.scenario = PedestrianCrossing(world=env.world,
    #                                       ego_vehicles=[env.ego],
    #                                       config=config,
    #                                       debug_mode=False,
    #                                       criteria_enable=True,
    #                                       timeout=60,
    #                                       )

    #     env.scenario_tree = env.scenario.scenario.scenario_tree
    #     env.scenario_criteria = env.scenario.criteria_list


    # def on_reset(self, env):
    #     self._apply_curriculum(env)

    #     CarlaDataProvider.set_client(env.client)
    #     CarlaDataProvider.set_world(env.world)
    #     CarlaDataProvider.set_traffic_manager_port(env.cfg.traffic_manager_port)
    #     CarlaDataProvider.register_actor(env.ego)

    #     env.scenario = None
    #     env.scenario_tree = None
    #     env.scenario_criteria = []

    #     max_attempts = 5
    #     scenario_ok = False

    #     for attempt in range(max_attempts):
    #         fraction = env._rng.uniform(*self.trigger_range)
    #         idx = int(fraction * len(env.route_waypoints))
    #         idx = max(1, min(idx, len(env.route_waypoints) - 1))
    #         trigger_point = env.route_waypoints[idx]

    #         config = PedestrianScenarioConfig(trigger_point=trigger_point,
    #                                           route=env.route_dense,
    #                                           )

    #         scenario = PedestrianCrossing(world=env.world,
    #                                       ego_vehicles=[env.ego],
    #                                       config=config,
    #                                       debug_mode=False,
    #                                       criteria_enable=True,
    #                                       timeout=60,
    #                                     )

    #         num_peds = len(getattr(scenario, "other_actors", []))
    #         print(f"[PedestrianTask] spawn attempt {attempt + 1}/{max_attempts} -> pedestrians={num_peds}")

    #         if num_peds > 0:
    #             env.scenario = scenario
    #             env.scenario_tree = scenario.scenario.scenario_tree
    #             env.scenario_criteria = scenario.criteria_list
    #             scenario_ok = True
    #             break

    #         try:
    #             scenario.remove_all_actors()
    #         except Exception:
    #             pass

    #     if not scenario_ok:
    #         raise RuntimeError(
    #             f"PedestrianTask could not create a valid scenario with pedestrians after {max_attempts} attempts.")


    def on_reset(self, env):
        self._apply_curriculum(env)

        CarlaDataProvider.set_client(env.client)
        CarlaDataProvider.set_world(env.world)
        CarlaDataProvider.set_traffic_manager_port(env.cfg.traffic_manager_port)
        CarlaDataProvider.register_actor(env.ego)

        env.scenario = None
        env.scenario_tree = None
        env.scenario_criteria = []
        env.skip_episode = False
        env.skip_reason = None

        # =====================================================
        # PULAR ROTAS CURTAS
        # =====================================================
        route_len = getattr(env, "_route_total_length", None)

        if route_len is None:
            try:
                env._ensure_route_cache()
                route_len = env._route_total_length
            except Exception:
                route_len = None

        if route_len is not None and route_len < 58.0:
            print("[PedestrianTask] skipping short route:",
                  "route=", getattr(env, "route_name", "unknown"),
                  "id=", getattr(env, "route_id", "unknown"),
                  "length=", route_len,
                )

            env.skip_episode = True
            env.skip_reason = "skip_short_route"
            return

        max_attempts = 1

        for attempt in range(max_attempts):
            fraction = env._rng.uniform(*self.trigger_range)
            idx = int(fraction * len(env.route_waypoints))
            idx = max(1, min(idx, len(env.route_waypoints) - 1))
            trigger_point = env.route_waypoints[idx]

            config = PedestrianScenarioConfig(trigger_point=trigger_point, 
                                              route=env.route_dense, 
                                              target_speed_kmh=env.cfg.target_speed_kmh,
                                              )

            scenario = PedestrianCrossing(world=env.world,
                                          ego_vehicles=[env.ego],
                                          config=config,
                                          debug_mode=False,
                                          criteria_enable=True,
                                          timeout=60,
                                         )
            
            # distancia = trigger_point - scenario.delta_dist

            num_peds = len(getattr(scenario, "other_actors", []))
            print(f"[PedestrianTask] attempt {attempt + 1}/{max_attempts} -> pedestrians={num_peds}")

            if num_peds > 0:
                env.scenario = scenario
                env.scenario_tree = scenario.scenario.scenario_tree
                env.scenario_criteria = scenario.criteria_list
                return

            try:
                scenario.remove_all_actors()
            except Exception:
                pass

        # print("[PedestrianTask] WARNING: no pedestrians spawned, skipping this episode")
        # env.skip_episode = True


        print("[PedestrianTask] WARNING: no pedestrians spawned, continuing without pedestrian")
        env.skip_episode = False
        env.skip_reason = None
        env.scenario = None
        env.scenario_tree = None
        env.scenario_criteria = []


    def record_episode_result(self, success):
        if not self.auto_curriculum:
            return

        if success is None:
            return

        self.recent_success.append(1.0 if success else 0.0)
        self.episodes_since_change += 1

        if len(self.recent_success) < self.window_size:
            return

        if self.episodes_since_change < self.cooldown_episodes:
            return

        success_rate = sum(self.recent_success) / len(self.recent_success)

        if success_rate >= self.promote_threshold and self.curriculum_level < self.max_level:
            self.curriculum_level += 1
            self.episodes_since_change = 0
            self.recent_success.clear()
            print(f"[Curriculum] upgraded pedestrian to level {self.curriculum_level}")
            return

        if self.allow_demotion and success_rate <= self.demote_threshold and self.curriculum_level > 0:
            self.curriculum_level -= 1
            self.episodes_since_change = 0
            self.recent_success.clear()
            print(f"[Curriculum] pedestrian dropped to level {self.curriculum_level}")


    def after_tick(self, env):
        if env.scenario_tree is not None:
            env.scenario_tree.tick_once()


    def _scenario_status(self, env):
        if env.scenario_tree is None:
            return False, False, {}

        status = env.scenario_tree.status
        if status == "SUCCESS":
            return True, False, {"scenario_event": "success"}
        if status == "FAILURE":
            return False, True, {"scenario_event": "failure"}
        return False, False, {}


    def compute_reward_done(self, env, info):
        if getattr(env, "skip_episode", False):
            task_info = {"finish": False,
                         "scenario_success": False,
                         "scenario_failure": False,
                         # "done_reason": "skip_no_pedestrian",
                         "done_reason": getattr(env, "skip_reason", "skip_no_pedestrian"),
                         "task_name": self.name,
                         "skip_episode": True,
                        }
            
            return 0.0, True, task_info
    
        scenario_success, scenario_failure, scenario_info = self._scenario_status(env)

        route_finish = info["route_finish"]
        collision = info["collision"]
        off_route = info["off_route"]
        stuck = info["stuck"]

        finish = route_finish and (not scenario_failure) and (not collision)
        done = bool(finish or collision or off_route or stuck or scenario_failure)

        if finish:
            done_reason = "finish"
        elif scenario_failure:
            done_reason = "scenario_failure"
        elif collision:
            done_reason = "collision"
        elif off_route:
            done_reason = "off_route"
        elif stuck:
            done_reason = "stuck"
        else:
            done_reason = None

        reward = 0.0

        # basic driving
        reward += 3.0 * info["progress_delta"]
        reward += 0.03 * info["dist_delta"]
        # reward += 1.2 * info["progress_delta"]
        # reward += 0.01 * info["dist_delta"]
        reward -= 0.04 * abs(info["lateral_error"])
        reward -= 0.015 * abs(info["heading_error"])
        reward += 0.08 * info["lateral_improvement"]
        # reward -= 0.03 * abs(info["control_steer"]) 
        # reward -= 0.08 * info["steer_delta"]
        reward -= 0.02 * abs(info["control_steer"])
        reward -= 0.06 * info["steer_delta"]

        if abs(info["control_steer"]) > 0.25:
            reward -= 0.15 * (abs(info["control_steer"]) - 0.25)

        speed = info["speed_kmh"]
        ped_dist = info.get("pedestrian_dist", float("inf"))
        ped_active = info.get("pedestrian_active", False)
        ped_conflict = info.get("pedestrian_to_conflict_dist", float("inf"))

        # Pawn nearby, but no major conflict yet.
        if ped_dist < 10.0 and speed > 10.0:
            reward -= 0.02 * (speed - 10.0)

        # Pawn in motion near the conflict zone
        if ped_active and ped_conflict < 3.0:
            if speed > 6.0:
                reward -= 0.08 * (speed - 6.0)
            if speed <= 4.0:
                reward += 0.10

        # conflict very close
        if ped_active and ped_dist < 6.0 and ped_conflict < 2.0:
            if speed > 3.0:
                reward -= 0.12 * (speed - 3.0)
            if speed <= 2.0:
                reward += 0.15

        # speed = info["speed_kmh"]
        # ped_dist = info.get("pedestrian_dist", float("inf"))
        # ped_active = info.get("pedestrian_active", False)
        # ped_conflict = info.get("pedestrian_to_conflict_dist", float("inf"))

        # # heavier steering penalty
        # reward -= 0.06 * abs(info["control_steer"])
        # reward -= 0.20 * info["steer_delta"]
        # if abs(info["control_steer"]) > 0.30:
        #     reward -= 0.10 * (abs(info["control_steer"]) - 0.30)

        # # Pawn visible/near: don't drive fast
        # if ped_dist < 10.0 and speed > 8.0:
        #     reward -= 0.04 * (speed - 8.0)

        # # Pawn in motion near the conflict zone: should almost stop
        # if ped_active and ped_conflict < 3.0:
        #     if speed > 4.0:
        #         reward -= 0.15 * (speed - 4.0)
        #     else:
        #         reward += 0.08

        # # conflict very close: should stop
        # if ped_active and ped_dist < 6.0 and ped_conflict < 2.0:
        #     if speed > 2.0:
        #         reward -= 0.25 * (speed - 2.0)
        #     else:
        #         reward += 0.15

        # bonus/penalty of the scenario
        if scenario_success:
            reward += 1.5
        if scenario_failure:
            reward -= 2.0

        # terminals
        if finish:
            reward += 2.0
        if collision:
            reward -= 3.5
        if off_route:
            reward -= 1.0
        if stuck:
            reward -= 0.5

        task_info = {"finish": finish,
                     "scenario_success": scenario_success,
                     "scenario_failure": scenario_failure,
                     "done_reason": done_reason,
                     "task_name": self.name,
                     "pedestrian_dist": ped_dist,
                     "pedestrian_active": ped_active,
                     "pedestrian_to_conflict_dist": ped_conflict,
                    }
        
        task_info.update(scenario_info)

        return reward, done, task_info



# task = PedestrianTask(
#     curriculum_level=0,
#     auto_curriculum=True,
#     max_level=3,
#     window_size=30,
#     promote_threshold=0.65,
#     cooldown_episodes=10,
#     allow_demotion=False,
# )

# env = CarlaRouteEnv(cfg, task=task)