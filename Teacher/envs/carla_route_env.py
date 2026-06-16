# envs/carla_route_env.py
import sys
from unicodedata import name
sys.path.append("/home/ana/Documents/Architecture_Transformers_SR/scenario_runner")
sys.path.append("/home/ana/CARLA_0.9.13/PythonAPI/carla")

import math
import queue
import random
import weakref
from collections import deque
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from srunner.tools.route_parser import RouteParser
from srunner.tools.route_manipulation import interpolate_trajectory


class EnvConfig:
    host = "127.0.0.1"
    port = 2000
    traffic_manager_port = 8000
    timeout = 120.0

    route_file = "/home/ana/Documents/Continual_Learning_MoE/Teacher/envs/generate_XML/routes_Town10HD_Opt.xml" 
    route_id = None
    route_town = "Town10HD_Opt"

    sync = True
    fixed_delta_seconds = 0.05

    max_episode_steps = 300
    target_speed_kmh = 25.0
    ego_filter = "vehicle.lincoln.mkz_2017"
    seed = 1

    render_rgb_camera = False
    front_camera_width = 640
    front_camera_height = 360
    front_camera_fov = 70

    spectator_follow = False
    spectator_height_m = 40.0
    spectator_rotate_with_ego = False

    show_bev = False
    bev_width = 800
    bev_height = 800
    bev_fov = -90
    bev_height_m = 35.0

    max_neighbors = 5
    num_npc_vehicles = 0
    spawn_radius_m = 30.0
    nearby_npc_radius_m = 30.0

    im_width = 120
    im_height = 160

    spectator_follow = True


class CollisionSensor:
    def __init__(self, parent_actor: carla.Actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    # @staticmethod
    # def _on_collision(weak_self, event):
    #     self = weak_self()
    #     if not self:
    #         return
    #     impulse = event.normal_impulse
    #     intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
    #     self.history.append((event.frame, intensity))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        impulse = event.normal_impulse
        intensity = math.sqrt(
            impulse.x ** 2 +
            impulse.y ** 2 +
            impulse.z ** 2
        )

        # print(
        #     "[COLLISION EVENT]",
        #     "frame=", event.frame,
        #     "actor=", event.other_actor.type_id,
        #     "intensity=", intensity
        # )

        self.history.append(
            (event.frame, intensity)
        )

    def clear(self):
        self.history.clear()

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
            try:
                self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None


class LaneInvasionSensor:
    def __init__(self, parent_actor: carla.Actor):
        self.sensor = None
        self.count = 0
        self._parent = parent_actor

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.count += 1

    def clear(self):
        self.count = 0

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
            try:
                self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None


class CarlaRouteEnv(object):
    def __init__(self, cfg: EnvConfig, task=None):
        self.cfg = cfg
        self.task = task
        self._rng = random.Random(cfg.seed)

        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None

        self.route_configs = []
        self.route_config = None
        self.route_dense = []
        self.route_waypoints = []
        self.route_index = 0

        self.ego = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.sensor_seg = None

        self.actor_handles = []
        self.npc_vehicles = []

        self.step_count = 0
        self.last_location = None
        self.last_progress = 0.0
        self.prev_dist_to_goal = None
        self.prev_lateral_error = 0.0
        self.stuck_steps = 0

        self.ego_history = deque(maxlen=10)
        self.neighbor_history = {}

        self.im_width = self.cfg.im_width
        self.im_height = self.cfg.im_height
        self.CAMERA_POS_Z = 50.0
        self.CAMERA_POS_X = 0.0

        self.image_seg = None
        self.image_for_CNN = None

        self.blueprint_library = None
        self.sem_cam = None
        self.cnn_model = load_model("/home/ana/Documents/Continual_Learning_MoE/Teacher/envs/train_CNN_image/CNN_image_model.h5", compile=False,)

        self.prev_steer = 0.0
        self.target_speed = self.cfg.target_speed_kmh

        self._route_xy = None
        self._route_s = None
        self._route_total_length = None

        # debug drawing
        self.draw_debug_routes = True

        # Alturas dos desenhos no mundo.
        # A rota XML fica um pouco mais baixa; a trajetória real do ego fica acima,
        # para aparecer melhor na captura de tela.
        self.debug_route_z = 0.5
        self.debug_policy_z = 1.5

        # Tempo de vida dos desenhos no CARLA.
        # IMPORTANTE:
        # O CARLA DebugHelper não tem um comando simples para apagar linhas já desenhadas.
        # Por isso, a trajetória do ego NÃO pode usar life_time muito alto, senão
        # as linhas de episódios antigos ficam marcadas no mapa.
        #
        # Solução usada aqui:
        # - a trajetória do episódio atual é redesenhada completa a cada step;
        # - cada desenho vive pouco tempo;
        # - quando começa outro episódio, a trajetória anterior some sozinha.
        self.debug_draw_life_time = 1.0
        self.debug_policy_life_time = 1.0

        # A cada quantos steps a trajetória completa será redesenhada.
        # 1 = mais bonito/estável para screenshot, mas desenha mais linhas.
        self.debug_policy_redraw_every_steps = 1

        # Histórico completo da trajetória realmente percorrida pelo ego.
        self.policy_route_history: List[carla.Location] = []

        self.scenario = None
        self.scenario_tree = None
        self.scenario_criteria = []



###############################################################################################################################################
    def _connect(self):
        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(self.cfg.timeout)
        self.traffic_manager = self.client.get_trafficmanager(self.cfg.traffic_manager_port)


     # ---------------------------
    # CNN / vision
    # ---------------------------
    def apply_cnn(self, im):
        """
        Applies inference to the image. 
        Generates an embedding.
        """

        img = np.float32(im) / 255.0
        img = cv2.resize(img, (self.im_height, self.im_width,))         # (120,160,3)
        img = tf.convert_to_tensor(np.expand_dims(img, axis=0))  #  (1,120,160,3)

        cnn_applied = self.cnn_model(img, training=False)         # (1,280)
        cnn_applied = np.squeeze(cnn_applied)  #  (280,)

        if cnn_applied.shape[0] != 280:
            raise ValueError(f"CNN returned shape {cnn_applied.shape}, but RL waits (280,). "
                             f"Make sure you loaded the embedding model, not the regression model."
                            )

        return cnn_applied
    

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # -> (240,320,3)
        self.image_seg = i
    # ========================================================================================================

    # -------------------------------
    # debug drawing
    # -------------------------------
    def _dbg_loc(self, loc: carla.Location, z_offset: float) -> carla.Location:
        return carla.Location(x=loc.x, y=loc.y, z=loc.z + z_offset)
    

    def _draw_xml_route(self, life_time: float = None):
        if life_time is None:
            life_time = self.debug_draw_life_time

        if self.world is None or len(self.route_waypoints) < 2:
            return

        dbg = self.world.debug

        # Rota XML/reference route. Deixei cinza para não confundir com
        # a trajetória real do ego, que será desenhada em verde.
        route_color = carla.Color(180, 180, 180)  # gray
        start_color = carla.Color(0, 0, 255)      # blue
        end_color = carla.Color(255, 0, 0)        # red
        dir_color = carla.Color(255, 255, 0)      # yellow

        for i in range(len(self.route_waypoints) - 1):
            a = self._dbg_loc(self.route_waypoints[i].location, self.debug_route_z)
            b = self._dbg_loc(self.route_waypoints[i + 1].location, self.debug_route_z)
            dbg.draw_line(a, b, thickness=0.12, color=route_color, life_time=life_time,)

        start_loc = self._dbg_loc(self.route_waypoints[0].location, self.debug_route_z + 0.3)
        end_loc = self._dbg_loc(self.route_waypoints[-1].location, self.debug_route_z + 0.3)

        dbg.draw_point(start_loc, size=0.16, color=start_color, life_time=life_time)
        dbg.draw_point(end_loc, size=0.16, color=end_color, life_time=life_time)

        dbg.draw_string(start_loc + carla.Location(z=0.35),
                        "START XML",
                        draw_shadow=False,
                        color=start_color,
                        life_time=life_time,
                        )

        dbg.draw_string(end_loc + carla.Location(z=0.35),
                        "GOAL XML",
                        draw_shadow=False,
                        color=end_color,
                        life_time=life_time,
                        )

        step = max(1, len(self.route_waypoints) // 20)
        for i in range(0, len(self.route_waypoints) - 1, step):
            a = self._dbg_loc(self.route_waypoints[i].location, self.debug_route_z + 0.15)
            b = self._dbg_loc(self.route_waypoints[min(i + 1, len(self.route_waypoints) - 1)].location, self.debug_route_z + 0.15)            
            dbg.draw_arrow(a, b, thickness=0.08, arrow_size=0.18, color=dir_color, life_time=life_time,)


    def _reset_policy_route_history(self):
        """Inicializa o histórico da trajetória real do ego no episódio atual."""
        self.policy_route_history = []

        if self.ego is not None and self.ego.is_alive:
            loc = self.ego.get_location()
            self.policy_route_history.append(
                carla.Location(x=loc.x, y=loc.y, z=loc.z)
            )


    def _draw_policy_start_marker(self, life_time: float = None):
        """Marca o ponto inicial da trajetória real do ego."""
        if life_time is None:
            life_time = self.debug_policy_life_time

        if self.world is None or len(self.policy_route_history) == 0:
            return

        dbg = self.world.debug
        start_loc = self._dbg_loc(
            self.policy_route_history[0],
            self.debug_policy_z + 0.20,
        )

        dbg.draw_point(
            start_loc,
            size=0.22,
            color=carla.Color(0, 0, 255),  # azul
            life_time=life_time,
        )

        dbg.draw_string(
            start_loc + carla.Location(z=0.35),
            "EGO START",
            draw_shadow=False,
            color=carla.Color(0, 0, 255),
            life_time=life_time,
        )


    def _draw_policy_end_marker(self, life_time: float = None):
        """Marca a posição final/atual da trajetória real do ego."""
        if life_time is None:
            life_time = self.debug_policy_life_time

        if self.world is None or len(self.policy_route_history) == 0:
            return

        dbg = self.world.debug
        end_loc = self._dbg_loc(
            self.policy_route_history[-1],
            self.debug_policy_z + 0.20,
        )

        dbg.draw_point(
            end_loc,
            size=0.22,
            color=carla.Color(255, 0, 0),  # vermelho
            life_time=life_time,
        )

        dbg.draw_string(
            end_loc + carla.Location(z=0.35),
            "EGO END",
            draw_shadow=False,
            color=carla.Color(255, 0, 0),
            life_time=life_time,
        )


    def _draw_policy_route_incremental(
        self,
        prev_loc: carla.Location,
        curr_loc: carla.Location,
        life_time: float = None,
    ):
        """Desenha um novo segmento da trajetória real percorrida pelo ego."""
        if life_time is None:
            life_time = self.debug_policy_life_time

        if self.world is None or prev_loc is None or curr_loc is None:
            return

        dbg = self.world.debug
        color = carla.Color(0, 255, 0)  # VERDE = trajetória real do ego

        a = self._dbg_loc(prev_loc, self.debug_policy_z)
        b = self._dbg_loc(curr_loc, self.debug_policy_z)

        dbg.draw_line(
            a,
            b,
            thickness=0.18,
            color=color,
            life_time=life_time,
        )

        dbg.draw_point(
            b,
            size=0.12,
            color=color,
            life_time=life_time,
        )


    def _draw_full_policy_route(self, life_time: float = None):
        """Redesenha a trajetória completa do ego acumulada no episódio."""
        if life_time is None:
            life_time = self.debug_policy_life_time

        if self.world is None or len(self.policy_route_history) < 2:
            return

        for i in range(len(self.policy_route_history) - 1):
            self._draw_policy_route_incremental(
                self.policy_route_history[i],
                self.policy_route_history[i + 1],
                life_time=life_time,
            )

        self._draw_policy_start_marker(life_time=life_time)
        self._draw_policy_end_marker(life_time=life_time)

    # ============================================================================================================================


    
    # ---------------------------
    # Route setup
    # ---------------------------
    def _load_route_configs(self):
        self.route_configs = RouteParser.parse_routes_file(route_filename=self.cfg.route_file,
                                                           scenario_file=None,
                                                           single_route=self.cfg.route_id,
                                                           )
        if not self.route_configs:
            raise RuntimeError("No route found in the XML file.")




    def _choose_route(self): 
        if self.cfg.route_id is not None:
            self.route_config = self.route_configs[0]
            return

        if self.cfg.route_town is not None:
            valid_routes = []
            for rc in self.route_configs:
                if rc.town == self.cfg.route_town:
                    valid_routes.append(rc)

            if not valid_routes:
                raise RuntimeError(f"No route found for town {self.cfg.route_town}")            
            self.route_config = self._rng.choice(valid_routes)

        else:
            self.route_config = self._rng.choice(self.route_configs)


    def _load_world_for_route(self):
        assert self.client is not None
        assert self.route_config is not None

        town = self.route_config.town

        if self.world is None:
            need_load = True
        else:
            need_load = False

        if not need_load:
            try:
                current_map = self.world.get_map().name.split("/")[-1]
                if current_map != town:
                    need_load = True
                else:
                    need_load = False

            except Exception:
                need_load = True

        if need_load:
            self.world = self.client.load_world(town)
            try:
                self.world.unload_map_layer(carla.MapLayer.Buildings)
            except Exception:
                pass

        else:
            self.world = self.client.get_world()

        self.map = self.world.get_map()

        settings = self.world.get_settings()

        if self.cfg.sync:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
        else:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)


        
    # def _prepare_route(self):
    #     assert self.world is not None
    #     assert self.route_config is not None

    #     _, dense_route = interpolate_trajectory(self.world, self.route_config.trajectory)
    #     self.route_dense = dense_route

    #     self.route_waypoints = []
    #     for wp in self.route_dense:
    #         self.route_waypoints.append(wp[0])


    #     self._route_xy = None
    #     self._route_s = None
    #     self._route_total_length = None

    #     if len(self.route_waypoints) < 2:
    #         raise RuntimeError("The interpolated route was too short.")
        

    def _prepare_route(self):
        assert self.world is not None
        assert self.route_config is not None

        _, dense_route = interpolate_trajectory(self.world, self.route_config.trajectory)
        self.route_dense = dense_route
        print(f"Tamanho da trajetoria interpolada: {len(self.route_dense)} waypoints")

        self.route_waypoints = []
        for wp in self.route_dense:
            self.route_waypoints.append(wp[0])

        self._route_xy = None
        self._route_s = None
        self._route_total_length = None

        if len(self.route_waypoints) < 2:
            raise RuntimeError("The interpolated route was too short.")
        
        self._ensure_route_cache()  # compute the length real the route use in the episode, so we can filter out very short routes that are not useful for training.

        # print("DEBUG ROUTE CONFIG")
        # print(vars(self.route_config))

        route_name = getattr(self.route_config, "name", "unknown")
        route_id = getattr(self.route_config, "id", "unknown")
        town = getattr(self.route_config, "town", "unknown")

        self.route_name = route_name
        self.route_id = route_id

        # print(f"[ROUTE LENGTH] "
        #       f"name={route_name} id={route_id} town={town} "
        #       f"waypoints={len(self.route_waypoints)} "
        #       f"meters={self._route_total_length:.2f}"
        #       )
                    


    def _spawn_ego(self):
        assert self.world is not None
        assert self.map is not None
        assert self.route_config is not None
        assert len(self.route_waypoints) > 0

        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.find(self.cfg.ego_filter)
        ego_bp.set_attribute("role_name", "hero")

        candidate_indices = [0, 3, 5, 8, 10, 15]
        candidate_indices = [i for i in candidate_indices if i < len(self.route_waypoints)]

        self.ego = None

        for idx in candidate_indices:
            base_tf = self.route_waypoints[idx]

            for dz in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5]:
                spawn_transform = carla.Transform(base_tf.location, base_tf.rotation)
                spawn_transform.location.z += dz

                ego_candidate = self.world.try_spawn_actor(ego_bp, spawn_transform)

                if ego_candidate is None:
                    continue

                temp_collision = CollisionSensor(ego_candidate)

                for _ in range(3):
                    self.world.tick()

                has_collision = len(temp_collision.history) > 0

                if has_collision:
                    print(f"[EGO SPAWN BAD] idx={idx} dz={dz} collision={temp_collision.history}")

                    temp_collision.destroy()
                    try:
                        ego_candidate.destroy()
                    except Exception:
                        pass

                    for _ in range(2):
                        self.world.tick()

                    continue

                temp_collision.destroy()

                self.ego = ego_candidate
                self.route_index = idx
                self.actor_handles.append(self.ego)

                print(f"[EGO SPAWN OK] route_idx={idx} dz={dz}")

                self.collision_sensor = CollisionSensor(self.ego)
                self.lane_sensor = LaneInvasionSensor(self.ego)
                return

        # print("[SPAWN DEBUG] failed for route:", getattr(self, "route_name", "unknown"))
        # print("[SPAWN DEBUG] first waypoint:",
        #     self.route_waypoints[0].location,
        #     self.route_waypoints[0].rotation)

        raise RuntimeError("Failed to spawn the ego vehicle in safe route waypoint.")



    # def _spawn_ego(self):
    #     assert self.world is not None
    #     assert self.map is not None
    #     assert self.route_config is not None
    #     assert len(self.route_waypoints) > 0

    #     bp_lib = self.world.get_blueprint_library()
    #     ego_bp = bp_lib.find(self.cfg.ego_filter)
    #     ego_bp.set_attribute("role_name", "hero")

    #     # base_tf = self.route_waypoints[0]
    #     candidate_indices = [0, 3, 5, 8, 10]
    #     candidate_indices = [i for i in candidate_indices if i < len(self.route_waypoints)]

    #     self.ego = None

    #     for idx in candidate_indices:
    #         base_tf = self.route_waypoints[idx]

    #         for dz in [0.5, 0.8, 1.0, 1.5, 2.0]:
    #             spawn_transform = carla.Transform(base_tf.location, base_tf.rotation)
    #             spawn_transform.location.z += dz

    #             self.ego = self.world.try_spawn_actor(ego_bp, spawn_transform)

    #             if self.ego is not None:
    #                 self.route_index = idx
    #                 print(f"[EGO SPAWN] route_idx={idx}")
    #                 break

    #         if self.ego is not None:
    #             break

    #     self.ego = None
    #     for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    #         spawn_transform = carla.Transform(base_tf.location, base_tf.rotation)
    #         spawn_transform.location.z += dz
    #         self.ego = self.world.try_spawn_actor(ego_bp, spawn_transform)
    #         if self.ego is not None:
    #             break

    #     if self.ego is None:
    #         print("[SPAWN DEBUG] route waypoint 0:",
    #             base_tf.location, base_tf.rotation)
    #         raise RuntimeError("Failed to spawn the ego vehicle at the start of the route.")

    #     self.actor_handles.append(self.ego)

    #     self.collision_sensor = CollisionSensor(self.ego)
    #     self.lane_sensor = LaneInvasionSensor(self.ego)


    # ====================== Traffic Manager ====================================================

    def _find_nearest_route_index(self, location: carla.Location, window: int = 40) -> int:
        start = max(0, self.route_index - 5)
        end = min(len(self.route_waypoints), self.route_index + window)

        best_idx = self.route_index
        best_dist = float("inf")

        for i in range(start, end):
            d = self._distance(location, self.route_waypoints[i].location)
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx


    def _spawn_background_traffic(self):
        assert self.world is not None
        assert self.map is not None
        assert self.traffic_manager is not None
        assert self.ego is not None
        assert len(self.route_waypoints) > 0

        bp_lib = self.world.get_blueprint_library()

        ego_filter = getattr(self.cfg, "ego_filter", "vehicle.lincoln.mkz_2017")
        preferred_ids = [ego_filter,
                        "vehicle.lincoln.mkz_2020",
                        "vehicle.charger_2020",
                        "vehicle.audi.etron",
                        "vehicle.tesla.model3",
                        "vehicle.mercedes.coupe",
                        "vehicle.audi.a2",
                        "vehicle.nissan.patrol",
                        ]

        preferred_bps = []
        fallback_bps = []

        for bp in bp_lib.filter("vehicle.*"):
            if bp.has_attribute("number_of_wheels"):
                try:
                    if int(bp.get_attribute("number_of_wheels")) != 4:
                        continue
                except Exception:
                    continue

            if bp.id in preferred_ids:
                preferred_bps.append(bp)
            else:
                fallback_bps.append(bp)

        preferred_order = {bp_id: i for i, bp_id in enumerate(preferred_ids)}
        preferred_bps.sort(key=lambda bp: preferred_order.get(bp.id, 999))

        candidate_bps = preferred_bps + fallback_bps
        if not candidate_bps:
            print("[TRAFFIC] no vehicle blueprints found")
            return

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location

        current_idx = self._find_nearest_route_index(ego_loc)
        level = getattr(getattr(self, "task", None), "curriculum_level", 0)

        print(f"[TRAFFIC] level={level} requested={self.cfg.num_npc_vehicles} "
              f"ego=({ego_loc.x:.2f}, {ego_loc.y:.2f}, {ego_loc.z:.2f}) "
              f"route_idx={current_idx}"
              )

        def _loc_key(loc, precision=1):
            return (round(loc.x, precision), round(loc.y, precision), round(loc.z, precision))

        def _is_valid_driving_neighbor(base_wp, other_wp):
            if other_wp is None:
                return False
            if other_wp.lane_type != carla.LaneType.Driving:
                return False

            yaw_a = base_wp.transform.rotation.yaw
            yaw_b = other_wp.transform.rotation.yaw
            yaw_diff = abs((yaw_a - yaw_b + 180.0) % 360.0 - 180.0)
            return yaw_diff <= 45.0

        def _try_spawn_vehicle(tf, source_name):
            nonlocal failed_spawn_attempts

            for bp in candidate_bps[:8]:
                try:
                    if bp.has_attribute("role_name"):
                        bp.set_attribute("role_name", "autopilot")
                except Exception:
                    pass

                actor = self.world.try_spawn_actor(bp, tf)
                if actor is not None:
                    actor.set_autopilot(True, self.traffic_manager.get_port())
                    self.traffic_manager.auto_lane_change(actor, True)
                    self.traffic_manager.distance_to_leading_vehicle(actor, 3.0)
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        actor,
                        self._rng.randint(-15, 5)
                    )

                    self.traffic_manager.ignore_lights_percentage(actor, 100.0)

                    self.npc_vehicles.append(actor)
                    self.actor_handles.append(actor)

                    print(
                        f"[TRAFFIC] SPAWN OK src={source_name} bp={bp.id} "
                        f"loc=({tf.location.x:.2f},{tf.location.y:.2f},{tf.location.z:.2f})"
                    )
                    return True
                else:
                    failed_spawn_attempts += 1

            return False

        # --------------------------------------------------
        # 1) candidatos controlados a partir do waypoint do ego
        # --------------------------------------------------
        controlled_candidates = []
        used_keys = set()

        ego_wp = self.map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if ego_wp is not None:
            # spawn ao longo da própria pista do ego
            forward_dists = [5.0, 7.0, 9.0]

            for dist in forward_dists:
                next_wps = ego_wp.next(dist)
                if not next_wps:
                    continue

                base_wp = next_wps[0]
                wp_list = [("ego_lane", base_wp)]

                left_wp = base_wp.get_left_lane()
                if _is_valid_driving_neighbor(base_wp, left_wp):
                    wp_list.append(("left_lane", left_wp))

                right_wp = base_wp.get_right_lane()
                if _is_valid_driving_neighbor(base_wp, right_wp):
                    wp_list.append(("right_lane", right_wp))

                for source_name, wp in wp_list:
                    tf = carla.Transform(
                        carla.Location(
                            x=wp.transform.location.x,
                            y=wp.transform.location.y,
                            z=wp.transform.location.z + 0.30,
                        ),
                        wp.transform.rotation,
                    )

                    k = _loc_key(tf.location)
                    if k in used_keys:
                        continue
                    used_keys.add(k)
                    controlled_candidates.append((source_name, tf))

            # tenta também um carro um pouco atrás do ego
            prev_wps = ego_wp.previous(6.0)
            if prev_wps:
                back_wp = prev_wps[0]
                tf = carla.Transform(
                    carla.Location(
                        x=back_wp.transform.location.x,
                        y=back_wp.transform.location.y,
                        z=back_wp.transform.location.z + 0.30,
                    ),
                    back_wp.transform.rotation,
                )

                k = _loc_key(tf.location)
                if k not in used_keys:
                    used_keys.add(k)
                    controlled_candidates.append(("behind_ego", tf))

        # --------------------------------------------------
        # 2) candidatos adicionais pela rota à frente
        # --------------------------------------------------
        route_start = max(0, current_idx + 5)
        route_end = min(len(self.route_waypoints), current_idx + 35)
        route_step = 5

        for ridx in range(route_start, route_end, route_step):
            base_tf = self.route_waypoints[ridx]

            try:
                base_wp = self.map.get_waypoint(
                    base_tf.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            except Exception:
                base_wp = None

            if base_wp is None:
                continue

            wp_list = [("route_main", base_wp)]

            left_wp = base_wp.get_left_lane()
            if _is_valid_driving_neighbor(base_wp, left_wp):
                wp_list.append(("route_side", left_wp))

            right_wp = base_wp.get_right_lane()
            if _is_valid_driving_neighbor(base_wp, right_wp):
                wp_list.append(("route_side", right_wp))

            for source_name, wp in wp_list:
                tf = carla.Transform(
                    carla.Location(
                        x=wp.transform.location.x,
                        y=wp.transform.location.y,
                        z=wp.transform.location.z + 0.30,
                    ),
                    wp.transform.rotation,
                )

                k = _loc_key(tf.location)
                if k in used_keys:
                    continue
                used_keys.add(k)
                controlled_candidates.append((source_name, tf))

        print(f"[TRAFFIC] controlled_candidates={len(controlled_candidates)}")

        failed_spawn_attempts = 0
        spawned = 0

        # --------------------------------------------------
        # 3) primeiro tenta os candidatos controlados
        # --------------------------------------------------
        for source_name, tf in controlled_candidates:
            if spawned >= self.cfg.num_npc_vehicles:
                break

            d = self._distance(tf.location, ego_loc)

            # agora permite vizinhos mais próximos
            if d < 4.5 or d > 25.0:
                continue

            if _try_spawn_vehicle(tf, source_name):
                spawned += 1

        # --------------------------------------------------
        # 4) fallback com spawn points do mapa
        # --------------------------------------------------
        if spawned < self.cfg.num_npc_vehicles:
            map_candidates = []

            for sp in self.map.get_spawn_points():
                k = _loc_key(sp.location)
                if k in used_keys:
                    continue

                d = self._distance(sp.location, ego_loc)
                if d < 8.0 or d > 50.0:
                    continue

                map_candidates.append(sp)

            map_candidates = sorted(
                map_candidates,
                key=lambda sp: self._distance(sp.location, ego_loc)
            )

            print(f"[TRAFFIC] fallback_map_candidates={len(map_candidates)}")

            for sp in map_candidates:
                if spawned >= self.cfg.num_npc_vehicles:
                    break

                if _try_spawn_vehicle(sp, "map_spawn"):
                    spawned += 1

        print(
            f"[TRAFFIC] DONE requested={self.cfg.num_npc_vehicles} "
            f"spawned={spawned} alive={len([v for v in self.npc_vehicles if v.is_alive])} "
            f"failed_spawn_attempts={failed_spawn_attempts}"
        )
    
    # ==============================================================================================================


    # ---------------------------
    # Geometry / route metrics
    # ---------------------------
    def _warmup_ticks(self, n=10):
        assert self.world is not None
        for _ in range(n):
            self.world.tick()


    def _kmh(self, vel: carla.Vector3D) -> float:
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    

    def _distance(self, a: carla.Location, b: carla.Location) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    

    def _ensure_route_cache(self):
        if self._route_xy is not None and self._route_s is not None:
            return

        pts = []
        for tf in self.route_waypoints:
            loc = tf.location
            pts.append((float(loc.x), float(loc.y)))

        self._route_xy = np.asarray(pts, dtype=np.float32)
        n = len(self._route_xy)

        self._route_s = np.zeros((n,), dtype=np.float32)
        for i in range(1, n):
            dx = self._route_xy[i, 0] - self._route_xy[i - 1, 0]
            dy = self._route_xy[i, 1] - self._route_xy[i - 1, 1]
            self._route_s[i] = self._route_s[i - 1] + np.hypot(dx, dy)


        if n > 1:
            self._route_total_length = float(self._route_s[-1])
        else:
            self._route_total_length = 1.0


    def _find_nearest_route_segment(self, location: carla.Location, window: int = 40):
        self._ensure_route_cache()

        px = float(location.x)
        py = float(location.y)

        n = len(self.route_waypoints)
        if n < 2:
            return 0, 0.0, px, py, 0.0

        start = max(0, self.route_index - 5)
        end = min(n - 1, self.route_index + window)

        best_i = max(0, min(start, n - 2))
        best_t = 0.0
        best_proj_x = px
        best_proj_y = py
        best_dist2 = float("inf")

        for i in range(start, end):
            ax, ay = self._route_xy[i]
            bx, by = self._route_xy[i + 1]

            abx = bx - ax
            aby = by - ay
            ab2 = abx * abx + aby * aby

            if ab2 < 1e-8:
                t = 0.0
                proj_x, proj_y = ax, ay
            else:
                apx = px - ax
                apy = py - ay
                t = (apx * abx + apy * aby) / ab2
                t = float(np.clip(t, 0.0, 1.0))
                proj_x = ax + t * abx
                proj_y = ay + t * aby

            dx = px - proj_x
            dy = py - proj_y
            dist2 = dx * dx + dy * dy

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_t = t
                best_proj_x = proj_x
                best_proj_y = proj_y

        ax, ay = self._route_xy[best_i]
        bx, by = self._route_xy[best_i + 1]
        seg_yaw = math.atan2(by - ay, bx - ax)

        return best_i, best_t, best_proj_x, best_proj_y, seg_yaw
    

    def _compute_route_errors(self) -> Tuple[float, float, float, float]:
        self._ensure_route_cache()

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)

        px = float(ego_loc.x)
        py = float(ego_loc.y)

        seg_i, seg_t, proj_x, proj_y, seg_yaw = self._find_nearest_route_segment(ego_loc)

        ax, ay = self._route_xy[seg_i]
        bx, by = self._route_xy[seg_i + 1]

        abx = bx - ax
        aby = by - ay
        seg_len = max(np.hypot(abx, aby), 1e-6)

        ux = abx / seg_len
        uy = aby / seg_len

        rx = px - proj_x
        ry = py - proj_y
        lateral_error = -uy * rx + ux * ry

        heading_error = ego_yaw - seg_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        progress_s = float(self._route_s[seg_i] + seg_t * seg_len)
        progress = progress_s / max(self._route_total_length, 1e-6)
        progress = float(np.clip(progress, 0.0, 1.0))

        remaining_s = max(self._route_total_length - progress_s, 0.0)
        lateral_abs = float(np.hypot(rx, ry))
        dist_to_goal = remaining_s + lateral_abs

        if seg_t < 0.5:
            projected_index = seg_i
        else:
            projected_index = seg_i +1

        self.route_index = int(np.clip(projected_index, 0, len(self.route_waypoints) - 1))

        return lateral_error, heading_error, progress, dist_to_goal
    

    # ---------------------------
    # Observation
    # ---------------------------
    def _angle_norm(self, yaw_deg: float) -> float:
        yaw_rad = math.radians(yaw_deg)
        return (yaw_rad + math.pi) % (2 * math.pi) - math.pi

    def _actor_state(self, actor: carla.Actor) -> np.ndarray:
        tf = actor.get_transform()
        vel = actor.get_velocity()
        return np.array([tf.location.x, 
                         tf.location.y, 
                         self._angle_norm(tf.rotation.yaw), vel.x, vel.y
                         ], dtype=np.float32,
                        )
    
    def _get_candidate_neighbors(self):
        candidates = []
        ego_loc = self.ego.get_location()

        for actor in self.npc_vehicles:
            if actor is None or not actor.is_alive:
                continue
            d = self._distance(actor.get_location(), ego_loc)
            if d <= self.cfg.nearby_npc_radius_m:
                candidates.append(("vehicle", actor, d))

        if self.scenario is not None and hasattr(self.scenario, "other_actors"):
            for actor in self.scenario.other_actors:
                if actor is None or not actor.is_alive:
                    continue
                d = self._distance(actor.get_location(), ego_loc)
                if d <= self.cfg.nearby_npc_radius_m:
                    candidates.append(("scenario", actor, d))

        candidates.sort(key=lambda x: x[2])
        return candidates[:self.cfg.max_neighbors]
    


    def _record_histories(self):
        if self.ego is not None:
            self.ego_history.append(self._actor_state(self.ego))

        neighbors = self._get_candidate_neighbors()
        active_ids = set()

        for _, actor, _ in neighbors:
            aid = actor.id
            active_ids.add(aid)

            if aid not in self.neighbor_history:
                self.neighbor_history[aid] = deque(maxlen=10)

            self.neighbor_history[aid].append(self._actor_state(actor))

        dead_ids = [aid for aid in list(self.neighbor_history.keys()) if aid not in active_ids]
        for aid in dead_ids:
            del self.neighbor_history[aid]

    
    def _world_to_ego_xy(self, wx: float, wy: float, ego_x: float, ego_y: float, ego_yaw: float):
        dx = wx - ego_x
        dy = wy - ego_y

        c = math.cos(ego_yaw)
        s = math.sin(ego_yaw)

        x_local = c * dx + s * dy
        y_local = -s * dx + c * dy
        return x_local, y_local
    

    def _normalize_angle(self, a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi


    def _state_to_ego_frame(self, state: np.ndarray, ego_x: float, ego_y: float, ego_yaw: float) -> np.ndarray:
        x, y, yaw, vx, vy = [float(v) for v in state]

        x_local, y_local = self._world_to_ego_xy(x, y, ego_x, ego_y, ego_yaw)
        yaw_rel = self._normalize_angle(yaw - ego_yaw)

        c = math.cos(ego_yaw)
        s = math.sin(ego_yaw)

        vx_local = c * vx + s * vy
        vy_local = -s * vx + c * vy

        return np.array([x_local, y_local, yaw_rel, vx_local, vy_local], dtype=np.float32)
    

    def _get_structured_obs(self):
        self._record_histories()

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)

        ego_x = float(ego_loc.x)
        ego_y = float(ego_loc.y)

        # --------------------------------------------------
        # 1) neighbor_trajs in ego frame
        # shape maintained: (6, 10, 5)
        # --------------------------------------------------
        neighbor_trajs = np.zeros((6, 10, 5), dtype=np.float32)

        ego_hist = list(self.ego_history)
        if len(ego_hist) > 0:
            ego_hist_local = [self._state_to_ego_frame(st, ego_x, ego_y, ego_yaw) for st in ego_hist]
            neighbor_trajs[0, -len(ego_hist_local):] = np.asarray(ego_hist_local, dtype=np.float32)

        neighbors = self._get_candidate_neighbors()
        for i, (_, actor, _) in enumerate(neighbors[:5], start=1):
            hist = list(self.neighbor_history.get(actor.id, []))
            if len(hist) > 0:
                hist_local = [self._state_to_ego_frame(st, ego_x, ego_y, ego_yaw) for st in hist]
                neighbor_trajs[i, -len(hist_local):] = np.asarray(hist_local, dtype=np.float32)

        # --------------------------------------------------
        # 2) ego_info maintains shape (5,)
        # now already in the ego frame
        # --------------------------------------------------
        ego_info = neighbor_trajs[0, -1].copy().astype(np.float32)

        # --------------------------------------------------
        # 3) neighbor_waypoints local to ego
        # shape maintained: (18, 10, 2)
        # --------------------------------------------------
        neighbor_waypoints = np.zeros((18, 10, 2), dtype=np.float32)

        start_idx = min(self.route_index + 1, len(self.route_waypoints) - 1)
        max_blocks = 18
        step_between_blocks = 5
        pts_per_block = 10

        for i in range(max_blocks):
            pts = []

            block_start = start_idx + i * step_between_blocks
            block_end = min(block_start + pts_per_block, len(self.route_waypoints))

            if block_start >= len(self.route_waypoints):
                break

            for j in range(block_start, block_end):
                wp = self.route_waypoints[j].location

                x_local, y_local = self._world_to_ego_xy(float(wp.x), float(wp.y), ego_x, ego_y, ego_yaw)
                pts.append([x_local, y_local])

            if len(pts) > 0:
                neighbor_waypoints[i, :len(pts)] = np.asarray(pts, dtype=np.float32)

        return neighbor_trajs, ego_info, neighbor_waypoints
    


    # ---------------------------
    # Control
    # ---------------------------
    def action_adapter(self, model_action):
        speed_norm = float(np.clip(model_action[0], -1.0, 1.0))

        min_speed_kmh = 12.0
        max_speed_kmh = self.cfg.target_speed_kmh

        speed_kmh = min_speed_kmh + 0.5 * (speed_norm + 1.0) * (max_speed_kmh - min_speed_kmh)
        speed_kmh = float(np.clip(speed_kmh, min_speed_kmh, max_speed_kmh))

        steer = float(np.clip(model_action[1], -1.0, 1.0))
        return speed_kmh, steer
    

    def _speed_control(self, target_speed_kmh: float) -> Tuple[float, float]:
        current_speed_kmh = self._kmh(self.ego.get_velocity())
        speed_error = target_speed_kmh - current_speed_kmh

        if speed_error > 1.0:
            throttle = np.clip(0.12 + 0.05 * speed_error, 0.0, 0.75)
            brake = 0.0
        elif speed_error < -1.0:
            throttle = 0.0
            brake = np.clip(0.05 * (-speed_error), 0.0, 0.40)
        else:
            throttle = 0.10 if target_speed_kmh > 1.0 else 0.0
            brake = 0.0

        return float(throttle), float(brake)



    def _destroy_sensor_seg(self):
        if self.sensor_seg is not None:
            try:
                self.sensor_seg.stop()
            except Exception:
                pass

            try:
                self.sensor_seg.destroy()
            except Exception:
                pass

            try:
                if self.sensor_seg in self.actor_handles:
                    self.actor_handles.remove(self.sensor_seg)
            except Exception:
                pass

            self.sensor_seg = None



     # ---------------------------
    # Actors
    # ---------------------------
    def _cleanup_actors(self):
        self._destroy_sensor_seg()

        # if self.camera_sensor is not None:
        #     self.camera_sensor.destroy()
        #     self.camera_sensor = None

        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.lane_sensor is not None:
            self.lane_sensor.destroy()
            self.lane_sensor = None

        # if self.bev_camera is not None:
        #     self.bev_camera.destroy()
        #     self.bev_camera = None

        if self.scenario is not None:
            try:
                self.scenario.remove_all_actors()
            except Exception:
                pass
            self.scenario = None

        self.scenario_tree = None
        self.scenario_criteria = []

        for actor in list(self.actor_handles):
            try:
                actor.destroy()
            except Exception:
                pass

        self.actor_handles.clear()
        self.npc_vehicles.clear()
        self.ego = None
        self.image_seg = None
        self.image_for_CNN = None
        self.neighbor_history = {}
        self.ego_history.clear()



    def close(self):
        self._cleanup_actors()

        if self.world is not None and self.cfg.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        if self.traffic_manager is not None and self.cfg.sync:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass

        cv2.destroyAllWindows()
###################################################################################################################################


    def _build_base_info(self):
        collision = self.collision_sensor is not None and len(self.collision_sensor.history) > 0
        lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()
        speed_kmh = self._kmh(self.ego.get_velocity())

        # nearest_vehicle_dist, nearest_vehicle_rel_speed, nearest_vehicle_ttc = self._get_nearest_vehicle_metrics()
        lead_vehicle_dist, lead_vehicle_rel_speed, lead_vehicle_ttc = self._get_lead_vehicle_metrics()

        pedestrian_dist, pedestrian_active, pedestrian_to_conflict_dist = self._get_pedestrian_metrics()

        if dist_to_goal < 5.0 or progress >= 0.995:
            route_finish = True
        else:
            route_finish = False


        # if abs(lateral_error) > 2.0:
        #     off_route = True
        if abs(lateral_error) > 2.5:
            off_route = True
        else:
            off_route = False

        if speed_kmh < 1.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0


        if self.stuck_steps >= 100:
            stuck = True
        else:
            stuck = False


        progress_delta = progress - self.last_progress
        self.last_progress = progress

        if self.prev_dist_to_goal is None:
            dist_delta = 0.0
        else:
            dist_delta = self.prev_dist_to_goal - dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        lateral_improvement = abs(self.prev_lateral_error) - abs(lateral_error)
        self.prev_lateral_error = lateral_error


        return {"collision": collision,
                "route_finish": route_finish,
                "off_route": off_route,
                "stuck": stuck,
                "progress": progress,
                "progress_delta": progress_delta,
                "dist_to_goal": dist_to_goal,
                "dist_delta": dist_delta,
                "lateral_error": lateral_error,
                "heading_error": heading_error,
                "speed_kmh": speed_kmh,
                "lateral_improvement": lateral_improvement,
                "route_index": self.route_index,

                # "nearest_vehicle_dist": nearest_vehicle_dist,
                # "nearest_vehicle_rel_speed": nearest_vehicle_rel_speed,
                # "nearest_vehicle_ttc": nearest_vehicle_ttc,

                "lead_vehicle_dist": lead_vehicle_dist,
                "lead_vehicle_rel_speed": lead_vehicle_rel_speed,
                "lead_vehicle_ttc": lead_vehicle_ttc,

                "pedestrian_dist": pedestrian_dist,
                "pedestrian_active": pedestrian_active,
                "pedestrian_to_conflict_dist": pedestrian_to_conflict_dist,
                }
    
    

    def _update_spectator(self):
        if not self.cfg.spectator_follow or self.ego is None or self.world is None:
            return

        spectator = self.world.get_spectator()
        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        yaw = ego_tf.rotation.yaw if self.cfg.spectator_rotate_with_ego else 0.0

        spec_tf = carla.Transform(carla.Location(x=ego_loc.x, y=ego_loc.y, z=self.cfg.spectator_height_m),
                                  carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
                                  )
        
        spectator.set_transform(spec_tf)
    


    def reset(self, *, seed=None, options=None):
        self.skip_episode = False
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if self.client is None:
            self._connect()

        if not self.route_configs:
            self._load_route_configs()

        self.step_count = 0
        self.route_index = 0
        self.prev_steer = 0.0
        self.last_progress = 0.0
        self.prev_dist_to_goal = None
        self.prev_lateral_error = 0.0
        self.stuck_steps = 0
        self.image_seg = None
        self.image_for_CNN = None
        self._obs_debug_counter = 0

        self._choose_route()
        target_town = self.route_config.town

        self._cleanup_actors()

        if self.world is None:
            need_load_world = True
        else:
            need_load_world = False

        if not need_load_world:
            try:
                current_map = self.world.get_map().name.split("/")[-1]
                need_load_world = current_map != target_town
            except Exception:
                need_load_world = True

        if need_load_world:
            self._load_world_for_route()
        else:
            for _ in range(5):
                self.world.tick()

        for _ in range(10):
            self.world.tick()

        self._prepare_route()
        self._spawn_ego()

        # Agora que o ego existe, começa o histórico da trajetória real deste episódio.
        self._reset_policy_route_history()

        # route_ok = self._prepare_route()

        # if not route_ok:
            # return self.reset(seed=seed, options=options)

        # self._spawn_ego()

        if self.task is not None:
            self.task.on_reset(self)

        if getattr(self, "skip_episode", False):
            print("[RESET] invalid route, trying another route")
            self._cleanup_actors()
            return self.reset(seed=seed, options=options)

        if self.cfg.num_npc_vehicles > 0:
            self._spawn_background_traffic()


        self._destroy_sensor_seg()

        self.blueprint_library = self.world.get_blueprint_library()
        self.sem_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.sem_cam.set_attribute("image_size_x", str(self.im_width))
        self.sem_cam.set_attribute("image_size_y", str(self.im_height))
        self.sem_cam.set_attribute("fov", str(self.cfg.front_camera_fov))

        cam_tf = carla.Transform(carla.Location(x=self.CAMERA_POS_X, y=0.0, z=self.CAMERA_POS_Z),
                                carla.Rotation(pitch=-90.0),
                                )

        self.sensor_seg = self.world.spawn_actor(self.sem_cam, cam_tf, attach_to=self.ego)
        self.actor_handles.append(self.sensor_seg)
        self.sensor_seg.listen(lambda data: self.process_img(data))

        self._warmup_ticks()
        self._update_spectator()

        if self.collision_sensor:
            self.collision_sensor.clear()
        if self.lane_sensor:
            self.lane_sensor.clear()

        self.ego_history.clear()
        self.neighbor_history = {}
        self.last_location = self.ego.get_location()

        lateral_error, _, progress, dist_to_goal = self._compute_route_errors()
        self.last_progress = progress
        self.prev_dist_to_goal = dist_to_goal
        self.prev_lateral_error = lateral_error

        if self.draw_debug_routes:
            self._draw_xml_route(life_time=self.debug_draw_life_time)
            self._draw_full_policy_route(life_time=self.debug_policy_life_time)

        obs = self._get_structured_obs()

        if self.image_seg is not None:
            self.image_for_CNN = self.apply_cnn(self.image_seg[:, :])
        else:
            self.image_for_CNN = np.zeros(280, dtype=np.float32)

        return obs, {"vision": self.image_for_CNN}
    


    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        
        prev_loc = self.ego.get_location()
        
        target_speed_kmh, rl_steer = self.action_adapter(action)
        self.target_speed = float(target_speed_kmh)

        throttle_cmd, brake_cmd = self._speed_control(self.target_speed)

        old_prev_steer = self.prev_steer
        # max_delta = 0.08
        # steer_cmd = float(np.clip(rl_steer, old_prev_steer - max_delta, old_prev_steer + max_delta))
        # steer_cmd = float(np.clip(steer_cmd, -0.45, 0.45))
        max_delta = 0.04
        steer_cmd = float(np.clip(rl_steer, old_prev_steer - max_delta, old_prev_steer + max_delta))
        steer_cmd = float(np.clip(steer_cmd, -0.25, 0.25))
        self.prev_steer = steer_cmd

        control = carla.VehicleControl(throttle=float(throttle_cmd),
                                       brake=float(brake_cmd),
                                       steer=steer_cmd,
                                       hand_brake=False,
                                       reverse=False,
                                       manual_gear_shift=False,
                                       )

        self.ego.apply_control(control)
        self.world.tick()
        self.step_count += 1


        curr_loc = self.ego.get_location()

        if self.draw_debug_routes:
            curr_debug_loc = carla.Location(x=curr_loc.x, y=curr_loc.y, z=curr_loc.z)
            self.policy_route_history.append(curr_debug_loc)

            # Redesenha a trajetória COMPLETA do episódio atual em verde.
            # Como life_time é curto, as linhas do episódio anterior desaparecem
            # logo após o reset, evitando acumular trajetórias antigas na tela.
            redraw_every = max(1, int(getattr(self, "debug_policy_redraw_every_steps", 1)))
            if self.step_count % redraw_every == 0:
                self._draw_full_policy_route(life_time=self.debug_policy_life_time)


        self._update_spectator()

        if self.task is not None:
            self.task.after_tick(self)

        obs = self._get_structured_obs()

        if self.image_seg is not None:
            self.image_for_CNN = self.apply_cnn(self.image_seg[:, :])
            # print('======= ANA =======', self.image_for_CNN)
        else:
            self.image_for_CNN = np.zeros(280, dtype=np.float32)

        base_info = self._build_base_info()
        base_info.update({"vision": self.image_for_CNN,
                          "target_speed_kmh": self.target_speed,
                          "control_steer": steer_cmd,
                          "rl_steer": float(rl_steer),
                          "control_throttle": float(throttle_cmd),
                          "control_brake": float(brake_cmd),
                          "steer_delta": abs(steer_cmd - old_prev_steer),
                          "stuck_steps": self.stuck_steps,
                          }
                        )

        if self.task is None:
            reward, done, task_info = 0.0, False, {"finish": False, "done_reason": None}
        else:
            reward, done, task_info = self.task.compute_reward_done(self, base_info)

        if done and self.draw_debug_routes:
            # Reforça a linha completa no fim, mas ainda com life_time curto.
            # Assim ela aparece no momento final, porém não fica contaminando
            # o próximo episódio.
            self._draw_full_policy_route(life_time=self.debug_policy_life_time)

        info = dict(base_info)
        info.update(task_info)
        info["reward"] = reward

        return obs, reward, done, info
    
# ============================== methods for tasks =============================================== #

    # def _get_nearest_vehicle_metrics(self):
    #    """
    #    Returns:
    #    nearest_dist: Euclidean distance to the nearest vehicle
    #    rel_speed: closing speed (positive = approaching vehicle)
    #    ttc: approximate time-to-collision
    #    """
    #    if self.ego is None:
    #        return float("inf"), 0.0, float("inf")

    #    ego_tf = self.ego.get_transform()
    #    ego_loc = ego_tf.location
    #    ego_vel = self.ego.get_velocity()

    #    ego_speed = self._kmh(ego_vel) / 3.6  # m/s

    #    nearest_dist = float("inf")
    #    nearest_rel_speed = 0.0

    #    for actor in self.npc_vehicles:
    #        if actor is None or not actor.is_alive:
    #            continue

    #        other_loc = actor.get_location()
    #        dist = self._distance(ego_loc, other_loc)

    #        if dist < nearest_dist:
    #            other_vel = actor.get_velocity()
    #            other_speed = self._kmh(other_vel) / 3.6  # m/s

    #            # Simple approach: if the ego is faster, there is closure.
    #            rel_speed = max(ego_speed - other_speed, 0.0)

    #            nearest_dist = dist
    #            nearest_rel_speed = rel_speed

    #    if nearest_dist == float("inf"):
    #        return float("inf"), 0.0, float("inf")

    #    if nearest_rel_speed > 1e-3:
    #        ttc = nearest_dist / nearest_rel_speed
    #    else:
    #        ttc = float("inf")

    #    return nearest_dist, nearest_rel_speed, ttc
    
    

    def _get_lead_vehicle_metrics(self):
        """
        Considera apenas veículos relevantes à frente do ego.
        Retorna:
            lead_dist
            lead_rel_speed
            lead_ttc
        """
        if self.ego is None:
            return float("inf"), 0.0, float("inf")

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_vel = self.ego.get_velocity()

        ego_speed = self._kmh(ego_vel) / 3.6  # m/s

        c = math.cos(ego_yaw)
        s = math.sin(ego_yaw)

        best_longitudinal = float("inf")
        lead_rel_speed = 0.0

        for actor in self.npc_vehicles:
            if actor is None or not actor.is_alive:
                continue

            other_tf = actor.get_transform()
            other_loc = other_tf.location
            other_yaw = math.radians(other_tf.rotation.yaw)

            dx = float(other_loc.x - ego_loc.x)
            dy = float(other_loc.y - ego_loc.y)

            # coordenadas no frame do ego
            x_local = c * dx + s * dy
            y_local = -s * dx + c * dy

            # só veículos à frente
            if x_local <= 0.0:
                continue

            # só veículos no corredor da frente (aprox mesma faixa / faixa vizinha próxima)
            if abs(y_local) > 3.0:
                continue

            # mesma direção aproximada
            yaw_diff = (other_yaw - ego_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_diff) > math.radians(35.0):
                continue

            if x_local < best_longitudinal:
                other_vel = actor.get_velocity()
                other_speed = self._kmh(other_vel) / 3.6
                rel_speed = max(ego_speed - other_speed, 0.0)

                best_longitudinal = x_local
                lead_rel_speed = rel_speed

        if best_longitudinal == float("inf"):
            return float("inf"), 0.0, float("inf")

        if lead_rel_speed > 1e-3:
            lead_ttc = best_longitudinal / lead_rel_speed
        else:
            lead_ttc = float("inf")

        return best_longitudinal, lead_rel_speed, lead_ttc
    

    def _get_pedestrian_metrics(self):
        if self.ego is None or self.scenario is None or not hasattr(self.scenario, "other_actors"):
            return float("inf"), False, float("inf")

        ego_loc = self.ego.get_location()
        min_dist = float("inf")
        ped_active = False
        min_conflict_dist = float("inf")

        conflict_loc = None
        if hasattr(self.scenario, "_collision_wp") and self.scenario._collision_wp is not None:
            conflict_loc = self.scenario._collision_wp.transform.location

        for actor in self.scenario.other_actors:
            if actor is None or not actor.is_alive:
                continue

            loc = actor.get_location()
            if loc is None:
                continue

            d = self._distance(ego_loc, loc)
            if d < min_dist:
                min_dist = d

            vel = actor.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            if speed > 0.2:
                ped_active = True

            if conflict_loc is not None:
                cdist = self._distance(loc, conflict_loc)
                if cdist < min_conflict_dist:
                    min_conflict_dist = cdist

        return min_dist, ped_active, min_conflict_dist