#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pedestrians crossing through the middle of the lane.
"""

from __future__ import print_function

import random
import py_trees
import carla
import math

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                    Idle,
                                                                    ActorTransformSetter,
                                                                    KeepVelocity
                                                                    )

from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                                DriveDistance,
                                                                                )

from srunner.scenarios.basic_scenario import BasicScenario


class PedestrianCrossing(BasicScenario):
    """
    Group of pedestrians crossing the street. 
    Adapted version for use with Gym + CARLA + ScenarioRunner.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True, timeout=60):
        
        self.world = world
        self.other_actors = []
        self.route_mode = False

        self._wmap = CarlaDataProvider.get_map()  # Map(name=Carla/Maps/town)
        self._trigger_location = config.trigger_points[0].location     
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)            
        self._rng = random.Random()

        self._ego_end_distance = 40.0
        self.timeout = timeout

        self._walker_data = []
        route = config.route 
        points = []
        for point, _ in route:
            points.append(point)
        
        dist_route = 0.0
        for i in range(1, len(points)):
            p1 = points[i-1].location
            p2 = points[i].location
            dist_route += p1.distance(p2) 

        delta_dist = self._rng.uniform(0.3 * dist_route, 0.4 * dist_route)
        random_pedestrians = random.randint(1, 3)

        for i in range(random_pedestrians):
            walker = {}

            spacing = self._rng.uniform(0.5, 1.5)  # random space between pedestrians
            offset_x = 0.0
            for _ in range(i):
                offset_x += spacing  # sum of accumulated spacings

            
            walker["x"] = delta_dist + offset_x
            walker["y"] = self._rng.uniform(0.0, 1.5) # slight variation in the Y (lateral)
            walker["z"] = 1.2
            walker["yaw"] = 270.0

            self._walker_data.append(walker)


        max_ego_speed = 30.0 / 3.6   # 8.33 m/s
        estimated_ego_speed = 0.6 * max_ego_speed

        for walker_data in self._walker_data:
            # walker_data["idle_time"] = self._rng.uniform(0.0, 1.0)  # time the pedestrian will remain still
            walker_data["speed"] = self._rng.uniform(1.3, 2.0)      # pedestrian movement speed
            # reaction_time = self._rng.uniform(3.5, 5.0)       # Reaction time between 2 and 5 seconds
            reaction_time = self._rng.uniform(1.0, 2.0)
            walker_data["trigger_dist"] = estimated_ego_speed * reaction_time # Calculates pedestrian activation distance based on v = 0.6 * vmax


        super().__init__("PedestrianCrossing", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable,)        

    def _warm_up_carla_data_provider(self, ticks=2):
        for _ in range(ticks):
            if CarlaDataProvider.is_sync_mode():
                self.world.tick()
            else:
                self.world.wait_for_tick()

            CarlaDataProvider.on_carla_tick()


    def _get_walker_transform(self, wp, displacement):
        disp_x = displacement["x"]
        disp_y = displacement["y"]
        disp_z = displacement["z"]
        disp_yaw = displacement["yaw"]

        # Displace it to the crosswalk. Move forwards towards the crosswalk 
        start_vec = wp.transform.get_forward_vector()
        start_right_vec = wp.transform.get_right_vector()

        spawn_location = wp.transform.location + carla.Location(disp_x * start_vec.x + disp_y * start_right_vec.x,
                                                                disp_x * start_vec.y + disp_y * start_right_vec.y,
                                                                disp_x * start_vec.z + disp_y * start_right_vec.z + disp_z,
                                                                )

        spawn_rotation = wp.transform.rotation
        spawn_rotation.yaw += disp_yaw

        return carla.Transform(spawn_location, spawn_rotation)

    
    
    def _adjust_to_nearest_sidewalk(self, spawn_transform, sid_wps):
        '''
        Sets the pedestrian spawn position to the nearest sidewalk waypoint.

        Parameters:
        - spawn_transform: desired carla.Transform for the pedestrian (can be outside the sidewalk)
        - sid_wps: list of carla.Waypoints that are on Sidewalk, Shoulder or Parking

        Returns:
        - carla.Transform set to sidewalk
        '''

        if not sid_wps:
            raise ValueError('The sidewalk waypoint list (sid_wps) is empty.')

        closest_wp = min(sid_wps, key=lambda wp: wp.transform.location.distance(spawn_transform.location))

        location = closest_wp.transform.location + carla.Location(z=1.2)
        road_yaw = closest_wp.transform.rotation.yaw
        adjusted_yaw = (road_yaw - 90.0) % 360.0
        rotation = carla.Rotation(pitch=0.0, yaw=adjusted_yaw, roll=0.0)

        return carla.Transform(location, rotation)
    
    
    def _find_free_sidewalk_transform(self, desired_transform, sid_wps, occupied_locations, min_dist=1.2):
        '''
        Finds a free sidewalk waypoint near the desired location.
        Parameters:
        - desired_transform: desired pedestrian waypoint (carla.Transform)
        - sid_wps: list of occupied sidewalk waypoints (carla.Location)
        - min_dist: minimum distance between pedestrians
        Returns:
        - free sidewalk waypoint (carla.Transform), or None if not found
        '''
        
        if not sid_wps:
            return None

        sorted_wps = sorted(sid_wps, key=lambda wp: wp.transform.location.distance(desired_transform.location))

        for wp in sorted_wps:
            candidate_loc = wp.transform.location + carla.Location(z=1.2)

            too_close = False
            for occ_loc in occupied_locations:
                if candidate_loc.distance(occ_loc) < min_dist:
                    too_close = True
                    break

            if too_close:
                continue

            road_yaw = wp.transform.rotation.yaw
            adjusted_yaw = (road_yaw - 90.0) % 360.0
            rotation = carla.Rotation(pitch=0.0, yaw=adjusted_yaw, roll=0.0)

            return carla.Transform(candidate_loc, rotation)

        return None
    

    def _find_sidewalk_waypoint(self, base_wp, max_hops=50):
        valid_lane_types = (carla.LaneType.Sidewalk,
                            # carla.LaneType.Shoulder,
                            # carla.LaneType.Parking,
                            )

        if base_wp is None:
            return None

        # print("base lane_type:", base_wp.lane_type)

        if base_wp.lane_type in valid_lane_types:
            return base_wp

        cur = base_wp
        for i in range(max_hops):
            nxt = cur.get_right_lane()
            # print(f"[RIGHT {i}] nxt =", None if nxt is None else nxt.lane_type)
            if nxt is None:
                break
            if nxt.lane_type in valid_lane_types:
                # print("Find it valid on the right")
                return nxt
            cur = nxt

        cur = base_wp
        for i in range(max_hops):
            nxt = cur.get_left_lane()
            # print(f"[LEFT {i}] nxt =", None if nxt is None else nxt.lane_type)
            if nxt is None:
                break
            if nxt.lane_type in valid_lane_types:
                # print("Achou válido à esquerda")
                return nxt
            cur = nxt

        # print("No valid lane found.")
        return None
    

    def _try_spawn_walker_nearby(self, base_transform, sid_wps, max_tries=12):
        candidates = sorted(sid_wps, key=lambda wp: wp.transform.location.distance(base_transform.location))

        tries = 0
        for wp in candidates:
            loc = wp.transform.location + carla.Location(z=1.2)

            tf = carla.Transform(loc, base_transform.rotation)
            walker = CarlaDataProvider.request_new_actor('walker.*', tf)
            tries += 1

            if walker is not None:
                return walker, tf

            if tries >= max_tries:
                break

        return None, None
    
    

    def _initialize_actors(self, config):
        self._collision_wp = self._reference_waypoint

        if self._collision_wp is None:
            raise ValueError("Pass an XML route")

        start_wp = self._find_sidewalk_waypoint(self._collision_wp, max_hops=50)

        if start_wp is None:
            self.other_actors = []
            self._walker_data = []
            return

        init_wp = start_wp
        roadid = init_wp.road_id
        sid_wps = [init_wp]

        for _ in range(200):
            next_wps = init_wp.next(0.5)
            if not next_wps:
                break

            next_wp = next_wps[0]
            if next_wp.road_id != roadid:
                break

            init_wp = next_wp
            sid_wps.append(init_wp)

        valid_walker_data = []
        spawned_locations = []

        for walker_data in self._walker_data:
            spawn_transform = self._get_walker_transform(start_wp, walker_data)
            spawn_transform = self._find_free_sidewalk_transform(spawn_transform,
                                                                sid_wps,
                                                                spawned_locations,
                                                                min_dist=1.2
                                                                )

            if spawn_transform is None:
                print("[DEBUG spawn] no free waypoint found, pedestrian jumping")
                continue


            spawn_transform = self._adjust_to_nearest_sidewalk(spawn_transform, sid_wps)

            road_right = self._collision_wp.transform.get_right_vector()
            spawn_loc = spawn_transform.location
            center_loc = self._collision_wp.transform.location

            to_center_x = center_loc.x - spawn_loc.x
            to_center_y = center_loc.y - spawn_loc.y

            dot = to_center_x * road_right.x + to_center_y * road_right.y

            if dot >= 0.0:
                cross_dir_x = road_right.x
                cross_dir_y = road_right.y
            else:
                cross_dir_x = -road_right.x
                cross_dir_y = -road_right.y

            yaw_cross = math.degrees(math.atan2(cross_dir_y, cross_dir_x))

            spawn_transform.rotation = carla.Rotation(pitch=0.0, yaw=yaw_cross, roll=0.0)

            too_close = False
            for loc in spawned_locations:
                if spawn_transform.location.distance(loc) < 1.2:  # minimum distance between pedestrians
                    too_close = True
                    break   
            
            if too_close:
                print("[DEBUG spawn] position very close to another pedestrian, jumping")
                continue
            


            collision_dist = spawn_transform.location.distance(self._collision_wp.transform.location)
            move_dist = 2.3 * collision_dist
            
            walker_data["transform"] = spawn_transform
            walker_data["distance"] = move_dist
            walker_data["duration"] = move_dist / walker_data["speed"]


            walker, final_transform = self._try_spawn_walker_nearby(spawn_transform, sid_wps, max_tries=12)

            if walker is None:
                print("[DEBUG spawn] failed on all retries, skipping this pedestrian")
                continue

            spawn_transform = final_transform
            walker_data["transform"] = spawn_transform


            self.other_actors.append(walker)
            valid_walker_data.append(walker_data)
            spawned_locations.append(spawn_transform.location)


        self._walker_data = valid_walker_data
    
        if self.other_actors:
            self._warm_up_carla_data_provider(ticks=2)


    
    def _create_behavior(self):
        """
        Define the full behavior tree of the scenario:
        Each pedestrian is triggered individually when the ego vehicle approaches,
        then waits (Idle) for a short moment and starts crossing.
        """
        sequence = py_trees.composites.Sequence(name="PedestrianCrossing")

        if not self.other_actors or not self._walker_data:
            sequence.add_child(DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))
            return sequence

        for walker_actor, walker_data in zip(self.other_actors, self._walker_data):
            sequence.add_child(ActorTransformSetter(walker_actor, walker_data["transform"], True))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                                                    name="WalkerMovement",
                                                    )

        for i, (walker_actor, walker_data) in enumerate(zip(self.other_actors, self._walker_data)):
            trigger = InTriggerDistanceToLocation(self.ego_vehicles[0],
                                                    # walker_data["transform"].location,
                                                    self._collision_wp.transform.location,
                                                    walker_data["trigger_dist"],
                                                    )

            walker_sequence = py_trees.composites.Sequence(name=f"WalkerCrossing_{i}")
            walker_sequence.add_child(trigger)
            # walker_sequence.add_child(Idle(walker_data["idle_time"]))
            
            walker_sequence.add_child(KeepVelocity(walker_actor,
                                                    walker_data["speed"],
                                                    # duration=walker_data["duration"],
                                                    distance=walker_data["distance"],
                                                    name=f"WalkerKeepVelocity_{i}",
                                                    )
                                        )
            
            walker_sequence.add_child(ActorDestroy(walker_actor, name=f"DestroyWalker_{i}"))            

            main_behavior.add_child(walker_sequence)


        main_behavior.add_child(DriveDistance(self.ego_vehicles[0], 
                                              self._ego_end_distance, 
                                              name="EndCondition"
                                              )
                                )

        sequence.add_child(main_behavior)
        return sequence
    

    def _setup_scenario_trigger(self, config):                                                       
        return None

    

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """

        if self.route_mode:
            return []
        
        return [CollisionTest(self.ego_vehicles[0])]
    

    def __del__(self):
        """
        Remove all actors upon deletion
        """   
        self.remove_all_actors()