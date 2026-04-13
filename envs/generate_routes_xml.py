import sys
sys.path.append("/home/ana/Documents/Architecture_Transformers_SR/scenario_runner")
sys.path.append("/home/ana/CARLA_0.9.13/PythonAPI/carla")
import carla
import random
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from agents.navigation.global_route_planner import GlobalRoutePlanner
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


client = carla.Client('localhost', port=2000)
client.set_timeout(seconds=10.0)
client.load_world('/Game/Carla/Maps/Town01_Opt')

world = client.get_world()
map = world.get_map()

spawn_points = map.get_spawn_points()
sampling_resolution = 0.1

valid_spawns = []
for sp in spawn_points:
    waypoint = map.get_waypoint(
        sp.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    valid_spawns.append(waypoint)

routes_data = []
for idx, current_wp in enumerate(valid_spawns):
    accumulated = 0.0
    final_route = [current_wp]

    while accumulated < 100.0:
        next_wps = current_wp.next(sampling_resolution)
        if not next_wps:
            break

        next_wp = next_wps[0]
        if next_wp.road_id != current_wp.road_id:
            break

        d = current_wp.transform.location.distance(next_wp.transform.location)
        accumulated += d
        current_wp = next_wp
        final_route.append(current_wp)

    current_wp = final_route[0]
    while accumulated < 100.0:
        previous_wps = current_wp.previous(sampling_resolution)
        if not previous_wps:
            break

        previous_wp = previous_wps[0]
        if previous_wp.road_id != current_wp.road_id:
            break

        final_route.insert(0, previous_wp)
        d = current_wp.transform.location.distance(previous_wp.transform.location)
        accumulated += d
        current_wp = previous_wp

    distances = [0.0]
    for i in range(1, len(final_route)):
        d = final_route[i].transform.location.distance(final_route[i - 1].transform.location)
        distances.append(distances[-1] + d)

    mu = np.mean(distances)
    sigma = np.std(distances)
    np.random.seed(42)
    sampled_dist = np.random.normal(loc=mu - 2 * sigma, scale=sigma, size=1)[0]

    trigger_wp = final_route[0]
    for i, d in enumerate(distances):
        if d >= sampled_dist:
            trigger_wp = final_route[i]
            break

    routes_data.append({
        'start': final_route[0].transform.location,
        'end': final_route[-1].transform.location,
        'trigger': trigger_wp.transform,
        'route': final_route
    })

#################################################### Gerar os arquivos xml ################################################################
def prettify_xml(elem):
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent='\t')

town_name = world.get_map().name.split('/')[-1]

root = ET.Element('routes')

for idx, route in enumerate(routes_data):
    route_elem = ET.SubElement(root, 'route', {'id': str(idx),
                                                'town': town_name
                                                }
                                )

    for wp_idx, wp in enumerate(route['route']):
        if wp_idx % 10 != 0:
            continue

        transform = wp.transform
        loc = transform.location
        rot = transform.rotation

        ET.SubElement(route_elem, 'waypoint', {
            'x': str(loc.x),
            'y': str(loc.y),
            'z': str(loc.z),
            'pitch': str(rot.pitch),
            'roll': str(rot.roll),
            'yaw': str(rot.yaw)
        })

xml = prettify_xml(root)

output_file = f'routes_{town_name}.xml'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(xml)

print(f'Arquivo XML salvo em: {output_file}')