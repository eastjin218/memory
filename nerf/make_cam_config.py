import json

cam_config = []

with open('/home/carla_client/carla/PythonAPI/aas/utils/dataset/configs/nerf/ori_config_val.json', 'r') as f:
    json_file = json.load(f)

for i in json_file:
    if i['y'] < 0 and i['x'] >0 :
        cam_data = {}
        cam_data['angle'] = 360 - i['angle']
        cam_data['distance'] = 10
        cam_data['pitch'] = 90 - i['roll']
        cam_config.append(cam_data)
    elif i['x'] > 0 and i['y'] > 0:
        cam_data = {}
        cam_data['angle'] = -i['angle']
        cam_data['distance'] = 10
        cam_data['pitch'] = 90 -i['roll']
        cam_config.append(cam_data)
    else:
        cam_data = {}
        cam_data['angle'] = i['angle']
        cam_data['distance'] = 10
        cam_data['pitch'] = 90 - i['roll']
        cam_config.append(cam_data)

print(len(cam_config))
with open('/home/carla_client/carla/PythonAPI/aas/utils/dataset/configs/nerf/cam_config_valid.json', 'w') as f:
    json.dump(cam_config, f, indent=2)
