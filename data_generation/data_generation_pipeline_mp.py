from datetime import datetime as dt
from timeit import default_timer as timer
import os
from RobotsGen.robot_gen_random import generate_robots, random_unique_generate_robots_2d_optim
import json
from data_generation_pipeline_mp_helper import workspace_gen_mp
from itertools import repeat


"""
    file to generate 1 million dataset (settings as is)
    settings can be adjusted at will for different data requirements
"""

# Settings:
storeData = True
compressed = True
auto_mp = True
numb_p = 16  # if auto_mp == True numb_cores will be set to maximum number of available processors

# Robot settings:
robot_type = '2d'  # '3d' # 'geometric_primitive' # 'geometric_primitive_revolute_only'
numb_robots = 1000000  # number of robot assemblies to generate
num_config = 0
nModule_min = 4
nModule_max = 12
nJoint_min = 1
nJoint_max = 6
link_sizes = [.2, .3]
rob_vis = False
rob_save = False

# Workspace generation settings:
min_val = -3  # minimum value of grid coordinate
max_val = 3  # maximum value of grid coordinate
d = 101  # resolution
numb_samples = 20000
r_factor = 0  # d * d * d * r random samples will be generated
ignore_self_collisions = False
with_manip_index = True
saving = False  # save to file
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "dev_data")
samplingType = '2d_optim'  # 'random','deliberate', '2d_optim'
info = True  # display information about process

# Implementation:
# Information about data generation process
start_timer = None
if info:
    start_timer = timer()

numb_voxels = d * d * d
generated_robots = []

# File structure preparation:
root_folder_name = f"Data_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}_r:{numb_robots}_s:{numb_voxels}"
data_path = os.path.join(base_path, root_folder_name)
if storeData and not os.path.exists(data_path):
    os.mkdir(data_path)

assemblies = []

# generate robots:
if robot_type == '2d':
    assemblies, modulesDB = random_unique_generate_robots_2d_optim(robot_module_db=robot_type, num_rob=numb_robots,
                                                                   num_config=num_config, nModule_min=nModule_min,
                                                                   nModule_max=nModule_max, nJoint_min=nJoint_min,
                                                                   nJoint_max=nJoint_max, link_sizes=link_sizes,
                                                                   visual=rob_vis, save=rob_save)
elif robot_type == '3d':
    # TODO: implementation for 3d robot generation
    assemblies, modulesDB = generate_robots(robot_module_db=robot_type, num_rob=numb_robots, num_config=num_config,
                                            nModule_min=nModule_min, nModule_max=nModule_max, nJoint_min=nJoint_min,
                                            nJoint_max=nJoint_max, link_sizes=link_sizes, visual=rob_vis, save=rob_save)
else:
    raise ValueError("Invalid robot type!")

print("Assemblies generated")
# generate workspace data
assemblies_safe = [a.to_json_data() for a in assemblies]
modulesDB_safe = modulesDB.to_json_string()
args_iter = list(zip(assemblies_safe,
                     repeat(data_path),
                     repeat(min_val),
                     repeat(max_val),
                     repeat(d),
                     repeat(numb_samples),
                     repeat(r_factor),
                     repeat(ignore_self_collisions),
                     repeat(with_manip_index),
                     repeat(modulesDB_safe),
                     repeat(storeData),
                     repeat(compressed),
                     repeat(samplingType)
                     ))

generated_robots = workspace_gen_mp(auto_mp, numb_p, args_iter)

if storeData:
    # Save settings within folder
    settings_file_path = os.path.join(data_path, "settings.txt")
    with open(settings_file_path, "w") as settings_file:
        parameter_string = f"rob: {numb_robots}\nmax_j: {nJoint_max}\nres: {d}"
        settings_file.write(parameter_string)

    robots_file_path = os.path.join(data_path, "robot_names.json")
    with open(robots_file_path, "w") as robots_file:
        json.dump(generated_robots, robots_file)

    modulesDB_path = os.path.join(data_path, "modules.json")
    modulesDB.to_json_file(modulesDB_path)

# Information about data generation process
if info:
    end_timer = timer()
    print(f"Overall execution time: {end_timer - start_timer}s")
    print(f"for {numb_robots} {robot_type} robots")
