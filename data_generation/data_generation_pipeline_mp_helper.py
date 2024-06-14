from datetime import datetime as dt
import os
from voxel_workspace_data_gen_fk import fk_voxel_workspace, fk_voxel_workspace_deliberate_configs,\
    fk_voxel_workspace_2d_optim
from RobotsGen.robot_gen_random import generate_robots
import uuid
import numpy as np
import pickle
from input_encoding_helper import assembly_to_binary_encoding
import multiprocessing as mp
from itertools import repeat
from timor.Module import ModulesDB, ModuleAssembly
import json
import tqdm


"""
    file includes helper function for parallelized workspace generation
    --> most interesting function is workspace_mp_wrapper(), it wraps different FK workspace generation functions
    --> used for 1 million dataset fk_voxel_workspace_2d_optim()
"""


def workspace_mp_wrapper(assembly_safe: dict, data_path: os.PathLike, min_val: int, max_val: int, resolution: int,
                         numb_samples: int, r_factor: int, ignore_self_collisions: bool, with_manip_index: bool,
                         modulesDB_safe: str, storeData: bool, compressed: bool, samplingType: str = 'random'):
    modulesDB = ModulesDB.from_json_string(modulesDB_safe, package_dir=None)
    assembly = ModuleAssembly.from_json_data(assembly_safe, modulesDB)
    robot = assembly.to_pin_robot()
    robot._name = str(uuid.uuid4())
    if samplingType == 'random':
        ws = fk_voxel_workspace(
            robot=robot,
            min_val=min_val,
            max_val=max_val,
            resolution=resolution,
            numb_samples=numb_samples,
            r_factor=r_factor,
            ignore_self_collisions=ignore_self_collisions,
            with_manip_index=with_manip_index,
            visualisation=False,
            displayAllSamples=False,
            saving=False,
            file_name="",
            compressed=compressed,
            info=False
        )
    elif samplingType == 'deliberate':
        ws = fk_voxel_workspace_deliberate_configs(
            robot=robot,
            min_val=min_val,
            max_val=max_val,
            resolution=resolution,
            r_factor=r_factor,
            ignore_self_collisions=ignore_self_collisions,
            with_manip_index=with_manip_index,
            visualisation=False,
            saving=False,
            file_name="",
            info=False
        )
    elif samplingType == '2d_optim':
        ws = fk_voxel_workspace_2d_optim(
            robot=robot,
            min_val=min_val,
            max_val=max_val,
            resolution=resolution,
            numb_samples=numb_samples,
            r_factor=0,
            ignore_self_collisions=ignore_self_collisions,
            with_manip_index=with_manip_index,
            visualisation=False,
            displayAllSamples=False,
            saving=False,
            file_name="",
            compressed=compressed,
            info=False
        )
    else:
        raise ValueError(f"Sampling type '{samplingType}' not implemented! "
                         f"Only 'random' and 'deliberate' are available")
    # saving:
    if storeData:
        fn_a = f"Assembly_r:{robot._name}.pickle"
        fn_b = f"Binary_Assembly_r:{robot._name}.npy"
        if compressed:
            fn_w = f"Grid_Workspace_r:{robot._name}.npz"
        else:
            fn_w = f"Grid_Workspace_r:{robot._name}.npy"
        with open(os.path.join(data_path, fn_a), 'wb') as f:
            pickle.dump(assembly.to_json_data(), f)
        bn_assembly = assembly_to_binary_encoding(assembly=assembly, db=modulesDB)
        np.save(os.path.join(data_path, fn_b), bn_assembly)
        np.savez_compressed(os.path.join(data_path, fn_w), ws)
    return str(robot._name)


def workspace_gen_mp(auto_mp: bool, numb_p: int, args_iter: list):
    # multiprocessing:
    print('starting MP')
    if auto_mp:
        numb_p = mp.cpu_count()
    with mp.Pool(numb_p) as pool:
        res = pool.starmap(workspace_mp_wrapper, tqdm.tqdm(args_iter, total=len(args_iter[0])))
    print('MP done')
    return res


if __name__ == '__main__':
    min_val = -2  # minimum value of grid coordinate
    max_val = 2  # maximum value of grid coordinate
    d = 30  # resolution
    r_factor = 5  # d * d * d * r random samples will be generated
    ignore_self_collisions = True
    with_manip_index = False
    visualisation = False  # show workspace visualisation
    saving = False  # save to file
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "dev_data")
    info = True  # display information about process
    storeData = True
    compressed = False
    auto_mp = True
    numb_p = 16  # if auto_mp == True numb_cores will be set to maximum number of available processors
    samplingType = 'normal'

    numb_robots = 32  # number of robot assemblies to generate
    num_config = 0
    nModule_min = 3
    nModule_max = 12
    nJoint_max = 6
    link_sizes = [.2, .5]
    rob_vis = False
    rob_save = False

    numb_voxels = d * d * d

    root_folder_name = f"Data_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}_r:{numb_robots}_s:{numb_voxels}"
    data_path = os.path.join(base_path, root_folder_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    assemblies, modulesDB = generate_robots(numb_robots, num_config, nModule_min, nModule_max,
                                            nJoint_max, link_sizes, visual=rob_vis, save=rob_save)

    assemblies_safe = [a.to_json_data() for a in assemblies]
    modulesDB_safe = modulesDB.to_json_string()
    print(assemblies_safe)

    args_iter = list(zip(assemblies_safe,
                         repeat(data_path),
                         repeat(min_val),
                         repeat(max_val),
                         repeat(d),
                         repeat(r_factor),
                         repeat(ignore_self_collisions),
                         repeat(with_manip_index),
                         repeat(modulesDB_safe),
                         repeat(storeData),
                         repeat(compressed),
                         repeat(samplingType)
                         ))
    generated_robots = workspace_gen_mp(auto_mp, numb_p, args_iter)
    print(generated_robots)

    # Save settings within folder
    settings_file_path = os.path.join(data_path, "settings.txt")
    with open(settings_file_path, "w") as settings_file:
        parameter_string = f"rob: {numb_robots}\nmax_j: {nJoint_max}\nres: {d}"
        settings_file.write(parameter_string)

    robots_file_path = os.path.join(data_path, "robot_names.json")
    with open(robots_file_path, "w") as robots_file:
        json.dump(generated_robots, robots_file)
