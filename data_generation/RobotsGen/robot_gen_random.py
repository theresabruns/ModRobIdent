'''
    Randomly generate robots that can move either in 2D or 3D according to the provided database
'''
import sys
import time
import timor
import os
import numpy as np
from pathlib import Path
from datetime import datetime as dt
from timor import ModuleAssembly, ModulesDB
from timor.utilities.file_locations import get_module_db_files
try:
    from .robot_factory import create_modules_for_2d, create_modules_for_3d
except ImportError:
    from robot_factory import create_modules_for_2d, create_modules_for_3d
import tqdm
from timeit import default_timer as timer


def generate_robots(robot_module_db, num_rob, nModule_min, nModule_max, nJoint_min, nJoint_max, link_sizes=[.2, .3],
                    num_config=0, orientation=False, visual=False, save=False):
    if nModule_min < 3:
        raise ValueError('Robots should consist at least 3 modules (base + link/joint + eef)')

    if robot_module_db == '2d':
        modules = create_modules_for_2d(link_sizes)
    elif robot_module_db == '3d':
        modules = create_modules_for_3d(link_sizes)
    elif robot_module_db == 'geometric_primitive':
        modules = ModulesDB.from_file(*get_module_db_files('geometric_primitive_modules'))
    elif robot_module_db == 'geometric_primitive_revolute_only':
        db = ModulesDB.from_file(*get_module_db_files('geometric_primitive_modules'))
        modules = db.filter(lambda m: m.id != "J1")
    elif type(robot_module_db) == ModulesDB:
        modules = robot_module_db
    else:
        raise ValueError("!!!!Invalid Database of Moudules!!!! robot_module_db should be a timor.ModuleDB \n \
     or string \'2d\'/\'3d\'/\'geometric_primitive\'/\'geometric_primitive_revolute_only\'")

    print(f'Available modules are {modules.all_module_ids}')
    rob_count = 0
    assemblies = []

    if save:
        base_path = Path(__file__).parent.resolve() / Path("./Data/dev_data")
        root_folder_name = (f"Data_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}_r:{num_rob}_s:"
                            f"{num_config}_{nModule_min}_{nModule_max}_{nJoint_max}")
        data_path = base_path / root_folder_name
        if not os.path.exists(data_path):
            data_path.mkdir(parents=True)

    while rob_count < num_rob:
        assembly = ModuleAssembly(modules)
        num_components = np.random.randint(nModule_min - 1, nModule_max)

        for _ in range(num_components):
            if assembly.nJoints >= nJoint_max:
                # Exceeds the upper bound of joint, stop assembling
                break
            assembly.add_random_from_db(lambda x: not any(c.type == 'eef' for c in x.available_connectors.values()))

        if assembly.nJoints < nJoint_min or assembly.nModules < nModule_min - 1:
            continue

        assembly.add_random_from_db(lambda x: any(c.type == 'eef' for c in x.available_connectors.values()))
        print(f'The assembled modules are {assembly.original_module_ids}')

        assert (nJoint_min <= assembly.nJoints <= nJoint_max)
        assert (nModule_min <= assembly.nModules <= nModule_max)

        robot = assembly.to_pin_robot()
        rob_count += 1

        if visual:
            viz = robot.visualize()

        if save:
            data = []
            robot._name = ''
            for id in assembly.original_module_ids:
                robot._name += ''.join(['_', id])
            fn_w = f"Workspace_samples:_{rob_count}{robot._name}.npy"
            fn_a = f"Assembly_samples:_{rob_count}{robot._name}.npy"

        for config_idx in range(num_config):
            random_c = robot.random_configuration()
            # No collision test so far since the joint consisting of two simple cylinder always "collide"
            robot.update_configuration(random_c)
            rob_tcp = robot.fk()
            if save:
                if orientation:
                    data.append(rob_tcp.projection.roto_translation_vector)
                else:
                    data.append(rob_tcp.projection.cartesian)

            if visual:
                # print(f"s_{config_idx} fk {rob_tcp.projection.roto_translation_vector[-3:]}")
                if robot_module_db == '2d':
                    assert (abs(rob_tcp.projection.roto_translation_vector[-1]) < 1e-15)
                viz.updatePlacements(timor.visualization.VISUAL)
                timor.utilities.visualization.place_arrow(viz=viz, name=str(config_idx), placement=rob_tcp)
                time.sleep(0.01)

        if save:
            np_data = np.asarray(data)
            np.save(data_path / fn_w, np_data)
            np.save(data_path / fn_a, assembly.to_json_data())

        assemblies.append(assembly)

    return (assemblies, modules)


def random_unique_generate_robots_2d_optim(robot_module_db, num_rob, nModule_min, nModule_max, nJoint_min, nJoint_max,
                                           link_sizes=[.2, .3], num_config=0, orientation=False, visual=False,
                                           save=False):
    if nModule_min < 3:
        raise ValueError('Robots should consist at least 3 modules (base + link/joint + eef)')

    if robot_module_db == '2d':
        modules = create_modules_for_2d(link_sizes)
    elif robot_module_db == '3d':
        modules = create_modules_for_3d(link_sizes)
    elif robot_module_db == 'geometric_primitive':
        modules = ModulesDB.from_file(*get_module_db_files('geometric_primitive_modules'))
    elif robot_module_db == 'geometric_primitive_revolute_only':
        db = ModulesDB.from_file(*get_module_db_files('geometric_primitive_modules'))
        modules = db.filter(lambda m: m.id != "J1")
    elif type(robot_module_db) == ModulesDB:
        modules = robot_module_db
    else:
        raise ValueError("!!!!Invalid Database of Moudules!!!! robot_module_db should be a timor.ModuleDB \n \
     or string \'2d\'/\'3d\'/\'geometric_primitive\'/\'geometric_primitive_revolute_only\'")

    print(f'Available modules are {modules.all_module_ids}')
    rob_count = 0
    assemblies = []
    assemblyGraphs = set()

    pbar = tqdm.tqdm(total=num_rob)

    while rob_count < num_rob:
        assembly = ModuleAssembly(modules)
        num_components = np.random.randint(nModule_min - 1, nModule_max)

        for _ in range(num_components):
            if assembly.nJoints >= nJoint_max:
                # Exceeds the upper bound of joint, stop assembling
                break
            assembly.add_random_from_db(lambda x: not any(c.type == 'eef' for c in x.available_connectors.values()))

        if assembly.nJoints < nJoint_min or assembly.nModules < nModule_min - 1:
            continue

        assembly.add_random_from_db(lambda x: any(c.type == 'eef' for c in x.available_connectors.values()))

        # assert (nJoint_min <= assembly.nJoints <= nJoint_max)
        # assert (nModule_min <= assembly.nModules <= nModule_max)

        if assembly.original_module_ids not in assemblyGraphs:
            pbar.update(1)
            assemblyGraphs.add(assembly.original_module_ids)
            rob_count += 1
            assemblies.append(assembly)

    return (assemblies, modules)


if __name__ == '__main__':
    r_num = 10000
    start = timer()
    result = random_unique_generate_robots_2d_optim(robot_module_db='2d', num_rob=r_num, nModule_min=3, nModule_max=12,
                                                    nJoint_min=1, nJoint_max=6, link_sizes=[.2, .3], num_config=0,
                                                    visual=False)
    end = timer()
    print(f"Execution time for {r_num} assemblies: {round(end - start, 2)}s")
    print(f"assemblies needs {sys.getsizeof(result[0]) / 1000000} MB")
    assert (len(result[0]) == r_num)
