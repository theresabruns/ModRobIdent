from datetime import datetime as dt
from timor.utilities import prebuilt_robots
import timor
import time
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from timor.Robot import PinRobot
import os
from fk_sampling_helper import deliberate_configuration_sampling
from RobotsGen.robot_gen_random import random_unique_generate_robots_2d_optim
import pinocchio as pin


"""
    This file includes 3 different implementations of FK workspace generation.
    Function: fk_voxel_workspace_2d_optim() is used for the generation of the 1 million dataset
"""


def compute_manip(robot: PinRobot, robot_type='2d') -> float:
    J = pin.computeJointJacobian(robot.model, robot.data, robot.q, robot.model.frames[robot.tcp].parent)
    if robot_type == '2d':
        J = J[[0, 1, 5]]  # only 3 DOF in 2d workspace

    return np.sqrt(np.linalg.det(J @ J.T))


def round_to_scale(p, min_val, max_val, res):
    x_p, y_p, z_p = p[0], p[1], p[2]
    step = (max_val - min_val) / (res - 1)
    x_p = (round((x_p - min_val) / step)) * step + min_val
    y_p = (round((y_p - min_val) / step)) * step + min_val
    z_p = (round((z_p - min_val) / step)) * step + min_val
    p_s = np.asarray([x_p, y_p, z_p])
    return p_s


def get_index_by_coord(p, min_val, max_val, res):
    x_p, y_p, z_p = p[0], p[1], p[2]
    step = (max_val - min_val) / (res - 1)
    x_p = (round((x_p - min_val) / step))
    y_p = (round((y_p - min_val) / step))
    z_p = (round((z_p - min_val) / step))

    x_p = max(0, x_p)
    x_p = min(x_p, (res - 1))
    y_p = max(0, y_p)
    y_p = min(y_p, (res - 1))
    z_p = max(0, z_p)
    z_p = min(z_p, (res - 1))

    return [x_p, y_p, z_p]


def fk_voxel_workspace(
        robot: PinRobot = None,
        min_val: int = -1,
        max_val: int = 1,
        resolution: int = 11,
        numb_samples: int = -1,
        r_factor: int = 0,
        ignore_self_collisions: bool = True,
        with_manip_index: bool = True,
        visualisation: bool = False,
        displayAllSamples: bool = False,
        saving: bool = True,
        file_name: str = "",
        compressed: bool = False,
        info: bool = False
):
    # Information about execution
    start_fk = None
    if info:
        start_fk = timer()

    # Data generation:
    data = []
    vals = np.linspace(min_val, max_val, resolution)
    data_grid = np.zeros((resolution, resolution, resolution))
    if numb_samples == -1:
        numb_samples = resolution * resolution * resolution * r_factor
    self_col_cntr = 0

    all_samples = []

    for i in range(0, numb_samples):
        random_c = robot.random_configuration()
        robot.update_configuration(random_c)
        if (not ignore_self_collisions) and robot.has_self_collision():
            if info:
                print(f"self coll at sample {i} with configuration {random_c}")
            self_col_cntr += 1
            i -= 1
        else:
            rob_tcp = robot.fk()
            robFk_translation = rob_tcp.projection.roto_translation_vector[-3:]
            all_samples.append(robFk_translation)
            try:
                robManip = robot.manipulability_index() if with_manip_index else 1.0
            except Exception as e:  # noqa
                # print(e)
                robManip = -1.0
            index = get_index_by_coord(robFk_translation, min_val, max_val, resolution)
            # print(f"s_{i} fk {robFk.projection.roto_translation_vector[-3:]} --> {index} index")
            if data_grid[index[0]][index[1]][index[2]] == 0.0:
                data_grid[index[0]][index[1]][index[2]] = robManip

    # Information about execution
    if info:
        end_fk = timer()
        print(f"Execution time: {end_fk - start_fk}s")
        print(f"Grid: {resolution}x{resolution}x{resolution} = {resolution * resolution * resolution} voxels")
        print(f"Sampling: {numb_samples} random configurations")
        print(f"resulting in non_zero points: {len(data_grid.nonzero()[0])}")
        print(f"encountered {self_col_cntr} self-collisions while sampling")
        print(f"discrete values:\n{vals}")

    # saving
    if saving:
        if compressed:
            np.savez_compressed(file_name, data_grid)
        else:
            np.save(file_name, data_grid)

    # display:
    if visualisation:
        value_mapping = np.linspace(min_val, max_val, resolution)
        data = data_grid
        x, y, z = data.nonzero()
        zipped_grid = list(zip(x, y, z))
        r = map(lambda c: data[c[0]][c[1]][c[2]], zipped_grid)
        zipped_grid_scaled_color = list(zip(
            map(lambda x_i: value_mapping[x_i], x),
            map(lambda y_i: value_mapping[y_i], y),
            map(lambda z_i: value_mapping[z_i], z),
            r
        ))
        grid_data_mp = np.array(zipped_grid_scaled_color)
        position_data = grid_data_mp
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("visualized fk")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        img = ax.scatter(
            position_data[:, 0],
            position_data[:, 1],
            position_data[:, 2],
            c=position_data[:, 3],
            cmap=plt.cool()
        )
        fig.colorbar(img)
        plt.show()

    if displayAllSamples:
        value_mapping = np.linspace(min_val, max_val, resolution)
        # print(value_mapping)
        x_data = list(map(lambda q: q[0], all_samples))
        y_data = list(map(lambda q: q[1], all_samples))
        z_data = list(map(lambda q: q[2], all_samples))

        samplePlt = plt.figure()
        bx = samplePlt.add_subplot(projection='3d')
        bx.scatter(
            x_data,
            y_data,
            z_data
        )
        bx.set_title(f"visualized fk samples: {numb_samples}")
        bx.set_xlabel("x samples")
        bx.set_ylabel("y samples")
        bx.set_zlabel("z samples")
        plt.show()

    return data_grid


def fk_voxel_workspace_deliberate_configs(
        robot: PinRobot = None,
        min_val: int = -1,
        max_val: int = 1,
        resolution: int = 11,
        r_factor: int = 10,
        ignore_self_collisions: bool = False,
        with_manip_index: bool = True,
        visualisation: bool = False,
        saving: bool = True,
        file_name: str = "",
        info: bool = False,
        d: int = 5
):
    # Information about execution
    start_fk = None
    if info:
        start_fk = timer()

    # Data generation:
    data = []
    vals = np.linspace(min_val, max_val, resolution)
    data_grid = np.zeros((resolution, resolution, resolution))
    numb_samples = resolution * resolution * resolution * r_factor
    self_col_cntr = 0

    configs = deliberate_configuration_sampling(robot, d)

    for i, config in enumerate(configs):
        robot.update_configuration(config)
        if (not ignore_self_collisions) and robot.has_self_collision():
            if info:
                print(f"self coll at sample {i} with configuration {config}")
            self_col_cntr += 1
        else:
            robFk = robot.fk()
            robManip = robot.manipulability_index() if with_manip_index else 1
            index = get_index_by_coord(robFk.projection.roto_translation_vector[-3:], min_val, max_val, resolution)
            if data_grid[index[0]][index[1]][index[2]] == 0.0:
                data_grid[index[0]][index[1]][index[2]] = robManip

    # Information about execution
    if info:
        end_fk = timer()
        print(f"Execution time: {end_fk - start_fk}s")
        print(f"Grid: {resolution}x{resolution}x{resolution} = {resolution * resolution * resolution} voxels")
        print(f"Sampling: {numb_samples} random configurations")
        print(f"resulting in non_zero points: {len(data_grid.nonzero()[0])}")
        print(f"encountered {self_col_cntr} self-collisions while sampling")
        print(f"discrete values:\n{vals}")

    # saving
    if saving:
        np.save(file_name, data_grid)

    # display:
    if visualisation:
        value_mapping = np.linspace(min_val, max_val, resolution)
        data = data_grid
        x, y, z = data.nonzero()
        zipped_grid = list(zip(x, y, z))
        r = map(lambda c: data[c[0]][c[1]][c[2]], zipped_grid)
        zipped_grid_scaled_color = list(zip(
            map(lambda x_i: value_mapping[x_i], x),
            map(lambda y_i: value_mapping[y_i], y),
            map(lambda z_i: value_mapping[z_i], z),
            r
        ))
        grid_data_mp = np.array(zipped_grid_scaled_color)
        position_data = grid_data_mp
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(
            position_data[:, 0],
            position_data[:, 1],
            position_data[:, 2],
            c=position_data[:, 3],
            cmap=plt.cool()
        )
        fig.colorbar(img)
        plt.show()

    return data_grid


def fk_voxel_workspace_2d_optim(
        robot: PinRobot = None,
        min_val: int = -1,
        max_val: int = 1,
        resolution: int = 11,
        numb_samples: int = -1,
        r_factor: int = 0,
        ignore_self_collisions: bool = True,
        with_manip_index: bool = True,
        visualisation: bool = False,
        displayAllSamples: bool = False,
        saving: bool = True,
        file_name: str = "",
        compressed: bool = False,
        info: bool = False
):
    # Information about execution
    upper_max_sample_number = 300000
    start_fk = None
    if info:
        start_fk = timer()

    # Data generation:
    data = []
    vals = np.linspace(min_val, max_val, resolution)
    data_grid = np.zeros((resolution, resolution, resolution))
    if numb_samples == -1:
        numb_samples = resolution * resolution * resolution * r_factor
    self_col_cntr = 0

    all_samples = []

    z_index = resolution // 2

    samp_cntr = 0
    i = 0
    while i < numb_samples:
        samp_cntr += 1
        random_c = robot.random_configuration()
        robot.update_configuration(random_c)
        if (not ignore_self_collisions) and robot.has_self_collision():
            # if info:
            #     print(f"self coll at sample {i} with configuration {random_c}")
            self_col_cntr += 1
            if samp_cntr > upper_max_sample_number:
                i = numb_samples
        else:
            i += 1
            rob_tcp = robot.fk()
            robFk_translation = rob_tcp.projection.roto_translation_vector[-3:]
            all_samples.append(np.append(robFk_translation[:2], 0.0))
            try:
                robManip = compute_manip(robot) if with_manip_index else 1.0
            except Exception as e:  # noqa
                # print(e)
                robManip = -1.0
            index = get_index_by_coord(robFk_translation, min_val, max_val, resolution)
            # print(f"s_{i} fk {robFk.projection.roto_translation_vector[-3:]} --> {index} index")
            if data_grid[index[0]][index[1]][z_index] == 0.0:
                data_grid[index[0]][index[1]][z_index] = robManip

    # Information about execution
    if info:
        end_fk = timer()
        print(f"Execution time: {end_fk - start_fk}s")
        print(f"Grid: {resolution}x{resolution}x{resolution} = {resolution * resolution * resolution} voxels")
        print(f"Sampling: {numb_samples} random configurations")
        print(f"Real number of configs: {samp_cntr}")
        print(f"Number of self-col: {self_col_cntr}")
        print(f"Number of valid configs: {samp_cntr - self_col_cntr}")
        print(f"resulting in non_zero points: {len(data_grid.nonzero()[0])}")
        print(f"encountered {self_col_cntr} self-collisions while sampling")
        print(f"discrete values:\n{vals}")

    # saving
    if saving:
        if compressed:
            np.savez_compressed(file_name, data_grid)
        else:
            np.save(file_name, data_grid)

    # display:
    if visualisation:
        value_mapping = np.linspace(min_val, max_val, resolution)
        data = data_grid
        x, y, z = data.nonzero()
        if x.size > 0:
            zipped_grid = list(zip(x, y, z))
            r = map(lambda c: data[c[0]][c[1]][c[2]], zipped_grid)
            zipped_grid_scaled_color = list(zip(
                map(lambda x_i: value_mapping[x_i], x),
                map(lambda y_i: value_mapping[y_i], y),
                map(lambda z_i: value_mapping[z_i], z),
                r
            ))
            grid_data_mp = np.array(zipped_grid_scaled_color)
            position_data = grid_data_mp
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("visualized fk")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            img = ax.scatter(
                position_data[:, 0],
                position_data[:, 1],
                position_data[:, 2],
                c=position_data[:, 3],
                cmap=plt.cool()
            )
            fig.colorbar(img)
            plt.show()
        else:
            print("No datapoints found for Workspace!")

    if displayAllSamples:
        value_mapping = np.linspace(min_val, max_val, resolution)
        # print(value_mapping)
        x_data = list(map(lambda q: q[0], all_samples))
        y_data = list(map(lambda q: q[1], all_samples))
        z_data = list(map(lambda q: q[2], all_samples))

        samplePlt = plt.figure()
        bx = samplePlt.add_subplot(projection='3d')
        bx.scatter(
            x_data,
            y_data,
            z_data
        )
        bx.set_title(f"visualized fk samples: {numb_samples}")
        bx.set_xlabel("x samples")
        bx.set_ylabel("y samples")
        bx.set_zlabel("z samples")
        plt.show()

    return data_grid


"""
    generate grid based data of robot configurations
"""
if __name__ == '__main__':
    robot = prebuilt_robots.get_six_axis_modrob()
    robot._name = "Default-6-axis"

    # Settings:
    min_val = -3  # minimum value of grid coordinate
    max_val = 3  # maximum value of grid coordinate
    numb_samples = 20000
    ignore_self_collisions = False
    with_manip_index = True
    visualisation = True  # show workspace visualisation
    saving = False  # save to file

    data_store_base_path = os.path.join(os.getcwd(), "Data", "dev_data")
    file_name = f"Grid_Workspace_{robot._name}_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}" \
                f"_samples:_{numb_samples}.npy"
    assembly_file_name = f"Assembly_{robot._name}_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}" \
                         f"_samples:_{numb_samples}.npy"
    fn = os.path.join(data_store_base_path, file_name)

    robot_type = '2d'
    numb_robots = 10  # number of robot assemblies to generate
    num_config = 0
    nModule_min = 3
    nModule_max = 12
    nJoint_min = 1
    nJoint_max = 3
    link_sizes = [.2, .3]
    rob_vis = False
    rob_save = False

    assemblies, modulesDB = random_unique_generate_robots_2d_optim(robot_module_db=robot_type, num_rob=numb_robots,
                                                                   num_config=num_config, nModule_min=nModule_min,
                                                                   nModule_max=nModule_max, nJoint_min=nJoint_min,
                                                                   nJoint_max=nJoint_max, link_sizes=link_sizes,
                                                                   visual=rob_vis, save=rob_save)

    for i, assembly in enumerate(assemblies):
        robot = assembly.to_pin_robot()
        print(f'test robot {i+1}')

        fk_voxel_workspace_2d_optim(
            robot=robot,
            min_val=min_val,
            max_val=max_val,
            resolution=101,
            numb_samples=numb_samples,
            r_factor=0,
            ignore_self_collisions=ignore_self_collisions,
            with_manip_index=with_manip_index,
            visualisation=True,
            displayAllSamples=True,
            saving=False,
            file_name="",
            compressed=True,
            info=True
        )

        viz = robot.visualize()
        viz.updatePlacements(timor.visualization.VISUAL)
        i = 0
        while 100 > i:
            random_c = robot.random_configuration()
            robot.update_configuration(random_c)
            if not robot.has_self_collision():
                rob_tcp = robot.fk()
                assert (abs(rob_tcp.projection.roto_translation_vector[-1]) < 1e-15)
                viz.updatePlacements(timor.visualization.VISUAL)
                timor.utilities.visualization.place_arrow(viz=viz, name='1', placement=rob_tcp)
                time.sleep(0.01)
                i += 1
