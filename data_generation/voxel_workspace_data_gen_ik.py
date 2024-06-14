# import time
from datetime import datetime as dt
# from timor.Robot import PinRobot
# import timor
from timor.utilities import prebuilt_robots
from timor.utilities.tolerated_pose import ToleratedPose, Transformation
import numpy as np
# import pandas as pd
from timeit import default_timer as timer
from voxel_workspace_data_gen_ik_helper import iterative_manip_matrix_opt, multiprocess_manip_matrix
# from voxel_workspace_data_gen_ik_helper import multiprocess_manip_matrix, iterative_manip_matrix
from display_sample_data_voxel_ws import displayVoxelData
import itertools


"""
    Comparison of different workspace generation methods to find optimal IK implementation
"""

robot = prebuilt_robots.get_six_axis_modrob()
robot._name = "Default-6-axis"


# Computes manipulability_index for 3d coordinates "co" for robot "robot"
# def manipulability_at(x, y, z):
#     co = np.array([x, y, z])
#     t = Transformation.from_translation(co)
#     p = ToleratedPose(t)
#     ik_q, ok = robot.ik_jacobian(p, ignore_self_collision=True)
#     if ok:
#         mpi = robot.manipulability_index()
#         return mpi
#     return 0.0


def manipulability_at_row(co):
    t = Transformation.from_translation(co)
    p = ToleratedPose(t)
    ik_q, ok = robot.ik_jacobian(p, ignore_self_collision=True)
    if ok:
        mpi = robot.manipulability_index()
        return mpi
    return 0.0


def multiprocessing_manip(arr):
    return np.apply_along_axis(manipulability_at_row, 1, arr)


# Settings:
n = 16  # number of parallel processes
d = 21  # resolution
min_val = -2  # minimum value of grid coordinate
max_val = 2  # maximum value of grid coordinate

data_store_base_path = "./Data/dev_data/"
number_of_samples = d * d * d
file_name = f"Grid_Workspace_{robot._name}_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}_samples:_{number_of_samples}.npy"
assembly_file_name = f"Assembly_{robot._name}_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}_samples:_{number_of_samples}.npy"
fn = data_store_base_path + file_name

# Data generation:
data = []
x = np.linspace(min_val, max_val, d)
y = np.linspace(min_val, max_val, d)
z = np.linspace(min_val, max_val, d)

# Simple loop iteration:
# start_loop = timer()
# for x_i in x:
#     for y_i in y:
#         for z_i in z:
#             data.append(manipulability_at_row(np.array((x_i, y_i, z_i))))
# np_data_loop = np.asarray(data).reshape((d, d, -1))
# stop_loop = timer()


# Pandas apply version:
# start_pd = timer()
# zipped_grid = list(itertools.product(x, y, z))
# grid_data = np.array(zipped_grid)
# pd_data = pd.DataFrame(grid_data)
# manipulability = pd_data.apply(manipulability_at_row, axis=1)
# grid_data = manipulability.to_numpy()
# np_data_pd = np.asarray(grid_data).reshape((5, 5, -1))
# stop_pd = timer()


# Numpy Vectorize version:
# start_np_vec = timer()
# zipped_grid_vec = list(itertools.product(x, y, z))
# grid_data_vec = np.array(zipped_grid_vec)
# vec_manipulability = np.vectorize(manipulability_at)
# manip_vec = vec_manipulability(grid_data_vec.transpose()[0],
#                                grid_data_vec.transpose()[1], grid_data_vec.transpose()[2])
# np_data_vec = np.asarray(manip_vec).reshape((5, 5, -1))
# stop_np_vec = timer()


# Numpy alogn_axis:
# start_np = timer()
# zipped_grid = list(itertools.product(x, y, z))
# grid_data = np.array(zipped_grid)
# manip = np.apply_along_axis(manipulability_at_row, 1, grid_data)
# np_data = np.asarray(manip).reshape((5, 5, -1))
# stop_np = timer()

# Try threading:
# Numpy alogn_axis + threads:
start_np_mp = timer()
zipped_grid_mp = list(itertools.product(x, y, z))
grid_data_mp = np.array(zipped_grid_mp)
manip_mp = multiprocess_manip_matrix(grid_data_mp, n)
np_data_mp = np.asarray(np.concatenate(manip_mp)).reshape((d, d, -1))
stop_np_mp = timer()

# Try iterative ik:
# start_iter_ik = timer()
# zipped_grid_iter = list(itertools.product(x, y, z))
# grid_data_iter = np.array(zipped_grid_iter)
# manip_iter = iterative_manip_matrix(grid_data_iter)
# np_data_iter = np.asarray(manip_iter).reshape((d, d, -1))
# stop_iter_ik = timer()

# Try iterative ik optimized:
start_iter_ik_opt = timer()
manip_iter = iterative_manip_matrix_opt(x, y, z)
np_data_iter_opt = np.asarray(manip_iter).reshape((d, d, -1))
stop_iter_ik_opt = timer()


# Execution speed comparison
# print(f"Loop: {stop_loop - start_loop}")
# print(f"Pandas: {stop_pd - start_pd}")
# print(f"Numpy Vec: {stop_np_vec - start_np_vec}")
# print(f"Numpy Axis: {stop_np - start_np}")
print(f"Execution time Numpy-Axis-MP: {stop_np_mp - start_np_mp}")
# print(f"Execution time Iterative-Ik: {stop_iter_ik - start_iter_ik}")
print(f"Execution time Iterative-Ik: {stop_iter_ik_opt - start_iter_ik_opt}")
# """
#     Results for 125 coordinates
#     Loop: 9.65043895100007
#     Pandas: 9.64313465899977
#     Numpy Vec: 9.672909493999214
#     Numpy Axis: 9.598153585000546
# """


# saving to file:
# np.save(fn, np_data_iter_opt)
# np.save(data_store_base_path + assembly_file_name, simple_joint_assembly.to_json_data())

# display data:
displayVoxelData(np_data_iter_opt, d, min_val, max_val, title="IK test")
