import json
import os
import numpy as np
import matplotlib.pyplot as plt
import timor
from timor.Module import ModulesDB, ModuleAssembly
import pickle
import time
from pathlib import Path


def displayVoxelData(data, d, min_val, max_val, title, z_slice=-1):
    if z_slice > -1:
        new_data = data[:, :, z_slice].reshape(d, d, -1)
        data = new_data
    value_mapping = np.linspace(min_val, max_val, d)
    x, y, z = data.nonzero()
    if len(x) > 0:
        zipped_grid = list(zip(x, y, z))
        # zipped_grid_scaled = list(zip(
        #     map(lambda i: value_mapping[i], x),
        #     map(lambda i: value_mapping[i], y),
        #     map(lambda i: value_mapping[i], z))
        # )
        r = map(lambda c: data[c[0]][c[1]][c[2]], zipped_grid)
        zipped_grid_scaled_color = list(zip(
            map(lambda i: value_mapping[i], x),
            map(lambda i: value_mapping[i], y),
            map(lambda i: value_mapping[i], z),
            r
        ))
        grid_data_mp = np.array(zipped_grid_scaled_color)

        position_data = grid_data_mp

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(position_data[:, 0], position_data[:, 1], position_data[:, 2], c=position_data[:, 3],
                         cmap=plt.cool())
        # img = ax.scatter(position_data[:, 1], position_data[:, 2], c=position_data[:, 3], cmap=plt.cool())
        fig.colorbar(img)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    else:
        print(f"No WS for {title} found!")
    plt.show()


def displayAssembly(assemblyFile, modulesDB):
    with open(assemblyFile, 'rb') as handle:
        json_assembly = pickle.load(handle)
    assembly = ModuleAssembly.from_json_data(json_assembly, modulesDB)
    robot = assembly.to_pin_robot()
    # fk_voxel_workspace_2d_optim(
    #     robot=robot,
    #     min_val=min_val,
    #     max_val=max_val,
    #     resolution=101,
    #     numb_samples=20000,
    #     r_factor=0,
    #     ignore_self_collisions=False,
    #     with_manip_index=True,
    #     visualisation=True,
    #     displayAllSamples=True,
    #     saving=False,
    #     file_name="",
    #     compressed=True,
    #     info=True
    # )
    viz = robot.visualize()
    viz.updatePlacements(timor.visualization.VISUAL)
    i = 0
    while 100 > i:
        random_c = robot.random_configuration()
        robot.update_configuration(random_c)
        rob_tcp = robot.fk()
        viz.updatePlacements(timor.visualization.VISUAL)
        timor.utilities.visualization.place_arrow(viz=viz, name='1', placement=rob_tcp)
        time.sleep(0.02)
        i += 1


if __name__ == "__main__":
    compressed = True
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "dev_data")
    # folder = 'Data_2023-07-01_15:51:54_r:100_s:1030301'
    # robIds = ["dfda0577-6de8-42ea-ac77-09840688702b"]

    folder = 'Data_2023-07-01_19:23:58_r:10000_s:1030301'

    allRobIds = []
    robots_file_path = os.path.join(base_path, folder, "robot_names.json")
    with open(robots_file_path, "r") as robots_file:
        allRobIds = json.load(robots_file)

    print(f"found {len(allRobIds)} robot assemblies")

    # robId = ['52355d03-17f9-46eb-9c0c-99691a29659d']
    robIds = allRobIds[:10]

    for robId in robIds:
        if compressed:
            fws = f"Grid_Workspace_r:{robId}.npz"
        else:
            fws = f"Grid_Workspace_r:{robId}.npy"
        fas = f"Assembly_r:{robId}.pickle"
        fbn = f"Binary_Assembly_r:{robId}.npy"
        fmod = "modules.json"
        fnWs = os.path.join(base_path, folder, fws)
        fnAs = os.path.join(base_path, folder, fas)
        fnBn = os.path.join(base_path, folder, fbn)
        fnMod = os.path.join(base_path, folder, fmod)

        d = 101  # resolution
        min_val = -3  # minimum value of grid coordinate
        max_val = 3  # maximum value of grid coordinate

        if compressed:
            wsData = np.load(fnWs)["arr_0"]
        else:
            wsData = np.load(fnWs)

        # load modulesDB
        db_import = ModulesDB.from_json_file(filepath=Path(fnMod), package_dir=None)

        displayVoxelData(wsData, d, min_val, max_val, title="test display")
        displayAssembly(fnAs, db_import)
