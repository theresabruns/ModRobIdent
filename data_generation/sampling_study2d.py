import timor
from timeit import default_timer as timer
import numpy as np
import uuid
from display_sample_data_voxel_ws import displayVoxelData
from voxel_workspace_data_gen_fk import fk_voxel_workspace_2d_optim
import os
from RobotsGen.robot_gen_random import random_unique_generate_robots_2d_optim
from datetime import datetime as dt
import time
import matplotlib.pyplot as plt
import json


"""
    Analyses, prints, plots & saves results of sampling study
    takes result of format ident/Studies/study_data_result_structure.json
"""


def analyseResults(results, saveFile=None):
    if saveFile is not None:
        with open(saveFile, "w") as dataFile:
            json.dump(results, dataFile, indent=4)
    robots_ids = results["robot_ids"]
    resolutions = results["resolutions"]
    sample_sizes = results["sample_sizes"]

    points_by_resolution_and_sample_size = {}
    for res in resolutions:
        points_by_resolution_and_sample_size[res] = {}
        for s in sample_sizes:
            points_by_resolution_and_sample_size[res][s] = []

    for rob_id in robots_ids:
        for resolution in resolutions:
            for sample_size in sample_sizes:
                dp = results["data"][rob_id][resolution]["fk"][sample_size]
                points_by_resolution_and_sample_size[resolution][sample_size].append(dp["cnt"])

    print(points_by_resolution_and_sample_size)
    for res in resolutions:
        pltData = {
            "x": [],
            "h": []
        }
        for s_cnt in sample_sizes:
            pltData['x'].append(s_cnt)
            pltData['h'].append(sum(points_by_resolution_and_sample_size[res][s_cnt]))
        print(pltData)
        fig, ax = plt.subplots()
        ax.bar(pltData['x'], pltData['h'])
        ax.set_title(f"Accumulated datapoints at {res} by sample size")
        ax.set_xlabel("sample size")
        ax.set_ylabel("data points")
        plt.show()


if __name__ == '__main__':
    numb_robots = 20
    resolutions = [101]  # [25, 50, 75, 100]  # 10, 15, 20, 25] #, 30, 35, 40]
    numb_samples = [1000, 10201, 20000, 30000, 40000, 50000, 100000]

    displayRobots = False
    displayWS = False
    ignore_self_collision = False
    with_manip_index = True

    min_val = -3
    max_val = 3

    nJoint_min = 4
    nJoint_max = 6
    robotType = '2d'

    results = {
        "robot_ids": [],
        "resolutions": [],
        "slices": [],
        "sample_sizes": [],
        "data": {},
        "settings": {
            "ignore_self_collision": ignore_self_collision,
            "with_manip_index": with_manip_index,
            "robotType": robotType,
            "min_val": min_val,
            "max_val": max_val,
            "nJoint_min": nJoint_min,
            "nJoint_max": nJoint_max
        }
    }

    robots = []
    if robotType == '2d':
        num_config = 0
        nModule_min = 10
        nModule_max = 12
        link_sizes = [.2, .3]
        rob_vis = False
        rob_save = False
        assemblies, modulesDB = random_unique_generate_robots_2d_optim(robot_module_db='2d', num_rob=numb_robots,
                                                                       num_config=num_config, nModule_min=nModule_min,
                                                                       nModule_max=nModule_max, nJoint_min=nJoint_min,
                                                                       nJoint_max=nJoint_max, link_sizes=link_sizes,
                                                                       visual=rob_vis, save=rob_save)
        for assembly in assemblies:
            robot = assembly.to_pin_robot()
            robot._name = str(uuid.uuid4())
            robots.append(robot)

    for s in numb_samples:
        results["sample_sizes"].append(str(s))

    for res in resolutions:
        results["resolutions"].append(str(res))

    for rdx, robot in enumerate(robots):
        print(f"Robot number {rdx}")
        print(f"Robot: {robot._name} with {len(robot.q)} joints")
        results["robot_ids"].append(robot._name)
        results["data"][robot._name] = {
            "info": {"n_joints": len(robot.q)}
        }

        if displayRobots:
            viz = robot.visualize()
            viz.updatePlacements(timor.visualization.VISUAL)
            viz_I = 0
            while 200 > viz_I:
                random_c = robot.random_configuration()
                robot.update_configuration(random_c)
                while robot.has_self_collision():
                    random_c = robot.random_configuration()
                    robot.update_configuration(random_c)
                rob_tcp = robot.fk()
                viz.updatePlacements(timor.visualization.VISUAL)
                timor.utilities.visualization.place_arrow(viz=viz, name=f'{str(viz_I)}', placement=rob_tcp)
                time.sleep(0.02)
                viz_I += 1

        for d in resolutions:
            print(f"Resolution: {d}")
            results["data"][robot._name][str(d)] = {
                "fk": {}
            }
            data_ik = []
            x = np.linspace(min_val, max_val, d)
            y = np.linspace(min_val, max_val, d)
            z = np.linspace(min_val, max_val, d)

            # FK
            for sample_size in numb_samples:
                print(f"computing fk with {sample_size} samples...")
                start_fk = timer()
                ws = fk_voxel_workspace_2d_optim(
                    robot=robot,
                    min_val=min_val,
                    max_val=max_val,
                    resolution=d,
                    numb_samples=sample_size,
                    r_factor=0,
                    ignore_self_collisions=ignore_self_collision,
                    with_manip_index=with_manip_index,
                    visualisation=False,
                    displayAllSamples=False,
                    saving=False,
                    file_name="",
                    compressed=True,
                    info=False
                )
                stop_fk = timer()
                data = np.nan_to_num(ws)
                x_fk, y_fk, z_fk = data.nonzero()
                cnt_fk = len(x_fk)

                # print(f"Execution time FK for res: {d}"
                #       f"\n\tsamples: {sample_size}"
                #       f"\n\tmanipCount: {cnt_fk}"
                #       f"\n\ttime: {round(stop_fk-start_fk, 2)}s")
                results["data"][robot._name][str(d)]["fk"][str(sample_size)] = {
                    "cnt": cnt_fk,
                    "time": round(stop_fk - start_fk, 2)
                }
                if displayWS:
                    displayVoxelData(data, d, min_val, max_val, f"resolution {d} with sample count: {sample_size}")

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Studies")
    save_file = os.path.join(base_path, f"results_2d_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}.json")
    analyseResults(results, save_file)
