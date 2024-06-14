import timor
from timeit import default_timer as timer
import numpy as np
import uuid
from timor.utilities import prebuilt_robots
from voxel_workspace_data_gen_ik_helper import iterative_manip_matrix_opt
from display_sample_data_voxel_ws import displayVoxelData
from voxel_workspace_data_gen_fk import fk_voxel_workspace
import os
from RobotsGen.robot_gen_random import generate_robots
from datetime import datetime as dt
from archive_data_generation.automatic_data_generation_helper import GeometricPrimitivesRevoluteOnly
from timor.utilities.prebuilt_robots import random_assembly
from random import randint
import time
import matplotlib.pyplot as plt
import json


"""
    Analyses, prints, plots & saves results of sampling study
    takes result of format ident/Studies/study_data_result_structure.json
"""


def analyse_results(results, save_file=None):
    if save_file is not None:
        with open(save_file, "w") as dataFile:
            json.dump(results, dataFile, indent=4)
    robots_ids = results["robot_ids"]
    resolutions = results["resolutions"]
    slices = results["slices"]
    sample_sizes = results["sample_sizes"]

    # by sample size
    cov_by_sample_plt_data = {
        "x": [],
        "h": []
    }

    for res in resolutions:
        coverage_by_sample = {}
        for n in sample_sizes:
            coverage_by_sample[str(n)] = []
        for rob in robots_ids:
            for slc in slices:
                for sample_size in sample_sizes:
                    cov = results["data"][rob][res][slc]["fk"][sample_size]["coverage"]
                    coverage_by_sample[str(sample_size)].append(cov)
        for nn in numb_samples:
            avg_cov = round(sum(coverage_by_sample[str(nn)]) / len(coverage_by_sample[str(nn)]), 2)
            cov_by_sample_plt_data['x'].append(str(nn))
            cov_by_sample_plt_data['h'].append(avg_cov)
            print(f"Avg. Coverage for {nn} samples: {avg_cov}% @ resolution {res}")
        cov_by_sample_plt_data['x'].append("IK (ref)")
        cov_by_sample_plt_data['h'].append(100)
        fig, ax = plt.subplots()
        ax.bar(cov_by_sample_plt_data['x'], cov_by_sample_plt_data['h'])
        ax.set_title(f"Average Coverage by Sample Size (6-axis default robot) at resolution {res}")
        ax.set_xlabel("sample size")
        ax.set_ylabel("coverage in %")
        plt.show()


if __name__ == '__main__':
    numb_robots = 1
    resolutions = [20, 40, 60, 80, 100]  # [25, 50, 75, 100]  # 10, 15, 20, 25] #, 30, 35, 40]
    numb_samples = [50000, 100000, 125000, 150000, 175000, 200000]

    displayRobots = True
    displayWS = True
    ignore_self_collision = False
    with_manip_index = False
    numb_slices = 3

    min_val = -2
    max_val = 2

    nJoint_min = 5
    nJoint_max = 5
    robotType = '3d default'

    max_iter_ik = 50

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
            "nJoint_max": nJoint_max,
            "max_iter_ik": max_iter_ik
        }
    }

    robots = []
    if robotType == 'geometric primitive':
        modules = GeometricPrimitivesRevoluteOnly()
        for nr in range(0, numb_robots):
            j = randint(nJoint_min, nJoint_max)
            assembly = random_assembly(j, modules)
            robot = assembly.to_pin_robot()
            while robot.has_self_collision():
                assembly = random_assembly(j, modules)
                robot = assembly.to_pin_robot()
            robot._name = str(uuid.uuid4())
            robots.append(robot)
    elif robotType == '2d':
        num_config = 10
        nModule_min = 3
        nModule_max = 7
        link_sizes = [.2, .3]
        rob_vis = False
        rob_save = False
        assemblies, modulesDB = generate_robots(robot_module_db='2d', num_rob=numb_robots, num_config=num_config,
                                                nModule_min=nModule_min, nModule_max=nModule_max, nJoint_min=nJoint_min,
                                                nJoint_max=nJoint_max, link_sizes=link_sizes, visual=rob_vis,
                                                save=rob_save)
        for assembly in assemblies:
            robot = assembly.to_pin_robot()
            robot._name = str(uuid.uuid4())
            robots.append(robot)
    elif robotType == '3d default':
        robot = prebuilt_robots.get_six_axis_modrob()
        robot._name = str(uuid.uuid4())
        robots.append(robot)

    for robot in robots:
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
            results["resolutions"].append(str(d))
            results["data"][robot._name][str(d)] = {}
            data_ik = []
            x = np.linspace(min_val, max_val, d)
            y = np.linspace(min_val, max_val, d)
            z = np.linspace(min_val, max_val, d)

            fk_workspaces = {}

            # FK
            for s in numb_samples:
                print(f"computing fk with {s} samples...")
                results["sample_sizes"].append(str(s))
                start_fk = timer()
                ws = fk_voxel_workspace(
                    robot=robot,
                    min_val=min_val,
                    max_val=max_val,
                    resolution=d,
                    numb_samples=s,
                    r_factor=0,
                    ignore_self_collisions=ignore_self_collision,
                    with_manip_index=with_manip_index,
                    visualisation=False,
                    displayAllSamples=False,
                    saving=False,
                    file_name="",
                    info=False
                )
                ws_clean = np.nan_to_num(ws)
                stop_fk = timer()
                # safe workspace for slice-wise comparison with ik:
                fk_workspaces[str(s)] = (ws_clean, round(stop_fk - start_fk, 2))

            for slc in range(0, numb_slices):
                # sliceId = randint(0, d - 1)
                sliceId = randint(round(d / 2) - round(d * 0.3), round(d / 2) + round(d * 0.3))
                zz = [list(z)[sliceId]]
                results["slices"].append(str(slc))
                results["data"][robot._name][str(d)][str(slc)] = {
                    "fk": {},
                    "ik": {},
                    "slc_info": {
                        "slice_id": sliceId,
                        "z_val": zz[0]
                    }
                }

                # IK
                ref_ik = 0
                print(f"slice {sliceId} --> z={zz[0]}")
                try:
                    print("computing ik...")
                    start_iter_ik_opt = timer()
                    # manip_iter = iterative_manip_matrix_opt(x, y, zz, 100, visual=True, rob=robot,
                    # ignore_self_col=ignore_self_collision)
                    manip_iter = iterative_manip_matrix_opt(x, y, zz, max_iter_ik, visual=False, rob=robot,
                                                            ignore_self_col=ignore_self_collision,
                                                            with_manip_index=with_manip_index)
                    np_data_iter_opt = np.asarray(manip_iter).reshape((d, d, -1))
                    x_ik, y_ik, z_ik = np_data_iter_opt.nonzero()
                    cnt_ik = len(x_ik)
                    stop_iter_ik_opt = timer()
                    print(f"Execution time Iterative-Ik for res {d}"
                          f"\n\tmanipCnt: {cnt_ik}\n\t"
                          f"time: {round(stop_iter_ik_opt - start_iter_ik_opt, 2)}s")
                    results["data"][robot._name][str(d)][str(slc)]["ik"] = {
                        "cnt": cnt_ik,
                        "time": round(stop_iter_ik_opt - start_iter_ik_opt, 2)
                    }
                    ref_ik = cnt_ik
                    if displayWS:
                        displayVoxelData(np_data_iter_opt, d, min_val, max_val, f"ik res: {d}")
                except Exception as e:
                    print("Exception computing IK:")
                    print(e)

                # FK slice evaluation:
                for sample_size in numb_samples:
                    data = fk_workspaces[str(sample_size)][0]
                    new_data = data[:, :, sliceId].reshape(d, d, -1)
                    x_fk, y_fk, z_fk = fk_workspaces[str(sample_size)][0].nonzero()
                    xy_slice = [cor for cor in zip(x_fk, y_fk, z_fk) if cor[2] == sliceId]
                    cnt_fk = len(xy_slice)
                    if ref_ik == 0:
                        coverage = round((1 - cnt_fk / (d * d)) * 100, 2)
                    else:
                        coverage = round((cnt_fk / ref_ik) * 100, 2)
                    if coverage > 100:
                        coverage = round(100 - (coverage - 100), 2)
                    print(f"Execution time FK for res: {d}"
                          f"\n\tsamples: {sample_size}"
                          f"\n\tmanipCount: {cnt_fk}"
                          f"\n\tcoverage: {coverage}%"
                          f"\n\ttime: {fk_workspaces[str(sample_size)][1]}s")
                    results["data"][robot._name][str(d)][str(slc)]["fk"][str(sample_size)] = {
                        "cnt": cnt_fk,
                        "coverage": coverage,
                        "time": fk_workspaces[str(sample_size)][1]
                    }
                    if displayWS:
                        displayVoxelData(fk_workspaces[str(sample_size)][0], d, min_val, max_val,
                                         f"resolution {d} at z-slice: {slc} with sample count: {sample_size}",
                                         sliceId)

    print(results)
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Studies")
    print(base_path)
    save_file = os.path.join(base_path, f"results_{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}.json")
    analyse_results(results, save_file)
