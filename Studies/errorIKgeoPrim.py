from timor import ModulesDB
import uuid
from timor.utilities.prebuilt_robots import random_assembly
from random import randint
import numpy as np
from timor.utilities import prebuilt_robots
from timor.utilities.tolerated_pose import ToleratedPose, Transformation
import timor
import time


def iterative_manip_matrix_opt(x, y, z, m_iter=200, visual=True, rob=None, ignore_self_col=False,
                               with_manip_index=True):
    if rob is not None:
        robot = rob
    else:
        robot = prebuilt_robots.get_six_axis_modrob()
    current_config = robot.q
    if visual:
        viz = robot.visualize()
        viz.updatePlacements(timor.visualization.VISUAL)
    res = []
    ik_res = []
    off_by_list = []
    for x_i in x:
        for y_i in y:
            for z_i in z:
                if not ignore_self_col:
                    while robot.has_self_collision():
                        print("SELF COL")
                        rc = robot.random_configuration()
                        robot.update_configuration(rc)
                t = Transformation.from_translation([x_i, y_i, z_i])
                if visual:
                    t.visualize(viz, name=str([x_i, y_i, z_i, "T"]))
                p = ToleratedPose(t)
                ik_q, ok = robot.ik_jacobian(p, ignore_self_collision=ignore_self_col, q_init=current_config,
                                             max_iter=m_iter)
                if ok:
                    print("OKOKOK")
                    current_config = ik_q
                    robot.update_configuration(ik_q)
                    if visual:
                        viz.updatePlacements(timor.visualization.VISUAL)
                        timor.utilities.visualization.place_arrow(viz=viz, name=str([x_i, y_i, z_i]),
                                                                  placement=robot.fk())
                    mpi = robot.manipulability_index() if with_manip_index else 1
                    res.append(mpi)
                else:
                    ik_res.append(ik_q.tolist())
                    robot.update_configuration(ik_q)
                    ik_pos = robot.fk().projection.roto_translation_vector[-3:]
                    distToGrid = np.sqrt((x_i - ik_pos[0]) ** 2 + (y_i - ik_pos[1]) ** 2 + (z_i - ik_pos[2]) ** 2)
                    off_by_list.append(distToGrid)
                    if visual:
                        viz.updatePlacements(timor.visualization.VISUAL)
                        timor.utilities.visualization.place_arrow(viz=viz, name=str([x_i, y_i, z_i]),
                                                                  placement=robot.fk())
                    res.append(0.0)
    print("IK results:")
    print(len(ik_res))
    ser = set(tuple(ll) for ll in ik_res)
    print(ser)
    print(len(ser))
    print("OFF EVAL:")
    print(f"min: {min(off_by_list)}")
    print(f"max: {max(off_by_list)}")
    print(f"avg: {sum(off_by_list) / len(off_by_list)}")
    return res


def GeometricPrimitivesRevoluteOnly():
    modules = ModulesDB.from_name('geometric_primitive_modules')
    clean_db = modules.filter(lambda m: m.id != "J1")
    return clean_db


if __name__ == '__main__':
    resolution = 20
    numb_samples = [50000]

    displayRobot = True
    displayWS = True
    ignore_self_collision = False
    with_manip_index = True
    numb_slices = 1

    min_val = -1.7
    max_val = 1.7

    nJoint_min = 5
    nJoint_max = 5

    x = np.linspace(min_val, max_val, resolution)
    y = np.linspace(min_val, max_val, resolution)

    # You can set z = [float] to just compute a slice
    z = np.linspace(min_val, max_val, resolution)
    # z = [0.0]

    modules = GeometricPrimitivesRevoluteOnly()
    j = randint(nJoint_min, nJoint_max)
    assembly = random_assembly(j, modules)
    robot = assembly.to_pin_robot()
    cntr = 0
    while robot.has_self_collision():
        if cntr < 50:
            random_c = robot.random_configuration()
            robot.update_configuration(random_c)
            cntr += 1
        else:
            cntr = 0
            assembly = random_assembly(j, modules)
            robot = assembly.to_pin_robot()
    robot._name = str(uuid.uuid4())

    if displayRobot:
        print("Show robot and some random configs")
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

    # Computing IK
    manip_iter = iterative_manip_matrix_opt(x, y, z, 50, visual=True, rob=robot,
                                            ignore_self_col=ignore_self_collision, with_manip_index=with_manip_index)
