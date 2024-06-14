import numpy as np
from timor.utilities import prebuilt_robots
from timor.utilities.tolerated_pose import ToleratedPose, Transformation
from multiprocessing import Pool
import timor


# Multiprocessing Helper:
def manipulability_at_row(co):
    robot = prebuilt_robots.get_six_axis_modrob()
    t = Transformation.from_translation(co)
    p = ToleratedPose(t)
    ik_q, ok = robot.ik_jacobian(p, ignore_self_collision=True, max_iter=200)
    if ok:
        mpi = robot.manipulability_index()
        return mpi
    return 0.0


def multiprocessing_manip(arr):
    res = np.apply_along_axis(manipulability_at_row, 1, arr)
    return res


def multiprocess_manip_matrix(arr, n):
    split = np.array_split(arr, n)
    p = Pool(processes=n)
    res = p.map(multiprocessing_manip, split)
    return res


# Iterative Helper:
def iterative_manip_matrix(arr):
    robot = prebuilt_robots.get_six_axis_modrob()
    current_config = robot.q
    res = []
    for i, co in enumerate(arr):
        t = Transformation.from_translation(co)
        p = ToleratedPose(t)
        ik_q, ok = robot.ik_jacobian(p, ignore_self_collision=True, q_init=current_config, max_iter=200)
        current_config = ik_q
        if ok:
            mpi = robot.manipulability_index()
            res.append(mpi)
        else:
            res.append(0.0)
    return res


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
    for x_i in x:
        for y_i in y:
            for z_i in z:
                t = Transformation.from_translation([x_i, y_i, z_i])
                if visual:
                    t.visualize(viz, name=str([x_i, y_i, z_i, "T"]))
                p = ToleratedPose(t)
                ik_q, ok = robot.ik_jacobian(p, ignore_self_collision=ignore_self_col, q_init=current_config,
                                             max_iter=m_iter)
                current_config = ik_q
                if ok:
                    robot.update_configuration(ik_q)
                    if visual:
                        viz.updatePlacements(timor.visualization.VISUAL)
                        timor.utilities.visualization.place_arrow(viz=viz, name=str([x_i, y_i, z_i]),
                                                                  placement=robot.fk())
                    mpi = robot.manipulability_index() if with_manip_index else 1
                    res.append(mpi)
                else:
                    res.append(0.0)
    return res
