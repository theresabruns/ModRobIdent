import numpy as np
from typing import List
from timor.Robot import PinRobot
from RobotsGen.robot_gen_random import generate_robots
from itertools import product


def deliberate_configuration_sampling(robot: PinRobot, d: int) -> List[np.ndarray]:
    print(robot.njoints)
    print(robot.q)
    print(robot.joints)
    print(robot.joint_limits)
    joint_ranges = []
    if robot.njoints > 0:
        for i in range(0, robot.njoints):
            print(i)
            minJ = robot.joint_limits[0][i]
            maxJ = robot.joint_limits[1][i]
            samples = np.linspace(minJ, maxJ, d)
            joint_ranges.append(samples)
        configs = [np.asarray(x) for x in product(*joint_ranges)]
        return configs
    return []


if __name__ == '__main__':
    numb_robots = 1  # number of robot assemblies to generate
    num_config = 0
    nModule_min = 3
    nModule_max = 12
    nJoint_max = 6
    link_sizes = [.2, .5]
    rob_vis = False
    rob_save = False

    assemblies, modulesDB = generate_robots(numb_robots, num_config, nModule_min, nModule_max, nJoint_max, link_sizes,
                                            visual=rob_vis, save=rob_save)

    assembly = assemblies[0]
    robot = assembly.to_pin_robot()
    configs = deliberate_configuration_sampling(robot, 10)
    print(configs)
