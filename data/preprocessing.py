import json
import numpy as np
from pathlib import Path


def select_rob_with_valid_manip_idx(source_folder, target_folder):
    with open(source_folder / 'robot_names.json') as file:
        robot_list = json.load(file)

    num_robs = len(robot_list)
    singular_rob_list = []
    nan_rob_list = []
    versatile_rob_list = []
    for i, robot_id in enumerate(robot_list):
        ws_file = source_folder / f'Grid_Workspace_r:{robot_id}.npz'
        ws_grid = np.load(ws_file, allow_pickle=True)
        ws_grid = ws_grid['arr_0']

        if -1 in ws_grid:
            singular_rob_list.append(robot_id)

        elif np.any(np.isnan(ws_grid)):
            nan_rob_list.append(robot_id)
        else:
            versatile_rob_list.append(robot_id)

        if i % (round(num_robs / 100)) == 0:
            print(f'Progress {round(i/round(num_robs / 100))}%')

    with open(target_folder / 'singular' / 'singular_robs.json', 'w') as file:
        json.dump(singular_rob_list, file)
    with open(target_folder / 'nan' / 'nan_robs.json', 'w') as file:
        json.dump(nan_rob_list, file)
    with open(target_folder / 'versatile' / 'versatile_robs.json', 'w') as file:
        json.dump(versatile_rob_list, file)


def create_robot_dict(data_folder, target_folder, robot_list_file=None):
    # source_folder = Path(__file__).resolve().parent / 'robots_mio' / 'robots_mio'
    if robot_list_file is None:
        with open(data_folder / 'robot_names.json') as file:
            robot_list = json.load(file)
    else:
        with open(robot_list_file) as file:
            robot_list = json.load(file)

    num_robs = len(robot_list)

    num_module_3 = []
    num_module_4 = []
    num_module_5 = []
    num_module_6 = []
    num_module_7 = []
    num_module_8 = []
    num_module_9 = []
    num_module_10 = []
    num_module_11 = []
    num_module_12 = []

    num_joint_1 = []
    num_joint_2 = []
    num_joint_3 = []
    num_joint_4 = []
    num_joint_5 = []
    num_joint_6 = []

    rob_dict = {
        'module_3': num_module_3,
        'module_4': num_module_4,
        'module_5': num_module_5,
        'module_6': num_module_6,
        'module_7': num_module_7,
        'module_8': num_module_8,
        'module_9': num_module_9,
        'module_10': num_module_10,
        'module_11': num_module_11,
        'module_12': num_module_12,

        'joint_1': num_joint_1,
        'joint_2': num_joint_2,
        'joint_3': num_joint_3,
        'joint_4': num_joint_4,
        'joint_5': num_joint_5,
        'joint_6': num_joint_6,
    }

    for i, robot_id in enumerate(robot_list):
        assembly_file = data_folder / f'Assembly_r:{robot_id}.pickle'
        assembly = np.load(assembly_file, allow_pickle=True)

        # group according to num of modules
        num_module = len(assembly['moduleOrder'])
        rob_dict[f'module_{num_module}'].append(robot_id)

        # group according to num of joints
        nJoints = 0
        for module in assembly['moduleOrder']:
            if 'J' in module:
                nJoints += 1

        rob_dict[f'joint_{nJoints}'].append(robot_id)

        if i % (round(num_robs / 100)) == 0:
            print(f'Progress {round(i / round(num_robs / 100))}%')

    rob_summary = {
        'module_3': len(num_module_3),
        'module_4': len(num_module_4),
        'module_5': len(num_module_5),
        'module_6': len(num_module_6),
        'module_7': len(num_module_7),
        'module_8': len(num_module_8),
        'module_9': len(num_module_9),
        'module_10': len(num_module_10),
        'module_11': len(num_module_11),
        'module_12': len(num_module_12),

        'joint_1': len(num_joint_1),
        'joint_2': len(num_joint_2),
        'joint_3': len(num_joint_3),
        'joint_4': len(num_joint_4),
        'joint_5': len(num_joint_5),
        'joint_6': len(num_joint_6),
    }

    with open(target_folder / 'rob_dict.json', 'w') as file:
        json.dump(rob_dict, file)

    with open(target_folder / 'rob_summary.json', 'w') as file:
        json.dump(rob_summary, file)


def create_even_distribute_subsets(num, target_folder, source_file=None, ratios=None, val_set=False):
    """
    default:
    1 joint: 10%
    2 joint: 10%
    3 joint: 20%
    4 joint: 20%
    5 joint: 20%
    6 joint: 20%
    """
    subset_list_train = []
    subset_list_val = []
    subset_list_test = []
    if ratios is None:
        num_list = [num * 0.1, num * 0.1, num * 0.2, num * 0.2, num * 0.2, num * 0.2]
    else:  # since samples in
        num_list = [num * ratios[0], num * ratios[1], num * ratios[2], num * ratios[3], num * ratios[4],
                    num * ratios[5]]

    if source_file is None:
        source_file = Path(__file__).resolve().parent / 'robots_mio_info' / 'rob_dict.json'

    with open(source_file) as file:
        robot_dict = json.load(file)

    for i, num in enumerate(num_list):
        # randomly selected robots from robots with a certain number of joints
        num = int(num)
        pool = robot_dict[f'joint_{i + 1}']
        selected_ind = np.random.choice(np.arange(len(pool)), size=num, replace=False)

        # ensure train set and test set have completely different samples
        if not val_set:
            for idx in selected_ind[:int(num * 0.8)]:
                subset_list_train.append(pool[idx])
            for idx in selected_ind[int(num * 0.8):]:
                subset_list_test.append(pool[idx])
        else:
            for idx in selected_ind[:int(num * 0.8)]:
                subset_list_train.append(pool[idx])
            for idx in selected_ind[int(num * 0.8):int(num * 0.9)]:
                subset_list_val.append(pool[idx])
            for idx in selected_ind[int(num * 0.9):]:
                subset_list_test.append(pool[idx])

        print(f'Samples with joint {i + 1} are added')

    with open(target_folder / 'train.json', 'w') as file:
        json.dump(subset_list_train, file)

    with open(target_folder / 'test.json', 'w') as file:
        json.dump(subset_list_test, file)

    if val_set:
        with open(target_folder / 'val.json', 'w') as file:
            json.dump(subset_list_val, file)


if __name__ == '__main__':
    DATA_DIR_10K = Path(__file__).resolve().parent / 'robots_10k'
    TARGET_LIST_DIR_10K = Path(__file__).resolve().parent / 'robots_10k_info'
    DATA_DIR_MIO = Path(__file__).resolve().parent / 'robots_mio' / 'robots_mio'
    TARGET_LIST_DIR_MIO = Path(__file__).resolve().parent / 'robots_mio_info'

    if not TARGET_LIST_DIR_10K.exists():
        TARGET_LIST_DIR_10K.mkdir()
    if not TARGET_LIST_DIR_MIO.exists():
        TARGET_LIST_DIR_MIO.mkdir()

    # select_rob_with_valid_manip_idx(DATA_DIR_MIO, TARGET_LIST_DIR_MIO)
    # select_rob_with_valid_manip_idx(DATA_DIR_10K, TARGET_LIST_DIR_10K)

    # create_robot_dict(DATA_DIR_10K, TARGET_LIST_DIR_10K)

    # create subsets from mio dataset: 2k (small set for HP tuning)
    TARGET_SUBSET_2K = Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_2k'

    if not TARGET_SUBSET_2K.exists():
        TARGET_SUBSET_2K.mkdir(parents=True)

    # create_even_distribute_subsets(2000, TARGET_SUBSET_2K)

    # create subsets from mio dataset: 5k (big set for HP tuning)
    TARGET_SUBSET_5K = Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_5k'

    if not TARGET_SUBSET_5K.exists():
        TARGET_SUBSET_5K.mkdir(parents=True)

    # create_even_distribute_subsets(5000, TARGET_SUBSET_5K)

    # create subsets from mio dataset: 100k (for final training)
    TARGET_SUBSET_100K = Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_100k'

    if not TARGET_SUBSET_100K.exists():
        TARGET_SUBSET_100K.mkdir(parents=True)
    # we only have 3085 robots with 6 joints
    # create_even_distribute_subsets(100000, TARGET_SUBSET_100K, ratios=[.1, .1, .2, .37, .2, .03], val_set=True)

    TARGET_SUBSET_10K = Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_10k'

    if not TARGET_SUBSET_10K.exists():
        TARGET_SUBSET_10K.mkdir(parents=True)

    # create_even_distribute_subsets(10000, TARGET_SUBSET_10K, val_set=True)

    # To know which kind of robots are singular, nan, and versatile
    ROBOT_LIST_VERSATILE = Path(__file__).resolve().parent / 'robots_mio_info' / 'versatile' / 'versatile_robs.json'
    TARGET_FOLDER_VERSATILE = Path(__file__).resolve().parent / 'robots_mio_info' / 'versatile'
    if not TARGET_FOLDER_VERSATILE.exists():
        TARGET_FOLDER_VERSATILE.mkdir(parents=True)

    # create_robot_dict(data_folder=DATA_DIR_MIO, target_folder=TARGET_FOLDER_VERSATILE,
    #                   robot_list_file=ROBOT_LIST_VERSATILE)

    ROBOT_LIST_NAN = Path(__file__).resolve().parent / 'robots_mio_info' / 'nan' / 'nan_robs.json'
    TARGET_FOLDER_NAN = Path(__file__).resolve().parent / 'robots_mio_info' / 'nan'
    if not TARGET_FOLDER_NAN.exists():
        TARGET_FOLDER_NAN.mkdir(parents=True)

    # create_robot_dict(data_folder=DATA_DIR_MIO, target_folder=TARGET_FOLDER_NAN, robot_list_file=ROBOT_LIST_NAN)

    ROBOT_LIST_SINGULAR = Path(__file__).resolve().parent / 'robots_mio_info' / 'singular' / 'singular_robs.json'
    TARGET_FOLDER_SINGULAR = Path(__file__).resolve().parent / 'robots_mio_info' / 'singular'
    if not TARGET_FOLDER_SINGULAR.exists():
        TARGET_FOLDER_SINGULAR.mkdir(parents=True)

    # create_robot_dict(data_folder=DATA_DIR_MIO, target_folder=TARGET_FOLDER_SINGULAR,
    #                   robot_list_file=ROBOT_LIST_SINGULAR)

    # create subsets from mio dataset: 2k (small set for HP tuning)
    TARGET_SUBSET_2K_REG = Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_2k_reg'

    if not TARGET_SUBSET_2K_REG.exists():
        TARGET_SUBSET_2K_REG.mkdir(parents=True)

    ROBOT_DICT_VERSATILE = Path(__file__).resolve().parent / 'robots_mio_info' / 'versatile' / 'rob_dict.json'
    # create_even_distribute_subsets(2000, TARGET_SUBSET_2K_REG, source_file=ROBOT_DICT_VERSATILE,
    #                               ratios=[0, 0, .25, .25, .25, .25])

    # create subsets from mio versatile dataset: 10k
    TARGET_SUBSET_10K_REG = Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_10k_reg'

    if not TARGET_SUBSET_10K_REG.exists():
        TARGET_SUBSET_10K_REG.mkdir(parents=True)

    ROBOT_DICT_VERSATILE = Path(__file__).resolve().parent / 'robots_mio_info' / 'versatile' / 'rob_dict.json'
    create_even_distribute_subsets(10000, TARGET_SUBSET_10K_REG, source_file=ROBOT_DICT_VERSATILE,
                                   ratios=[0, 0, .25, .25, .25, .25], val_set=True)

    # create subsets from mio versatile dataset: 100k
    TARGET_SUBSET_100K_REG = Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_100k_reg'

    if not TARGET_SUBSET_100K_REG.exists():
        TARGET_SUBSET_100K_REG.mkdir(parents=True)

    ROBOT_DICT_VERSATILE = Path(__file__).resolve().parent / 'robots_mio_info' / 'versatile' / 'rob_dict.json'
    create_even_distribute_subsets(100000, TARGET_SUBSET_100K_REG, source_file=ROBOT_DICT_VERSATILE,
                                   ratios=[0, 0, .25, .52, .20, .03], val_set=True)

    # create a small train subset from 100k for fine-tuning
    ROBOT_LIST_VERSATILE_100K_TRAIN = (Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_100k_reg'
                                       / 'train.json')
    TARGET_FOLDER_VERSATILE_100K_TRAIN = (Path(__file__).resolve().parent / 'robots_mio_subset' / 'robots_100k_reg'
                                          / 'subset_train')
    if not TARGET_FOLDER_VERSATILE_100K_TRAIN.exists():
        TARGET_FOLDER_VERSATILE_100K_TRAIN.mkdir(parents=True)

    create_robot_dict(data_folder=DATA_DIR_MIO, target_folder=TARGET_FOLDER_VERSATILE_100K_TRAIN,
                      robot_list_file=ROBOT_LIST_VERSATILE_100K_TRAIN)

    create_even_distribute_subsets(25000, TARGET_FOLDER_VERSATILE_100K_TRAIN,
                                   source_file=TARGET_FOLDER_VERSATILE_100K_TRAIN / 'rob_dict.json',
                                   ratios=[0, 0, .25, .52, .20, .03], val_set=True)
