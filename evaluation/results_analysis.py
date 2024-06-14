import json
import sys
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from pathlib import Path
from displayModelResults import predict_workspace
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

base_path = Path(__file__).resolve().parent.parent  # /ident folder
sys.path.insert(0, str(base_path))
from model import BinaryTwoDimLSTMModel  # noqa
from utils.metrics import compute_pos_recall, compute_precision  # noqa
from utils.loss import WassersteinLoss, CombinedLoss, JaccardLoss  # noqa


DATA_DIR = base_path / 'data' / 'robots_mio' / 'robots_mio'
DICT_PATH_MIO = base_path / 'data' / 'robots_mio_info' / 'rob_dict.json'
TEST_100K = base_path / 'data' / 'robots_mio_subset' / 'robots_100k' / 'test.json'
TEST_100K_REG = base_path / 'data' / 'robots_mio_subset' / 'robots_100k_reg' / 'test.json'
MODEL_BI_RECALL = base_path / 'final_models' / 'model_binary_recall_best' / 'model.pth'
MODEL_BI_PRECISION = base_path / 'final_models' / 'model_binary_precision_best' / 'model.pth'
MODEL_REG = base_path / 'final_models' / 'model_reg_from_scratch' / 'model.pth'
BINARY = True
REG = False
losses = {
    'joint_1': [], 'joint_2': [], 'joint_3': [],
    'joint_4': [], 'joint_5': [], 'joint_6': [],

    'module_3': [], 'module_4': [], 'module_5': [],
    'module_6': [], 'module_7': [], 'module_8': [],
    'module_9': [], 'module_10': [], 'module_11': [], 'module_12': [],
}

id_for_indexing = {
    'joint_1': [], 'joint_2': [], 'joint_3': [],
    'joint_4': [], 'joint_5': [], 'joint_6': [],

    'module_3': [], 'module_4': [], 'module_5': [],
    'module_6': [], 'module_7': [], 'module_8': [],
    'module_9': [], 'module_10': [], 'module_11': [], 'module_12': [],
}

nan_dict = {}

with open(DICT_PATH_MIO) as file:
    rob_mio_dict = json.load(file)


def find_num_joints_and_modules(rob_id):
    # joint
    if rob_id in rob_mio_dict['joint_4']:
        nJoint = 4
    elif rob_id in rob_mio_dict['joint_5']:
        nJoint = 5
    elif rob_id in rob_mio_dict['joint_3']:
        nJoint = 3
    elif rob_id in rob_mio_dict['joint_1']:
        nJoint = 1
    elif rob_id in rob_mio_dict['joint_2']:
        nJoint = 2
    elif rob_id in rob_mio_dict['joint_6']:
        nJoint = 6

    # module
    if rob_id in rob_mio_dict['module_12']:
        nModule = 12
    elif rob_id in rob_mio_dict['module_11']:
        nModule = 11
    elif rob_id in rob_mio_dict['module_10']:
        nModule = 10
    elif rob_id in rob_mio_dict['module_9']:
        nModule = 9
    elif rob_id in rob_mio_dict['module_8']:
        nModule = 8
    elif rob_id in rob_mio_dict['module_7']:
        nModule = 7
    elif rob_id in rob_mio_dict['module_6']:
        nModule = 6
    elif rob_id in rob_mio_dict['module_5']:
        nModule = 5
    elif rob_id in rob_mio_dict['module_4']:
        nModule = 4
    elif rob_id in rob_mio_dict['module_3']:
        nModule = 3

    return nJoint, nModule


if BINARY:
    PRECISION_BEST = True  # Set to False to get recall best model

    precisions = {
        'joint_1': [], 'joint_2': [], 'joint_3': [],
        'joint_4': [], 'joint_5': [], 'joint_6': [],

        'module_3': [], 'module_4': [], 'module_5': [],
        'module_6': [], 'module_7': [], 'module_8': [],
        'module_9': [], 'module_10': [], 'module_11': [], 'module_12': [],
    }

    recalls = {
        'joint_1': [], 'joint_2': [], 'joint_3': [],
        'joint_4': [], 'joint_5': [], 'joint_6': [],

        'module_3': [], 'module_4': [], 'module_5': [],
        'module_6': [], 'module_7': [], 'module_8': [],
        'module_9': [], 'module_10': [], 'module_11': [], 'module_12': [],
    }

    accuracies = {
        'joint_1': [], 'joint_2': [], 'joint_3': [],
        'joint_4': [], 'joint_5': [], 'joint_6': [],

        'module_3': [], 'module_4': [], 'module_5': [],
        'module_6': [], 'module_7': [], 'module_8': [],
        'module_9': [], 'module_10': [], 'module_11': [], 'module_12': [],
    }

    aucs = {
        'joint_1': [], 'joint_2': [], 'joint_3': [],
        'joint_4': [], 'joint_5': [], 'joint_6': [],

        'module_3': [], 'module_4': [], 'module_5': [],
        'module_6': [], 'module_7': [], 'module_8': [],
        'module_9': [], 'module_10': [], 'module_11': [], 'module_12': [],
    }

    with open(TEST_100K) as file:
        rob_list = json.load(file)

    if PRECISION_BEST:
        model = BinaryTwoDimLSTMModel(input_size=7,
                                      hidden_size=900,
                                      num_layers=1,
                                      output_size=101,
                                      optim_typ='binary',
                                      drop_prob=0.5,
                                      mode='linear')
        model.load_state_dict(torch.load(MODEL_BI_PRECISION, map_location=torch.device('cpu')))
        SAVE_FOLDER = base_path / 'evaluation' / 'binary' / 'precision_best'
        SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    else:  # recall best
        model = BinaryTwoDimLSTMModel(input_size=7,
                                      hidden_size=700,
                                      num_layers=1,
                                      output_size=101,
                                      optim_typ='binary',
                                      drop_prob=0.5,
                                      mode='linear')

        model.load_state_dict(torch.load(MODEL_BI_RECALL, map_location=torch.device('cpu')))
        SAVE_FOLDER = base_path / 'evaluation' / 'binary' / 'recall_best'
        SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    criterion = CombinedLoss(gamma=0.9)

    for i, rob_id in enumerate(rob_list):
        nJoint, nModule = find_num_joints_and_modules(rob_id)
        input_name = DATA_DIR / f'Binary_Assembly_r:{rob_id}.npy'
        label_name = DATA_DIR / f'Grid_Workspace_r:{rob_id}.npz'
        inputs = np.load(input_name, allow_pickle=True)
        labels = np.load(label_name, allow_pickle=True)
        labels = labels['arr_0']
        non_zero_indices = np.nonzero(labels)
        labels = labels[:, :, np.unique(non_zero_indices[2])].squeeze().astype('float32')
        labels[labels != 0] = 1

        outputs = predict_workspace(model=model, inputs=inputs).squeeze().astype('float32')
        loss = criterion(torch.tensor(outputs), torch.tensor(labels)).item()
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        recall = compute_pos_recall(labels.flatten(), outputs.flatten())
        precision = compute_precision(labels.flatten(), outputs.flatten())
        accuracy = accuracy_score(labels.flatten(), outputs.flatten())
        auc = roc_auc_score(labels.flatten(), outputs.flatten())

        # add the values to corresponding lists
        losses[f'joint_{nJoint}'].append(loss)
        losses[f'module_{nModule}'].append(loss)
        accuracies[f'joint_{nJoint}'].append(accuracy)
        accuracies[f'module_{nModule}'].append(accuracy)
        aucs[f'joint_{nJoint}'].append(auc)
        aucs[f'module_{nModule}'].append(auc)
        id_for_indexing[f'joint_{nJoint}'].append(rob_id)
        id_for_indexing[f'module_{nModule}'].append(rob_id)

        if math.isnan(recall) or math.isnan(precision):
            flattened_confusion_matrix = confusion_matrix(labels.flatten(), outputs.flatten()).ravel()
            assembly_name = DATA_DIR / f'Assembly_r:{rob_id}.pickle'
            assembly = np.load(assembly_name, allow_pickle=True)
            nan_dict.update({
                rob_id: {
                    'assembly': assembly['moduleOrder'],
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'confusion_matrix': flattened_confusion_matrix.tolist()

                }
            })
        else:
            precisions[f'joint_{nJoint}'].append(precision)
            precisions[f'module_{nModule}'].append(precision)
            recalls[f'joint_{nJoint}'].append(recall)
            recalls[f'module_{nModule}'].append(recall)

        if (i + 1) % 100 == 0:
            print(f'progress {(i + 1) / 100}%')

    avg_losses = {}
    avg_precisions = {}
    avg_recalls = {}
    avg_accuracies = {}
    avg_aucs = {}

    for key, value in losses.items():
        if not value:  # empty list
            continue
        else:
            avg = sum(value) / len(value)
            avg_losses[key] = avg

    for key, value in precisions.items():
        if not value:  # empty list
            continue
        else:
            avg = sum(value) / len(value)
            avg_precisions[key] = avg

    for key, value in recalls.items():
        if not value:  # empty list
            continue
        else:
            avg = sum(value) / len(value)
            avg_recalls[key] = avg

    for key, value in accuracies.items():
        if not value:  # empty list
            continue
        else:
            avg = sum(value) / len(value)
            avg_accuracies[key] = avg

    for key, value in aucs.items():
        if not value:  # empty list
            continue
        else:
            avg = sum(value) / len(value)
            avg_aucs[key] = avg

    # save results
    with open(SAVE_FOLDER / 'losses.json', 'w') as file:
        json.dump(losses, file)
    with open(SAVE_FOLDER / 'precisions.json', 'w') as file:
        json.dump(precisions, file)
    with open(SAVE_FOLDER / 'recalls.json', 'w') as file:
        json.dump(recalls, file)
    with open(SAVE_FOLDER / 'accuracies.json', 'w') as file:
        json.dump(accuracies, file)
    with open(SAVE_FOLDER / 'aucs.json', 'w') as file:
        json.dump(aucs, file)
    with open(SAVE_FOLDER / 'rob_ids.json', 'w') as file:
        json.dump(id_for_indexing, file)
    with open(SAVE_FOLDER / 'avg_losses.json', 'w') as file:
        json.dump(avg_losses, file)
    with open(SAVE_FOLDER / 'avg_recalls.json', 'w') as file:
        json.dump(avg_recalls, file)
    with open(SAVE_FOLDER / 'avg_precisions.json', 'w') as file:
        json.dump(avg_precisions, file)
    with open(SAVE_FOLDER / 'avg_accuracies.json', 'w') as file:
        json.dump(avg_accuracies, file)
    with open(SAVE_FOLDER / 'avg_aucs.json', 'w') as file:
        json.dump(avg_aucs, file)
    with open(SAVE_FOLDER / 'nan_samples.json', 'w') as file:
        json.dump(nan_dict, file)

elif REG:
    with open(TEST_100K_REG) as file:
        rob_list = json.load(file)

    model = BinaryTwoDimLSTMModel(input_size=7,
                                  hidden_size=300,
                                  num_layers=1,
                                  output_size=101,
                                  optim_typ='manip_ind',
                                  drop_prob=0.5,
                                  mode='linear')

    model.load_state_dict(torch.load(MODEL_REG, map_location=torch.device('cpu')))
    criterion = nn.MSELoss()

    for i, rob_id in enumerate(rob_list):
        nJoint, nModule = find_num_joints_and_modules(rob_id)
        input_name = DATA_DIR / f'Binary_Assembly_r:{rob_id}.npy'
        label_name = DATA_DIR / f'Grid_Workspace_r:{rob_id}.npz'
        inputs = np.load(input_name, allow_pickle=True)
        labels = np.load(label_name, allow_pickle=True)
        labels = labels['arr_0']
        non_zero_indices = np.nonzero(labels)
        labels = labels[:, :, np.unique(non_zero_indices[2])].squeeze().astype('float32')

        outputs = predict_workspace(model=model, inputs=inputs).squeeze().astype('float32')
        loss = criterion(torch.tensor(outputs), torch.tensor(labels)).item()

        # add the values to corresponding lists
        losses[f'joint_{nJoint}'].append(loss)
        losses[f'module_{nModule}'].append(loss)
        id_for_indexing[f'joint_{nJoint}'].append(rob_id)
        id_for_indexing[f'module_{nModule}'].append(rob_id)

        if (i + 1) % 100 == 0:
            print(f'progress {(i + 1) / 100}%')

    avg_losses = {}

    for key, value in losses.items():
        if not value:  # empty list
            continue
        else:
            avg = sum(value) / len(value)
            avg_losses[key] = avg

    # save results
    SAVE_FOLDER = base_path / 'evaluation' / 'manip_ind'
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    with open(SAVE_FOLDER / 'losses.json', 'w') as file:
        json.dump(losses, file)
    with open(SAVE_FOLDER / 'rob_ids.json', 'w') as file:
        json.dump(id_for_indexing, file)
    with open(SAVE_FOLDER / 'avg_losses.json', 'w') as file:
        json.dump(avg_losses, file)

# compute analysis on the robots with 12 modules and different joints
# and on those with 2 joints and different modules
# to see what are the factors that really affects the performance
else:
    ROBOT_DICT = base_path / 'evaluation' / 'binary' / 'precision_best' / 'rob_ids.json'
    NAN_DICT = base_path / 'evaluation' / 'binary' / 'precision_best' / 'nan_samples.json'
    RECALL_DICT = base_path / 'evaluation' / 'binary' / 'precision_best' / 'recalls.json'
    PRECISION_DICT = base_path / 'evaluation' / 'binary' / 'precision_best' / 'precisions.json'
    SAVE_FOLDER_M12 = base_path / 'evaluation' / 'binary' / 'precision_best_m_12'
    SAVE_FOLDER_M12.mkdir(parents=True, exist_ok=True)
    SAVE_FOLDER_J2 = base_path / 'evaluation' / 'binary' / 'precision_best_j_2'
    SAVE_FOLDER_J2.mkdir(parents=True, exist_ok=True)

    with open(ROBOT_DICT) as file:
        rob_dict = json.load(file)
    with open(NAN_DICT) as file:
        nan_dict = json.load(file)
        nan_rob_list = pd.Series(nan_dict.keys())
    with open(RECALL_DICT) as file:
        recall_dict = json.load(file)
        recall_j_2 = pd.Series(recall_dict['joint_2'])
        recall_m_12 = pd.Series(recall_dict['module_12'])
    with open(PRECISION_DICT) as file:
        precision_dict = json.load(file)
        precision_j_2 = pd.Series(precision_dict['joint_2'])
        precision_m_12 = pd.Series(precision_dict['module_12'])

    avg_precisions_m12 = {}
    avg_recalls_m12 = {}
    num_robs_m12 = {}
    avg_precisions_j2 = {}
    avg_recalls_j2 = {}
    num_robs_j2 = {}

    joint1 = pd.Series(rob_dict['joint_1'])
    joint2 = pd.Series(rob_dict['joint_2'])
    joint3 = pd.Series(rob_dict['joint_3'])
    joint4 = pd.Series(rob_dict['joint_4'])
    joint5 = pd.Series(rob_dict['joint_5'])
    joint6 = pd.Series(rob_dict['joint_6'])
    module_7 = pd.Series(rob_dict['module_7'])
    module_8 = pd.Series(rob_dict['module_8'])
    module_9 = pd.Series(rob_dict['module_9'])
    module_10 = pd.Series(rob_dict['module_10'])
    module_11 = pd.Series(rob_dict['module_11'])
    module_12 = pd.Series(rob_dict['module_12'])

    # exclude the nan samples in the list
    module_12_no_nan = module_12[~module_12.isin(nan_rob_list)]
    module_12_no_nan = module_12_no_nan.reset_index(drop=True)
    assert (len(module_12_no_nan) == len(recall_m_12))
    joint_2_no_nan = joint2[~joint2.isin(nan_rob_list)]
    joint_2_no_nan = joint_2_no_nan.reset_index(drop=True)
    assert (len(joint_2_no_nan) == len(precision_j_2))

    # robots with 12 modules and different joints
    for i in range(6):
        robots_same_joints = pd.Series(rob_dict[f'joint_{i + 1}'])
        m_12_same_joints_indices = module_12_no_nan[module_12_no_nan.isin(robots_same_joints)].index
        m_12_same_joints_rec = recall_m_12[m_12_same_joints_indices].sum() / len(m_12_same_joints_indices)
        m_12_same_joints_prec = precision_m_12[m_12_same_joints_indices].sum() / len(m_12_same_joints_indices)
        avg_recalls_m12[f'joint_{i + 1}'] = m_12_same_joints_rec
        avg_precisions_m12[f'joint_{i + 1}'] = m_12_same_joints_prec
        num_robs_m12[f'joint_{i + 1}'] = len(m_12_same_joints_indices)
        print(len(m_12_same_joints_indices))
        print(m_12_same_joints_rec)
        print(m_12_same_joints_prec)

    # robots with 2 joints and different modules
    for i in range(6):
        robots_same_modules = pd.Series(rob_dict[f'module_{i + 7}'])
        j_2_same_modules_indices = joint_2_no_nan[joint_2_no_nan.isin(robots_same_modules)].index
        j_2_same_modules_rec = recall_j_2[j_2_same_modules_indices].sum() / len(j_2_same_modules_indices)
        j_2_same_modules_prec = precision_j_2[j_2_same_modules_indices].sum() / len(j_2_same_modules_indices)
        avg_recalls_j2[f'module_{i + 7}'] = j_2_same_modules_rec
        avg_precisions_j2[f'module_{i + 7}'] = j_2_same_modules_prec
        num_robs_j2[f'module_{i + 7}'] = len(j_2_same_modules_indices)
        print(len(j_2_same_modules_indices))
        print(j_2_same_modules_rec)
        print(j_2_same_modules_prec)

    with open(SAVE_FOLDER_M12 / 'avg_precisions.json', 'w') as file:
        json.dump(avg_precisions_m12, file)
    with open(SAVE_FOLDER_M12 / 'avg_recalls.json', 'w') as file:
        json.dump(avg_recalls_m12, file)
    with open(SAVE_FOLDER_M12 / 'num_robs.json', 'w') as file:
        json.dump(num_robs_m12, file)
    with open(SAVE_FOLDER_J2 / 'avg_precisions.json', 'w') as file:
        json.dump(avg_precisions_j2, file)
    with open(SAVE_FOLDER_J2 / 'avg_recalls.json', 'w') as file:
        json.dump(avg_recalls_j2, file)
    with open(SAVE_FOLDER_J2 / 'num_robs.json', 'w') as file:
        json.dump(num_robs_j2, file)
