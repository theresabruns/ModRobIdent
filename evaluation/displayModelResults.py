import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

base_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_path))
from model import BinaryTwoDimLSTMModel  # noqa
from utils.metrics import compute_pos_recall, compute_precision  # noqa


def predict_workspace(model, inputs):
    length = inputs.shape[0]
    inputs = torch.unsqueeze(torch.from_numpy(inputs.astype('float32')), 0)
    length = torch.tensor([length])
    model.eval()

    with torch.no_grad():
        out = model(inputs, length)
    return out.detach().numpy()


def display2Dpixel(data, d, min_val, max_val):
    """
    Based on the function 'displayVoxelData' in data_generation/display_sample_data_voxel_ws.py
    """
    value_mapping = np.linspace(min_val, max_val, d)
    x, y = np.where(data != 0)
    r = data[x, y]
    zipped_grid_scaled_color = list(zip(
        map(lambda i: value_mapping[i], x),
        map(lambda i: value_mapping[i], y),
        r
    ))
    grid_data_mp = np.array(zipped_grid_scaled_color)

    position_data = grid_data_mp

    return position_data


if __name__ == '__main__':
    # Input (Binary assembly:
    VISUAL_TYPE = "reg_from_scratch"  # "reg_from_scratch", "binary"
    folder = "robots_10k"
    robots_to_test = os.path.join(base_path, 'data', 'model_results_by_hand', 'robots_diff_joints.json')
    robotIds = []
    with open(robots_to_test, "r") as robots_file:
        robotIds = json.load(robots_file)

    # Model:
    if VISUAL_TYPE == 'binary':
        model_path = os.path.join(base_path, 'final_models', 'model_binary_precision_best')
    elif VISUAL_TYPE == 'reg_from_scratch':
        model_path = os.path.join(base_path, 'final_models', 'model_reg_from_scratch')

    if VISUAL_TYPE == 'binary':
        model = BinaryTwoDimLSTMModel(input_size=7,
                                      hidden_size=900,
                                      num_layers=1,
                                      output_size=101,
                                      optim_typ='binary',
                                      drop_prob=0.5,
                                      mode='linear')

    elif VISUAL_TYPE == 'reg_from_scratch':
        model = BinaryTwoDimLSTMModel(input_size=7,
                                      hidden_size=300,
                                      num_layers=1,
                                      output_size=101,
                                      optim_typ='manip_ind',
                                      drop_prob=0.5,
                                      mode='linear')

    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"),
                                     map_location=torch.device('cpu')))

    all_results = []

    for robId in robotIds:
        fws = f"Grid_Workspace_r:{robId}.npz"
        fas = f"Assembly_r:{robId}.pickle"
        fbn = f"Binary_Assembly_r:{robId}.npy"
        fmod = "modules.json"
        fnWs = os.path.join(base_path, 'data', folder, fws)
        fnAs = os.path.join(base_path, 'data', folder, fas)
        fnBn = os.path.join(base_path, 'data', folder, fbn)
        fnMod = os.path.join(base_path, 'data', folder, fmod)
        wsData = np.load(fnWs)["arr_0"]
        non_zero_indices = np.nonzero(wsData)
        ws_2d = wsData[:, :, np.unique(non_zero_indices[2])]
        if VISUAL_TYPE == 'binary':
            ws_2d = wsData[:, :, np.unique(non_zero_indices[2])]
            ws_2d[ws_2d != 0] = 1
        d = 101  # resolution
        min_val = -3  # minimum value of grid coordinate
        max_val = 3  # maximum value of grid coordinate

        inputs = np.load(fnBn, allow_pickle=True)
        modelOut = predict_workspace(model, inputs)
        if VISUAL_TYPE == 'binary':
            modelOut[modelOut > 0.5] = 1
            modelOut[modelOut <= 0.5] = 0

            # compute metrics
            pos_recall = compute_pos_recall(ws_2d.flatten(), modelOut.flatten())
            precision = compute_precision(ws_2d.flatten(), modelOut.flatten())
            print(f'pos_recall: {pos_recall}')
            print(f'precision: {precision}')

        position_data = display2Dpixel(modelOut.squeeze(), d, min_val, max_val)
        plt.figure()
        plt.scatter(position_data[:, 0], position_data[:, 1], c=position_data[:, 2], cmap=plt.cm.cool)
        plt.colorbar()
        plt.title(f"predicted workspace for {robId}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.show()

        position_data = display2Dpixel(ws_2d.squeeze(), d, min_val, max_val)
        plt.figure()
        plt.scatter(position_data[:, 0], position_data[:, 1], c=position_data[:, 2], cmap=plt.cm.cool)
        plt.colorbar()
        plt.title(f"reference workspace for {robId}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.show()
