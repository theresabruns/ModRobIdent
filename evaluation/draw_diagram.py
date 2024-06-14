import seaborn as sns
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from displayModelResults import predict_workspace, display2Dpixel
# from displayModelResults import *

base_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_path))
from model import BinaryTwoDimLSTMModel  # noqa
from utils.metrics import compute_pos_recall, compute_precision  # noqa


group_results = True
visualization = True
BASE_PATH = Path(__file__).resolve().parent.parent  # ident root folder

# To draw the bar chart for group results of predicting reachability
if group_results:
    save = True
    RESULTS_FOLDER_BINARY = Path(__file__).resolve().parent / 'binary' / 'precision_best'
    RESULTS_FOLDER_REG = Path(__file__).resolve().parent / 'manip_ind'

    with open(RESULTS_FOLDER_BINARY / 'avg_precisions.json') as file:
        avg_precisions = json.load(file)

    with open(RESULTS_FOLDER_BINARY / 'avg_recalls.json') as file:
        avg_recalls = json.load(file)

    joint_group = list(avg_precisions.keys())[:6]
    module_group = list(avg_precisions.keys())[6:]

    joint_recalls = list(avg_recalls.values())[:6]
    joint_precisions = list(avg_precisions.values())[:6]
    module_recalls = list(avg_recalls.values())[6:]
    module_precisions = list(avg_precisions.values())[6:]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # plot bar chart according to the num of joints
    group_names = joint_group + joint_group
    subgroups = ['recall'] * 6 + ['precsion'] * 6
    data_values = joint_recalls + joint_precisions

    sns.barplot(x=group_names, y=data_values, ax=axes[0], hue=subgroups, width=0.6)

    for p in axes[0].patches:  # Add value annotations to the bars
        axes[0].annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 5),
                         textcoords='offset points',
                         fontsize=10)

    axes[0].set_title('Group results regarding the number of joints', y=1.05, fontsize=11, fontname='Times New Roman')
    axes[0].legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize='small',)
    # axes[0].tick_params(axis='x', rotation=5)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # plot bar chart according to the num of modules
    group_names = module_group + module_group
    subgroups = ['recall'] * 7 + ['precsion'] * 7
    data_values = module_recalls + module_precisions

    sns.barplot(x=group_names, y=data_values, ax=axes[1], hue=subgroups, width=0.7)

    for i, p in enumerate(axes[1].patches):  # Add value annotations to the bars

        axes[1].annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 5),
                         textcoords='offset points',
                         fontsize=10)

    axes[1].set_title('Group results regarding the number of modules', y=1.05, fontsize=11, fontname='Times New Roman')
    axes[1].legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize='small')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    # axes[1].tick_params(axis='x', rotation=5)

    # plt.tight_layout()  # Adjust layout for better spacing
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.subplots_adjust(hspace=0.4)
    plt.figtext(0.085, 0.00, "Total Recall: 0.890\nTotal Precision: 0.897", ha='left', fontsize=10)
    if save:
        plt.savefig(base_path / 'evaluation' / 'group_results.png', dpi=600)
    plt.show()


# To visualize the predictied and true workspace
if visualization:
    vertical = False
    save = True
    DOF1_UUID = "1e5fd1a7-4200-42ab-925f-e2e0c7c8dcbc"
    DOF2_UUID = "b3202808-cacd-45be-a50b-343b1b6594ab"
    DOF3_UUID = "5f7db8a4-307f-4c23-81ff-b50f7ba67219"
    DOF4_UUID = "326cf803-a3df-4940-9a38-961af483a660"
    DOF5_UUID = "dab0a3dc-9252-4728-a89a-eda851450da4"
    DOF6_UUID = "64782b30-38d7-48c4-bd75-e6ff5f8be9d9"
    MODEL_REG = BASE_PATH / 'final_models' / 'model_reg_from_scratch' / 'model.pth'
    MODEL_BI = BASE_PATH / 'final_models' / 'model_binary_precision_best' / 'model.pth'
    DATA_FOLDER = BASE_PATH / 'data' / 'robots_10k'

    model_reg = BinaryTwoDimLSTMModel(input_size=7,
                                      hidden_size=300,
                                      num_layers=1,
                                      output_size=101,
                                      optim_typ='manip_ind',
                                      drop_prob=0.5,
                                      mode='linear')

    model_reg.load_state_dict(torch.load(MODEL_REG, map_location=torch.device('cpu')))

    model_bi = BinaryTwoDimLSTMModel(input_size=7,
                                     hidden_size=900,
                                     num_layers=1,
                                     output_size=101,
                                     optim_typ='binary',
                                     drop_prob=0.5,
                                     mode='linear')

    model_bi.load_state_dict(torch.load(MODEL_BI, map_location=torch.device('cpu')))

    samples = [DOF1_UUID, DOF2_UUID, DOF3_UUID, DOF4_UUID, DOF5_UUID, DOF6_UUID]  # , DOF6_UUID, DOF4_UUID
    num_samples = len(samples)

    if vertical:
        fig, axes = plt.subplots(num_samples, 2, figsize=(5, 2 * num_samples))

        for i, sample_id in enumerate(samples):
            fas = f"Assembly_r:{sample_id}.pickle"
            fbn = f"Binary_Assembly_r:{sample_id}.npy"
            fws = f"Grid_Workspace_r:{sample_id}.npz"

            fnWs = DATA_FOLDER / fws
            fnAs = DATA_FOLDER / fas
            fnBn = DATA_FOLDER / fbn
            wsData = np.load(fnWs)["arr_0"]
            non_zero_indices = np.nonzero(wsData)
            ws_2d = wsData[:, :, np.unique(non_zero_indices[2])]

            d = 101  # resolution
            min_val = -3  # minimum value of grid coordinate
            max_val = 3  # maximum value of grid coordinate

            inputs = np.load(fnBn, allow_pickle=True)
            modelOut_reg = predict_workspace(model_reg, inputs)

            mse_loss = ((ws_2d.squeeze() - modelOut_reg.squeeze()) ** 2).mean(axis=None)
            # print(loss)

            # prediction
            position_data = display2Dpixel(modelOut_reg.squeeze(), d, min_val, max_val)
            scatter = axes[i][0].scatter(position_data[:, 0], position_data[:, 1], c=position_data[:, 2],
                                         cmap=plt.cm.cool)
            axes[i][0].set_xlim(min_val, max_val)
            axes[i][0].set_ylim(min_val, max_val)
            axes[i][0].set_title('placeholder for DOF')
            cbar = plt.colorbar(scatter, ax=axes[i][0])

            # label
            position_data = display2Dpixel(ws_2d.squeeze(), d, min_val, max_val)
            scatter = axes[i][1].scatter(position_data[:, 0], position_data[:, 1], c=position_data[:, 2],
                                         cmap=plt.cm.cool)
            axes[i][1].set_xlim(min_val, max_val)
            axes[i][1].set_ylim(min_val, max_val)
            axes[i][1].annotate('placeholder for loss value', xy=(0.5, 0.05), xycoords='axes fraction',
                                fontsize=10, ha='center', va='top')
            # sns.scatterplot(x=position_data[:, 0], y=position_data[:, 1], ax=ax, palette='cool')
            cbar = plt.colorbar(scatter, ax=axes[i][1])

        if save:
            plt.savefig('manip_ind_vertical.png', dpi=600)

    else:  # horizontal
        fig, axes = plt.subplots(3, num_samples, figsize=(5 * num_samples, 12))
        for i, ax in enumerate(axes.flatten()):

            sample_idx = i % num_samples
            fas = f"Assembly_r:{samples[sample_idx]}.pickle"
            fbn = f"Binary_Assembly_r:{samples[sample_idx]}.npy"
            fws = f"Grid_Workspace_r:{samples[sample_idx]}.npz"

            fnWs = DATA_FOLDER / fws
            fnAs = DATA_FOLDER / fas
            fnBn = DATA_FOLDER / fbn
            wsData = np.load(fnWs)["arr_0"]
            non_zero_indices = np.nonzero(wsData)
            ws_2d = wsData[:, :, np.unique(non_zero_indices[2])]
            ws_2d_bi = ws_2d.copy()
            ws_2d_bi[ws_2d_bi != 0] = 1

            d = 101  # resolution
            min_val = -3  # minimum value of grid coordinate
            max_val = 3  # maximum value of grid coordinate

            inputs = np.load(fnBn, allow_pickle=True)
            modelOut_reg = predict_workspace(model_reg, inputs)
            modelOut_bi = predict_workspace(model_bi, inputs)
            modelOut_bi[modelOut_bi > 0.5] = 1
            modelOut_bi[modelOut_bi <= 0.5] = 0

            # avoid nan loss value
            ws_2d[np.isnan(ws_2d)] = 0
            mse_loss = ((ws_2d.squeeze() - modelOut_reg.squeeze()) ** 2).mean(axis=None)
            # print(mse_loss)
            precision = compute_precision(ws_2d_bi.flatten(), modelOut_bi.flatten())
            recall = compute_pos_recall(ws_2d_bi.flatten(), modelOut_bi.flatten())

            if i // num_samples == 0:  # prediction for manip
                position_data = display2Dpixel(modelOut_reg.squeeze(), d, min_val, max_val)
                scatter = ax.scatter(position_data[:, 0], position_data[:, 1], c=position_data[:, 2], cmap=plt.cm.cool)
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.set_title(f'DOF {sample_idx + 1}', fontsize=24)
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.annotate(f'loss: {mse_loss:.4f}', xy=(0.05, 0.1), xycoords='axes fraction',
                            fontsize=20, ha='left', va='top')

            if i // num_samples == 1:  # label
                position_data = display2Dpixel(ws_2d.squeeze(), d, min_val, max_val)
                scatter = ax.scatter(position_data[:, 0], position_data[:, 1], c=position_data[:, 2], cmap=plt.cm.cool)
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.tick_params(axis='both', which='both', labelsize=20)

            if i // num_samples == 2:  # prediction for reachability
                position_data = display2Dpixel(modelOut_bi.squeeze(), d, min_val, max_val)
                scatter = ax.scatter(position_data[:, 0], position_data[:, 1], c=position_data[:, 2], cmap=plt.cm.cool)
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.annotate(f'precision: {precision:.4f} \nrecall: {recall:.4f}', xy=(0.05, 0.165),
                            xycoords='axes fraction', fontsize=20, ha='left', va='top')

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.ax.tick_params(labelsize=20)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        if save:
            plt.savefig(base_path / 'evaluation' / 'manip_ind_horizontal_6.png', dpi=150)

    plt.tight_layout()
    plt.show()
