import os
import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime as dt
from tqdm import tqdm
import argparse
from pathlib import Path

from utils.dataloader import get_dummyloaders, get_dataloaders
from model import BinaryTwoDimLSTMModel, SentimentLSTM
from utils.loss import CombinedLoss
from utils.metrics import compute_pos_recall, compute_precision
from sklearn.metrics import f1_score
import json


def make_my_objective(parameter):
    def get_objective_value(trial):
        #  Optuna parameters:
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        hidden_size = trial.suggest_int('hidden_size', 100, 2000, step=100)
        epochs = trial.suggest_int('epochs', 10, 100, step=10)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        pdrop = trial.suggest_float('pdrop', 0.0, 0.5, step=0.1)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        combo_loss_scale = trial.suggest_float('combo_loss_scale', 0.5, 1.0, step=0.1)

        #  User parameters:
        data_type = parameter['data_type']
        data_volume = parameter['data_volume']
        output_dir = parameter['output_dir']
        # batch_size = parameter['batch_size']
        # pdrop = parameter['pdrop']
        # combo_loss_scale = parameter['combo_loss_scale']
        # num_layers = parameter['num_layers']
        module_type = parameter['module_type']
        # epochs = parameter['epochs']
        id = parameter['id']
        optim_mode = parameter['optim_mode']
        save_model = parameter['save_model']
        base_model_path = parameter['model_save_path']
        model_name = parameter['model_name']
        num_workers = parameter["num_workers"]

        datetime = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
        writer = SummaryWriter(
            log_dir=f"tuning_runs/{output_dir}_{data_type}/id{id}_lr{str(lr)}_{module_type}_{str(datetime)}/")

        # get device info
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        if data_type == "dummy":
            print("Using dummy data")
            # get dataloaders
            train_loader, val_loader, vocab_size = get_dummyloaders(batch_size)

            # initialize model
            net = SentimentLSTM(vocab_size=vocab_size + 1,
                                hidden_size=256,
                                num_layers=num_layers,
                                output_size=1,
                                drop_prob=pdrop)
            net = net.to(device)

            # define loss function
            criterion = nn.BCELoss()

        elif data_type == "2d":
            # Get data
            if data_volume == "10k":
                data_dir = Path(__file__).resolve().parent / 'data' / 'robots_10k'
                if optim_mode == 'binary':
                    train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                        data_type=data_type,
                                                                                        device=device,
                                                                                        random_split=True,
                                                                                        binary=True,
                                                                                        data_dir=data_dir)
                    criterion = CombinedLoss(gamma=combo_loss_scale)
                    # criterion = nn.BCELoss()
                    # criterion = WassersteinLoss()
                    # criterion = wasserstein_distance
                    # criterion = JaccardLoss()

                else:  # predict manip_index: regression task
                    # only train with the samples with valid manipulability indices
                    robot_name_file = (Path(__file__).resolve().parent / 'data' / 'robots_10k_info'
                                       / 'versatile_robs.json')
                    train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                        data_type=data_type,
                                                                                        device=device,
                                                                                        random_split=True, binary=False,
                                                                                        data_dir=data_dir,
                                                                                        robot_list=robot_name_file)
                    # TODO: choose proper loss for regression task. Gradients explode when using MSE.
                    criterion = nn.MSELoss()
                    # criterion = nn.L1Loss()
            elif data_volume == '2k':
                try:
                    data_dir = Path(__file__).resolve().parent / 'data' / 'robots_mio' / 'robots_mio'
                except FileNotFoundError:
                    data_dir = Path(__file__).resolve().parent / 'data' / 'robots_mio'

                if optim_mode == 'binary':
                    robot_name_file_train = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                             / 'robots_2k' / 'train.json')
                    robot_name_file_val = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                           / 'robots_2k' / 'test.json')
                    train_loader, val_loader, input_size, output_size = get_dataloaders(
                        batch_size=batch_size,
                        data_type=data_type,
                        device=device,
                        random_split=False, binary=True,
                        num_workers=num_workers,
                        data_dir=data_dir,
                        train_list=robot_name_file_train,
                        val_list=robot_name_file_val
                    )
                    criterion = CombinedLoss(gamma=combo_loss_scale)
                    # criterion = nn.BCELoss()
                    # criterion = WassersteinLoss()
                    # criterion = wasserstein_distance
                    # criterion = JaccardLoss()

                else:  # predict manip_index: regression task
                    robot_name_file_train = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                             / 'robots_2k_reg' / 'train.json')
                    robot_name_file_val = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                           / 'robots_2k_reg' / 'test.json')
                    train_loader, val_loader, input_size, output_size = get_dataloaders(
                        batch_size=batch_size,
                        data_type=data_type,
                        device=device,
                        random_split=False,
                        binary=False,
                        num_workers=num_workers,
                        data_dir=data_dir,
                        train_list=robot_name_file_train,
                        val_list=robot_name_file_val
                    )
                    criterion = nn.MSELoss()
                    # criterion = nn.L1Loss()

            else:
                try:
                    data_dir = Path(__file__).resolve().parent / 'data' / 'robots_mio' / 'robots_mio'
                except FileNotFoundError:
                    data_dir = Path(__file__).resolve().parent / 'data' / 'robots_mio'

                if optim_mode == 'binary':
                    train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                        data_type=data_type,
                                                                                        device=device,
                                                                                        random_split=True, binary=True,
                                                                                        data_dir=data_dir)
                    criterion = CombinedLoss(gamma=combo_loss_scale)
                    # criterion = nn.BCELoss()
                    # criterion = WassersteinLoss()
                    # criterion = wasserstein_distance
                    # criterion = JaccardLoss()

                else:  # predict manip_index: regression task
                    robot_name_file = (Path(__file__).resolve().parent / 'data' / 'robots_manip_split_mio'
                                       / 'versatile_robs.json')
                    train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                        data_type=data_type,
                                                                                        device=device,
                                                                                        random_split=True, binary=False,
                                                                                        data_dir=data_dir,
                                                                                        robot_list=robot_name_file)
                    # criterion = nn.MSELoss()
                    criterion = nn.L1Loss()

            # initialize model
            net = BinaryTwoDimLSTMModel(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        output_size=output_size,
                                        drop_prob=pdrop,
                                        optim_typ=optim_mode,
                                        mode=module_type)

            net = net.to(device)
        else:
            exit(0)

        # set optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        # optimizer = optim.Adamax(net.parameters(), lr=lr)
        # clip = 5.0  # gradient clipping

        # training loop
        for e in range(epochs):
            print(f"--------- Epoch {e + 1} ----------")

            net.train()
            train_losses = []
            val_losses = []
            total_epoch_recalls = 0
            total_epoch_precisions = 0
            total_f1_score = 0

            print("---- TRAINING ----")
            for inputs, labels, lengths in tqdm(train_loader, desc="Training"):
                net.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                assert (-1 not in labels)

                outputs = net(inputs, lengths)
                train_loss = criterion(outputs, labels)
                train_losses.append(train_loss.item())

                train_loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()

            # Log gradients and weights for debugging
            weights = []
            weight_layer_names = []
            for layer_name, param in net.named_parameters():
                if not param.requires_grad:
                    raise ValueError(f"{layer_name} requires gradient computation!")

                if 'weight' in layer_name:
                    weights.append(param)
                    weight_layer_names.append(layer_name)
                    gradient = param.grad
                    if type(gradient) is None:
                        raise TypeError(f"The gradient of {layer_name} is a NoneType Object!")
                    else:
                        print(f"Gradient value: {gradient}")

            # log to tensorboard
            avg_train_loss = sum(train_losses) / len(train_losses)
            writer.add_scalar('Loss/train', avg_train_loss, e)
            print(f'Loss: {train_loss}')
            for name, weight in zip(weight_layer_names, weights):
                writer.add_histogram(f'{name}/weights', weight.data, e)
                writer.add_histogram(f'{name}/gradients', weight.grad, e)

            # validation loop
            print("--- VALIDATION ---")
            with torch.no_grad():
                net.eval()
                for inputs, labels, lengths in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs, lengths)
                    val_loss = criterion(outputs, labels)
                    val_losses.append(val_loss.item())

                    if optim_mode == 'binary':
                        outputs_bi = torch.clone(outputs)
                        outputs_bi[outputs_bi > 0.5] = 1
                        outputs_bi[outputs_bi <= 0.5] = 0
                        pos_recall = compute_pos_recall(labels.view(-1).cpu(), outputs_bi.view(-1).cpu())
                        precision = compute_precision(labels.view(-1).cpu(), outputs_bi.view(-1).cpu())
                        f1 = f1_score(labels.view(-1).cpu(), outputs_bi.view(-1).cpu())
                        total_epoch_recalls += pos_recall
                        total_epoch_precisions += precision
                        total_f1_score += f1

                # log to tensorboard
                avg_val_loss = sum(val_losses) / len(val_losses)
                writer.add_scalar('Loss/val', avg_val_loss, e)
                print(f'Loss: {val_loss}')

                if optim_mode == 'binary':
                    num_batches_val = len(val_losses)
                    average_recall = total_epoch_recalls / num_batches_val
                    print(f'recall: {average_recall}')
                    average_precision = total_epoch_precisions / num_batches_val
                    print(f'precision: {average_precision}')
                    avg_f1 = total_f1_score / num_batches_val
                    print(f'f1 score: {avg_f1}')
                    writer.add_scalar('Metrics/pos_recall', average_recall, e)
                    writer.add_scalar('Metrics/precision', average_precision, e)

            # Pruner can not be implement on multi-objective optimization so far
            # See https://github.com/optuna/optuna/issues/3450
            if optim_mode != 'binary':
                trial.report(value=avg_val_loss, step=e)
                if trial.should_prune():
                    raise optuna.TrialPruned

        if save_model == 'weights':
            folder = os.path.join(base_model_path,
                                  f"t_{trial.number}-{model_name}-{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}")
            Path(folder).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(),
                       os.path.join(folder,
                                    "model.pth"))
            settings_path = os.path.join(folder,
                                         "model_settings.json")
            with open(settings_path, "w") as settings_file:
                if data_type == "2d":
                    m = "BinaryTwoDimLSTMModel"
                else:
                    m = "Unknown"
                model_settings = {
                    'model': m,
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'pdrop': pdrop,
                    'module_type': module_type
                }
                json.dump(model_settings, settings_file)

        if optim_mode == 'binary':
            return avg_val_loss, average_recall, average_precision
        else:
            return avg_val_loss
    return get_objective_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="2d", choices=["2d", "6d", "dummy"])
    parser.add_argument("--data_volume", default="2k", choices=["2k", "10k", "1mio"])
    parser.add_argument("--output_dir", type=str, default="train_results")
    # parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    # parser.add_argument("--batch_size", type=int, default=1, help="batch size used for training and validation")
    # parser.add_argument("--pdrop", type=float, default=0, help="Dropout probability used in LSTM layers")
    # parser.add_argument("--combo_loss_scale", type=float, default=0.5, help="Scale factor for BCE in the combo loss")
    # parser.add_argument("--num_layers", type=int, default=1, help="# of recurrent LSTM layers")
    parser.add_argument("--num_workers", type=int, default=8, help="# of processors for dataloading")
    # parser.add_argument("--hidden_size", type=int, default=2048, help="hidden size used in the LSTM model layers")
    parser.add_argument("--module_type", default="linear", choices=["linear", "cnn"])
    # parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--id", type=int, default=0, help="identifier for the current run")
    parser.add_argument("--optim_mode", default="binary", choices=["binary", "manip_ind"])
    parser.add_argument("--save_model", default="weights", choices=["no", "weights"])
    parser.add_argument("--model_save_path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    parser.add_argument("--model_name", default="model")

    args = parser.parse_args()
    kwargs = vars(args)
    print(args)

    if kwargs["id"] == 0:
        study_name = kwargs['data_volume'] + "_trial_" + str(dt.now().strftime('%Y-%m-%d_%H:%M:%S'))
    else:
        study_name = kwargs['data_volume'] + "_trial_" + str(kwargs["id"])

    # Set up multiprocessing for dataloaders
    mp.set_start_method('spawn')
    print(f"Number of processor for dataloading is {kwargs['num_workers']}")
    # NOTE: based on Optuna pyTorch examples:
    # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    # https://medium.com/swlh/optuna-hyperparameter-optimization-in-pytorch-9ab5a5a39e77
    # https://towardsdatascience.com/hyperparameter-tuning-of-neural-networks-with-optuna-and-pytorch-22e179efc837
    if kwargs['optim_mode'] == 'binary':
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name=study_name,
            directions=["minimize", "maximize", "maximize"],
            # pruner=optuna.pruners.HyperbandPruner(  # pruner parameters
            #    min_resource=3,
            #    max_resource='auto',
            #    reduction_factor=5
            # )
        )

        # some decent HP sets from previous studies
        study.enqueue_trial(
            {
                'batch_size': 32,
                'epochs': 50,
                "hidden_size": 900,
                'learning_rate': .004014030472773061,
                'num_layers': 1,
                'pdrop': 0.5,
                'combo_loss_scale': 0.9
            }
        )

        study.enqueue_trial(
            {
                'batch_size': 16,
                'epochs': 40,
                "hidden_size": 300,
                'learning_rate': 0.007023522566033369,
                'num_layers': 1,
                'pdrop': 0.5
            }
        )

    else:
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.HyperbandPruner(  # pruner parameters
                min_resource=3,
                max_resource='auto',
                reduction_factor=5
            )
        )

        # some decent HP sets from previous studies
        study.enqueue_trial(
            {
                'batch_size': 32,
                'epochs': 50,
                "hidden_size": 900,
                'learning_rate': .004014030472773061,
                'num_layers': 1,
                'pdrop': 0.5
            }
        )

        study.enqueue_trial(
            {
                'batch_size': 16,
                'epochs': 40,
                "hidden_size": 300,
                'learning_rate': 0.007023522566033369,
                'num_layers': 1,
                'pdrop': 0.5
            }
        )

        study.enqueue_trial(
            {
                'batch_size': 16,
                'epochs': 40,
                "hidden_size": 700,
                'learning_rate': 0.004239965887327274,
                'num_layers': 1,
                'pdrop': 0.5
            }
        )

    study.optimize(make_my_objective(kwargs), n_trials=30, timeout=None)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # For multi-objective optimization there can be several "best" trails
    print("Best trials:")
    trials = study.best_trials

    for trial in study.best_trials:

        print("  Value: ", trial.values)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    # main(**kwargs)
