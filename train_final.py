import os
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime as dt
from tqdm import tqdm
import argparse
from pathlib import Path

from utils.dataloader import get_dataloaders, get_test_loader
from model import BinaryTwoDimLSTMModel
from utils.loss import CombinedLoss
from utils.metrics import compute_pos_recall, compute_precision
from sklearn.metrics import roc_auc_score


def main(parameter):
    #  User parameters:
    lr = parameter['lr']
    hidden_size = parameter['hidden_size']
    data_type = parameter['data_type']
    data_volume = parameter['data_volume']
    output_dir = parameter['output_dir']
    batch_size = parameter['batch_size']
    pdrop = parameter['pdrop']
    combo_loss_scale = parameter['combo_loss_scale']
    num_layers = parameter['num_layers']
    module_type = parameter['module_type']
    epochs = parameter['epochs']
    id = parameter['id']
    optim_mode = parameter['optim_mode']
    test_mode = parameter['test_mode']
    base_model_path = parameter['model_save_path']
    model_name = parameter['model_name']
    num_workers = parameter["num_workers"]
    model_random_init = parameter["model_random_init"]
    model_import_path = parameter["model_import_path"]

    datetime = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
    writer = SummaryWriter(
        log_dir=f"final_runs/{output_dir}_{data_type}/id{id}_lr{str(lr)}_{module_type}_{str(datetime)}/")

    # get device info
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    if data_type == "2d":
        # Get data
        try:
            DATA_DIR = Path(__file__).resolve().parent / 'data' / 'robots_mio' / 'robots_mio'
        except FileNotFoundError:
            DATA_DIR = Path(__file__).resolve().parent / 'data' / 'robots_mio'

        if data_volume == '10k':  # use 10k to test the functionality of the whole pipeline before test with 100k
            if optim_mode == 'binary':
                robot_name_file_train = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                         / 'robots_10k' / 'train.json')
                robot_name_file_val = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                       / 'robots_10k' / 'val.json')
                train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                    data_type=data_type, device=device,
                                                                                    random_split=False, binary=True,
                                                                                    num_workers=num_workers,
                                                                                    data_dir=DATA_DIR,
                                                                                    train_list=robot_name_file_train,
                                                                                    val_list=robot_name_file_val)
                criterion = CombinedLoss(gamma=combo_loss_scale)

            else:  # predict manip_index: regression task
                robot_name_file_train = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                         / 'robots_10k_reg' / 'train.json')
                robot_name_file_val = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                       / 'robots_10k_reg' / 'val.json')
                train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                    data_type=data_type, device=device,
                                                                                    random_split=False, binary=False,
                                                                                    num_workers=num_workers,
                                                                                    data_dir=DATA_DIR,
                                                                                    train_list=robot_name_file_train,
                                                                                    val_list=robot_name_file_val)
                criterion = nn.MSELoss()

        elif data_volume == '100k':
            if optim_mode == 'binary':
                robot_name_file_train = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                         / 'robots_100k' / 'train.json')
                robot_name_file_val = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                       / 'robots_100k' / 'val.json')
                train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                    data_type=data_type, device=device,
                                                                                    random_split=False, binary=True,
                                                                                    num_workers=num_workers,
                                                                                    data_dir=DATA_DIR,
                                                                                    train_list=robot_name_file_train,
                                                                                    val_list=robot_name_file_val)
                criterion = CombinedLoss(gamma=combo_loss_scale)

            else:  # predict manip_index: regression task
                robot_name_file_train = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                         / 'robots_100k_reg' / 'train.json')
                robot_name_file_val = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset'
                                       / 'robots_100k_reg' / 'val.json')
                train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size,
                                                                                    data_type=data_type, device=device,
                                                                                    random_split=False, binary=False,
                                                                                    num_workers=num_workers,
                                                                                    data_dir=DATA_DIR,
                                                                                    train_list=robot_name_file_train,
                                                                                    val_list=robot_name_file_val)
                criterion = nn.MSELoss()

        else:
            raise ValueError("NO such data volume!!!")

        if not test_mode:
            # initialize model
            net = BinaryTwoDimLSTMModel(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        output_size=output_size,
                                        drop_prob=pdrop,
                                        optim_typ=optim_mode,
                                        mode=module_type)

            if not model_random_init:  # train a model based on an existing model
                try:
                    net.load_state_dict(torch.load(model_import_path))
                except RuntimeError:
                    net.load_state_dict(torch.load(model_import_path, map_location=torch.device('cpu')))

            net = net.to(device)

    else:
        exit(0)

    if not test_mode:
        # set optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        # optimizer = optim.Adamax(net.parameters(), lr=lr)

        # initialize lowest validation loss
        lowest_val_loss = math.inf

        # training loop
        for e in range(epochs):
            print(f"--------- Epoch {e + 1} ----------")

            net.train()
            train_losses = []
            val_losses = []
            total_epoch_recalls = 0
            total_epoch_precisions = 0

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

            # log to tensorboard
            avg_train_loss = sum(train_losses) / len(train_losses)
            writer.add_scalar('Loss/train', avg_train_loss, e)
            print(f'Loss: {train_loss}')

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
                        total_epoch_recalls += pos_recall
                        total_epoch_precisions += precision

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
                    writer.add_scalar('Metrics/pos_recall', average_recall, e)
                    writer.add_scalar('Metrics/precision', average_precision, e)

                # save the model with the lowest val loss
                if avg_val_loss < lowest_val_loss:
                    lowest_val_loss = avg_val_loss
                    folder = os.path.join(base_model_path,
                                          f"{model_name}-{dt.now().strftime('%Y-%m-%d_%H:%M:%S')}")
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    torch.save(net.state_dict(), os.path.join(folder, "model.pth"))

    # -----TEST-----
    test_total_recall = 0
    test_total_precision = 0
    test_total_roc_auc = 0
    test_losses = []
    test_total_nan_batch = 0
    total_test_time = 0
    if data_type == '2d':
        final_net = BinaryTwoDimLSTMModel(input_size=input_size,
                                          hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          output_size=output_size,
                                          optim_typ=optim_mode,
                                          drop_prob=pdrop,
                                          mode=module_type)
    else:
        exit(0)

    final_net.to(device)

    if not test_mode:
        if device == torch.device("cuda"):
            final_net.load_state_dict(torch.load(os.path.join(folder, "model.pth")))
        elif device == torch.device("cpu"):
            final_net.load_state_dict(torch.load(os.path.join(folder, "model.pth"), map_location=torch.device('cpu')))

        # Set epoch = 0 to run the test set on the trained model so local var folder is not assigned
    else:  # test mode
        if device == torch.device("cuda"):
            final_net.load_state_dict(torch.load(os.path.join(model_import_path)))
        elif device == torch.device("cpu"):
            final_net.load_state_dict(torch.load(os.path.join(model_import_path), ap_location=torch.device('cpu')))

    if optim_mode == 'binary':
        if data_volume == '10k':
            test_robots_file = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset' / 'robots_10k'
                                / 'test.json')
            test_loader = get_test_loader(batch_size=1, data_type=data_type, device=device, num_workers=num_workers,
                                          binary=True, data_dir=DATA_DIR, robot_list=test_robots_file)
        if data_volume == '100k':
            test_robots_file = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset' / 'robots_100k'
                                / 'test.json')
            test_loader = get_test_loader(batch_size=1, data_type=data_type, device=device, num_workers=num_workers,
                                          binary=True, data_dir=DATA_DIR, robot_list=test_robots_file)
    else:  # manip_ind
        if data_volume == '10k':
            test_robots_file = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset' / 'robots_10k_reg'
                                / 'test.json')
            test_loader = get_test_loader(batch_size=1, data_type=data_type, device=device, num_workers=num_workers,
                                          binary=False, data_dir=DATA_DIR, robot_list=test_robots_file)
        if data_volume == '100k':
            test_robots_file = (Path(__file__).resolve().parent / 'data' / 'robots_mio_subset' / 'robots_100k_reg'
                                / 'test.json')
            test_loader = get_test_loader(batch_size=1, data_type=data_type, device=device, num_workers=num_workers,
                                          binary=False, data_dir=DATA_DIR, robot_list=test_robots_file)

    with torch.no_grad():
        for inputs, labels, lengths in tqdm(test_loader, desc="Test"):
            final_net.eval()
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()
            outputs = final_net(inputs, lengths)
            end_time = time.time()
            inference_time = end_time - start_time
            total_test_time += inference_time

            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            if optim_mode == 'binary':
                outputs_bi = torch.clone(outputs)
                outputs_bi[outputs_bi > 0.5] = 1
                outputs_bi[outputs_bi <= 0.5] = 0
                pos_recall = compute_pos_recall(labels.view(-1).cpu(), outputs_bi.view(-1).cpu())
                precision = compute_precision(labels.view(-1).cpu(), outputs_bi.view(-1).cpu())
                roc_auc = roc_auc_score(labels.view(-1).cpu(), outputs_bi.view(-1).cpu())
                test_total_roc_auc += roc_auc

                if math.isnan(pos_recall) or math.isnan(precision):
                    test_total_nan_batch += 1
                    continue
                else:
                    test_total_recall += pos_recall
                    test_total_precision += precision

    avg_test_loss = sum(test_losses) / len(test_loader)
    print(f'Avg Test Loss: {avg_test_loss}')
    print(f'Inference time for {data_volume} samples on {device} is {total_test_time:.3f} seconds')
    if optim_mode == 'binary':
        num_batches_test = len(test_loader)
        average_recall = test_total_recall / (num_batches_test - test_total_nan_batch)
        print(f'recall: {average_recall}')
        average_precision = test_total_precision / (num_batches_test - test_total_nan_batch)
        print(f'precision: {average_precision}')
        average_auc = test_total_roc_auc / num_batches_test
        print(f'auc: {average_auc}')
        print(f'number of nan batches: {test_total_nan_batch}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="2d", choices=["2d", "6d"])
    parser.add_argument("--data_volume", default="100k", choices=["10k", "100k"])
    parser.add_argument("--output_dir", type=str, default="train_results")
    parser.add_argument("--lr", type=float, default=0.0007155167770564766, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size used for training and validation")
    parser.add_argument("--pdrop", type=float, default=0.5, help="Dropout probability used in LSTM layers")
    parser.add_argument("--combo_loss_scale", type=float, default=0.9, help="Scale factor for BCE in the combo loss")
    parser.add_argument("--num_layers", type=int, default=1, help="# of recurrent LSTM layers")
    parser.add_argument("--num_workers", type=int, default=8, help="# of processors for dataloading")
    parser.add_argument("--hidden_size", type=int, default=700, help="hidden size used in the LSTM model layers")
    parser.add_argument("--module_type", default="linear", choices=["linear", "cnn"])
    parser.add_argument("--epochs", type=int, default=40, help="number of training epochs")
    parser.add_argument("--id", type=int, default=0, help="identifier for the current run")
    parser.add_argument("--optim_mode", default="binary", choices=["binary", "manip_ind"])
    parser.add_argument("--model_save_path",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_models'))
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--test_mode", type=bool, default=False,
                        help="whether train a new model or directly test an existing model")
    parser.add_argument("--model_random_init", type=bool, default=True,
                        help="whether train the model based on an existing model")
    parser.add_argument("--model_import_path",
                        default=os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), 'final_models/model_binary_precision_best/model.pth'),
                        help="if in test mode, this is the model to be test; "
                             "if model is not randomly initialized, this is the initial model for training")

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

    main(kwargs)
