import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime as dt
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path

from utils.dataloader import get_dummyloaders, get_dataloaders
from model import BinaryTwoDimLSTMModel, SentimentLSTM
from utils.loss import CombinedLoss


# TODO: store trained model weights


def main(data, output_dir, lr, batch_size, pdrop, combo_loss_scale, num_layers, hidden_size, module_type, epochs, id,
         single_sample):
    datetime = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
    writer = SummaryWriter(log_dir=f"runs1/{output_dir}_{data}/id{id}_lr{str(lr)}_{module_type}_{str(datetime)}/")

    # get device info
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    if data == "dummy":
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
    elif data == "2d":
        # Get data
        if single_sample:
            # ------ train data ------
            train_input_file = "data_generation/Data/one_sample_2d/train/Binary_Assembly_r:0f11d85f-03e1-4132-b64f" \
                               "-cd7c5da9a1f5.npy"
            train_label_file = "data_generation/Data/one_sample_2d/train/Grid_Workspace_r:0f11d85f-03e1-4132-b64f" \
                               "-cd7c5da9a1f5.npy"
            train_input = np.load(train_input_file, allow_pickle=True)
            train_label = np.load(train_label_file, allow_pickle=True)
            train_input, train_label = (torch.from_numpy(train_input.astype('float32')).to(device),
                                        torch.from_numpy(train_label.astype('float32')).to(device))
            train_input, train_label = torch.unsqueeze(train_input, 0), torch.unsqueeze(train_label, 0)

            # transform labels into 2d space
            non_zero_indices = torch.nonzero(train_label)
            twodim_labels = train_label[:, :, :, non_zero_indices[0, -1]]
            # non_zero_indices_two = torch.nonzero(twodim_labels)
            train_label = twodim_labels

            input_size = train_input.shape[-1]  # if input (12, 5) -> 12 modules = input length, 5 types = input size
            output_size = train_label.shape[-1]

            # ------ val data ------
            val_input_file = "data_generation/Data/one_sample_2d/val/Binary_Assembly_r:5c86619e-2823-40b3-87a4" \
                             "-a2a43a13dbb0.npy"
            val_label_file = "data_generation/Data/one_sample_2d/val/Grid_Workspace_r:5c86619e-2823-40b3-87a4" \
                             "-a2a43a13dbb0.npy"
            val_input = np.load(val_input_file, allow_pickle=True)
            val_label = np.load(val_label_file, allow_pickle=True)

            val_input = torch.from_numpy(val_input.astype('float32')).to(device)
            val_label = torch.from_numpy(val_label.astype('float32')).to(device)

            val_input = torch.unsqueeze(val_input, 0)
            val_label = torch.unsqueeze(val_label, 0)

            # transform labels into 2d space
            non_zero_val = torch.nonzero(val_label)
            twodim_val_labels = val_label[:, :, :, non_zero_val[0, -1]]
            # non_zero_indeces_two_val = torch.nonzero(twodim_val_labels)
            val_label = twodim_val_labels
        else:
            # get dataloaders
            """data_dir = Path(__file__).resolve().parent / 'data_generation' / 'Data' / 'few_sample_2d'
            train_loader, val_loader, input_size, output_size = get_dataloaders_random_split(batch_size,
            data_dir, data, device)"""

            # TODO: adapt the data import process to the new version of dataset and dataloader
            train_dir = Path(__file__).resolve().parent / 'data' / 'robots_few_2d' / 'train'
            val_dir = Path(__file__).resolve().parent / 'data' / 'robots_few_2d' / 'test'
            train_loader, val_loader, input_size, output_size = get_dataloaders(batch_size=batch_size, data_type=data,
                                                                                device=device, random_split=False,
                                                                                train_dir=train_dir, val_dir=val_dir)

        # define loss function
        # criterion = nn.BCELoss()
        # criterion = WassersteinLoss()
        criterion = CombinedLoss(gamma=combo_loss_scale)
        # criterion = wasserstein_distance
        # criterion = JaccardLoss()

        # initialize model
        net = BinaryTwoDimLSTMModel(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    output_size=output_size,
                                    drop_prob=pdrop,
                                    mode=module_type)

        """net = BinaryTwoDimRNNModel(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              output_size=output_size,
                              drop_prob=pdrop,
                              mode=module_type)"""
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

        if single_sample:
            net.train()

            print("---- TRAINING ----")
            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            train_pred = net(train_input)

            # calculate the loss and perform backprop
            train_true = train_label
            train_loss = criterion(train_pred, train_true)
            train_loss.backward()

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

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
            # nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # log to tensorboard
            writer.add_scalar('Loss/train', train_loss, e)
            print(f'Loss: {train_loss}')
            for name, weight in zip(weight_layer_names, weights):
                writer.add_histogram(f'{name}/weights', weight.data, e)
                writer.add_histogram(f'{name}/gradients', weight.grad, e)

            # validation loop
            print("--- VALIDATION ---")
            with torch.no_grad():
                net.eval()

                val_pred = net(val_input)
                val_true = val_label
                val_loss = criterion(val_pred, val_true)
                # val_loss = torch.tensor(val_loss)

                # log to tensorboard
                writer.add_scalar('Loss/val', val_loss, e)
                print(f'Loss: {val_loss}')

        else:  # loop trough dataloaders
            net.train()
            train_losses = []
            total_train_loss = 0.0
            val_losses = []
            total_val_loss = 0.0

            print("---- TRAINING ----")
            for inputs, labels in tqdm(train_loader, desc="Training"):
                net.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                train_loss = criterion(outputs, labels)
                train_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

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
            avg_epoch_loss = sum(train_losses) / len(train_losses)
            total_train_loss += avg_epoch_loss * inputs.size(0)
            average_train_loss = total_train_loss / len(train_loader.dataset)
            writer.add_scalar('Loss/train', average_train_loss, e)
            print(f'Loss: {train_loss}')
            for name, weight in zip(weight_layer_names, weights):
                writer.add_histogram(f'{name}/weights', weight.data, e)
                writer.add_histogram(f'{name}/gradients', weight.grad, e)

            # validation loop
            print("--- VALIDATION ---")
            with torch.no_grad():
                net.eval()
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    val_loss = criterion(outputs, labels)
                    val_losses.append(val_loss.item())

                # log to tensorboard
                avg_valepoch_loss = sum(val_losses) / len(val_losses)
                total_val_loss += avg_valepoch_loss * inputs.size(0)
                average_val_loss = total_val_loss / len(val_loader.dataset)
                writer.add_scalar('Loss/val', average_val_loss, e)
                print(f'Loss: {val_loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="2d", choices=["2d", "6d", "dummy"])
    parser.add_argument("--output_dir", type=str, default="train_results")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size used for training and validation")
    parser.add_argument("--pdrop", type=float, default=0, help="Dropout probability used in LSTM layers")
    parser.add_argument("--combo_loss_scale", type=float, default=1, help="Scale factor for BCE in the combo loss")
    parser.add_argument("--num_layers", type=int, default=1, help="# of recurrent LSTM layers")
    parser.add_argument("--hidden_size", type=int, default=50, help="hidden size used in the LSTM model layers")
    parser.add_argument("--module_type", default="linear", choices=["linear", "cnn"])
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--id", type=int, default=0, help="identifier for the current run")
    parser.add_argument("--single_sample", type=bool, default=False, help="set for initial overfitting tests")

    args = parser.parse_args()
    kwargs = vars(args)
    print(args)

    main(**kwargs)
