import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import numpy as np
import os
import json
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from string import punctuation
from collections import Counter


class ModRobDataset(Dataset):
    def __init__(self, data_dir, data_type, device, binary=True, robot_list=None):
        self.data_dir = data_dir
        self.device = device
        self.data_type = data_type
        if robot_list is None:
            robot_list_file = data_dir / 'robot_names.json'
        else:
            robot_list_file = robot_list
        self.label_binary = binary

        with open(robot_list_file) as file:
            self.robot_list = json.load(file)

    def __len__(self):
        return len(self.robot_list)

    def __getitem__(self, item):
        input_name = self.data_dir / f'Binary_Assembly_r:{self.robot_list[item]}.npy'
        assembly_name = self.data_dir / f'Assembly_r:{self.robot_list[item]}.pickle'
        label_name = self.data_dir / f'Grid_Workspace_r:{self.robot_list[item]}.npz'
        inputs = np.load(input_name, allow_pickle=True)
        assembly = np.load(assembly_name, allow_pickle=True)
        labels = np.load(label_name, allow_pickle=True)
        labels = labels['arr_0']

        # The raw data is labeled with manip_index
        if self.label_binary:
            labels[labels != 0] = 1
            assert (np.array_equal(np.unique(labels), np.array([0, 1])))  # only 0 and 1

        inputs = torch.from_numpy(inputs.astype('float32')).to(self.device)
        labels = torch.from_numpy(labels.astype('float32')).to(self.device)

        if self.data_type == '2d':
            # transform labels into 2d space
            non_zero_indices = torch.nonzero(labels)
            twodim_labels = labels[:, :, non_zero_indices[0, -1]]
            # non_zero_indices_two = torch.nonzero(twodim_labels)
            labels = twodim_labels
        return inputs, labels, assembly['moduleOrder']


def custom_collate_fn(batch):
    # Pad the variable-sized inputs to the maximum length in the batch
    max_length = max([sample.size(0) for sample, _, _ in batch])
    padded_inputs = []
    labels = []
    lengths = []
    for input, label, _ in batch:
        length = input.size(0)
        padded_input = torch.nn.functional.pad(input, (0, 0, 0, max_length - input.size(0)))
        padded_inputs.append(padded_input)
        labels.append(label)
        lengths.append(length)

    padded_inputs = torch.stack(padded_inputs)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    return padded_inputs, labels, lengths


# TODO: should be improved for more sophisticated use later
def create_subdataset(source_dir, length, nJoint_min, n_Joint_max, n_Module_min, n_Module_max, target_dir=None):
    source_dataset = ModRobDataset(source_dir, '2d', 'cpu')
    count = 0
    subdataset = []
    robot_list = []
    for i in range(len(source_dataset)):
        _, _, assembly = source_dataset[i]
        if not (n_Module_min <= len(assembly) <= n_Module_max):
            continue
        else:
            nJoint = 0
            for module in assembly:
                if 'J' in module:
                    nJoint += 1
            if not (nJoint_min <= nJoint < n_Joint_max):
                continue
            else:
                subdataset.append(assembly)
                robot_list.append(source_dataset.robot_list[i])
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                input_file = source_dir / f'Binary_Assembly_r:{source_dataset.robot_list[i]}.npy'
                assembly_file = source_dir / f'Assembly_r:{source_dataset.robot_list[i]}.pickle'
                label_file = source_dir / f'Grid_Workspace_r:{source_dataset.robot_list[i]}.npz'

                shutil.copy(input_file, target_dir)
                shutil.copy(assembly_file, target_dir)
                shutil.copy(label_file, target_dir)

                count += 1

        if length == count:
            robot_list_path = target_dir / 'robot_names.json'
            with open(robot_list_path, 'w') as file:
                json.dump(robot_list, file)

            break

    return subdataset  # just for checking the robot assemblies


def get_dataloaders(batch_size, data_type, device, random_split, binary, num_workers=1,
                    data_dir=None, robot_list=None, train_list=None, val_list=None):

    if random_split:
        dataset = ModRobDataset(data_dir, data_type, device, binary=binary, robot_list=robot_list)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
        minitrain_data, _ = torch.utils.data.random_split(train_data, [2000, train_size - 2000])
        minival_data, _ = torch.utils.data.random_split(val_data, [500, val_size - 500])

    else:
        train_data = ModRobDataset(data_dir, data_type, device, binary=binary, robot_list=train_list)
        val_data = ModRobDataset(data_dir, data_type, device, binary=binary, robot_list=val_list)

    # get input size from a sample
    sample_input, sample_label, _ = train_data[0]
    input_size = sample_input.size(-1)
    output_size = sample_label.size(-1)

    # get dataloaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn,
                              num_workers=num_workers)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn,
                            num_workers=num_workers)
    if random_split:
        minitrain_loader = DataLoader(minitrain_data, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn,
                                      num_workers=num_workers)
        minival_loader = DataLoader(minival_data, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn,
                                    num_workers=num_workers)
        return minitrain_loader, minival_loader, input_size, output_size
    return train_loader, val_loader, input_size, output_size


def get_test_loader(batch_size, data_type, device, binary, num_workers=1,
                    data_dir=None, robot_list=None):
    test_data = ModRobDataset(data_dir, data_type, device, binary=binary, robot_list=robot_list)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,
                             collate_fn=custom_collate_fn, num_workers=num_workers)
    return test_loader


def get_dummyloaders(batch_size):
    # Read in CSV files
    datafile = '../IMDB Dataset.csv'
    df = pd.read_csv(datafile)

    # Pre-processing steps
    # to lowercase
    df['review'] = df['review'].apply(lambda x: x.lower())
    # remove symbols
    df['clean_text'] = df['review'].apply(lambda x: ''.join([c for c in x if c not in punctuation]))

    # perform word encoding
    # get review lengths
    df['len_review'] = df['clean_text'].apply(lambda x: len(x))
    # extract words, count and sort by frequency
    all_text = df['clean_text'].tolist()
    all_text = ' '.join(all_text)
    all_words = all_text.split()
    word_counts = Counter(all_words)
    sorted_words = word_counts.most_common(len(word_counts))  # returns top 'len(word_counts)' of elements
    # convert words to integer encodings
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}
    vocab_size = len(vocab_to_int)

    reviews_split = df['clean_text'].tolist()
    reviews_int = []
    for review in reviews_split:
        r = [vocab_to_int[w] for w in review.split()]
        reviews_int.append(r)

    # get label encodings
    labels_split = df['sentiment'].tolist()
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

    # Remove outliers, truncate and pad
    reviews_len = [len(x) for x in reviews_int]
    reviews_int = [reviews_int[i] for i, l in enumerate(reviews_len) if l > 0]

    def pad_features(reviews_int, seq_length):
        """ Return features with each review padded with 0's or truncated to the input seq_length"""
        features = np.zeros((len(reviews_int), seq_length), dtype=int)
        for i, review in enumerate(reviews_int):
            review_len = len(review)
            if review_len <= seq_length:
                zeroes = list(np.zeros(seq_length - review_len))
                new = zeroes + review
            elif review_len > seq_length:
                new = review[0:seq_length]
            features[i, :] = np.array(new)
        return features

    features = pad_features(reviews_int, 200)

    # Create Datasets and DataLoaders

    x_train = torch.from_numpy(features[0:int(0.8 * len(features))])
    y_train = torch.from_numpy(encoded_labels[0:int(0.8 * len(features))])
    x_val = torch.from_numpy(features[int(0.8 * len(features)):])
    y_val = torch.from_numpy(encoded_labels[int(0.8 * len(features)):])

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    train_indices = range(0, 5000)
    val_indices = range(0, 1000)
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(val_data, val_indices)

    train_loader = DataLoader(train_subset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_subset, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader, vocab_size


if __name__ == '__main__':
    CREATE_SUBSET = False
    VISUALIZE_DISTRIB = False
    CHECKOUT = False

    DATA_DIR_MIO = Path(__file__).resolve().parent.parent / 'data' / 'robots_mio' / 'robots_mio'
    DATA_DIR_10K = Path(__file__).resolve().parent.parent / 'data' / 'robots_10k'
    ROBOT_LIST_10K = Path(__file__).resolve().parent.parent / 'data' / 'robots_10k_info' / 'versatile_robs.json'
    TARGET_TRAIN = Path(__file__).resolve().parent.parent / 'data' / 'robots_few_2d' / 'train'
    TARGET_TEST = Path(__file__).resolve().parent.parent / 'data' / 'robots_few_2d' / 'test'

    if CREATE_SUBSET:
        full_dataset = ModRobDataset(DATA_DIR_MIO, '2d', 'cpu')

        # Create a dummy dataset with few simple robots
        subdataset = create_subdataset(source_dir=DATA_DIR_MIO, nJoint_min=1, n_Joint_max=2, n_Module_min=4,
                                       n_Module_max=5, length=20, target_dir=TARGET_TRAIN)
        assert (len(subdataset) == 20)

        for assembly in subdataset:
            print(assembly)

        # Create a dummy dataset with few complex robots
        create_subdataset(source_dir=DATA_DIR_MIO, nJoint_min=4, n_Joint_max=6, n_Module_min=6, n_Module_max=10,
                          length=5, target_dir=TARGET_TEST)

    # Visualize the distribution of the labels
    if VISUALIZE_DISTRIB:
        dataset_10k_manip = ModRobDataset(data_dir=DATA_DIR_10K, data_type='2d', device='cpu', binary=False,
                                          robot_list=ROBOT_LIST_10K)

        for i in range(len(dataset_10k_manip)):
            _, labels, assembly = dataset_10k_manip[i]
            flattened_labels = labels.flatten()
            flattened_non_zero_labels = flattened_labels[flattened_labels != 0]
            print(torch.min(flattened_non_zero_labels))
            print(assembly)
            plt.hist(flattened_non_zero_labels, bins=50)
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title('Value Distribution')
            plt.show()

    # check out the "joint distribution" of a dataset
    if CHECKOUT:
        LIST_2K_test = Path(__file__).resolve().parent.parent / 'data' / 'robots_mio_subset' / 'robots_2k' / 'test.json'
        dataset_2k = ModRobDataset(data_dir=DATA_DIR_MIO, data_type='2d', device='cpu', robot_list=LIST_2K_test)
        LIST_100K_TEST = (Path(__file__).resolve().parent.parent / 'data' / 'robots_mio_subset' / 'robots_100k'
                          / 'test.json')
        dataset_100k_test = ModRobDataset(data_dir=DATA_DIR_MIO, data_type='2d', device='cpu', binary=False,
                                          robot_list=LIST_100K_TEST)

        print(len(dataset_100k_test))

        joint_list = []
        for i in range(len(dataset_100k_test)):
            _, _, assembly = dataset_100k_test[i]
            # count joint
            nJoint = 0
            for module in assembly:
                if 'J' in module:
                    nJoint += 1

            joint_list.append(nJoint)

        counted_values = Counter(joint_list)
        for value, count in counted_values.items():
            print(f"{value}: {count}")
