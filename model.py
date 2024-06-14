import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.5, mode='cnn'):
        super(BinaryLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.square_size = int(np.sqrt(hidden_size))
        self.flat_size = output_size * output_size * output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=drop_prob, batch_first=True)
        self.mode = mode
        self.fc = nn.Linear(hidden_size, self.flat_size)
        self.cnn1 = nn.Conv2d(1, self.output_size * 2, kernel_size=(3, 3), padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.cnn2 = nn.Conv2d(self.output_size * 2, self.output_size, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, (final_h, final_c) = self.lstm(x, (h0, c0))

        if self.mode == 'linear':  # linear, fully connected module
            out = self.fc(final_h[-1])
            out = torch.reshape(out, (self.output_size, self.output_size, self.output_size))
        else:  # CNN module
            out = torch.reshape(final_h, (1, self.square_size, self.square_size))
            out = F.relu(self.cnn1(out))
            out = self.maxpool(out)
            out = F.relu(self.cnn2(out))
            out = F.interpolate(out.unsqueeze(0), size=(self.output_size, self.output_size), mode='bilinear',
                                align_corners=False)
        return out


class BinaryTwoDimLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, optim_typ, drop_prob=0.5, mode='cnn'):
        super(BinaryTwoDimLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.square_size = int(np.sqrt(hidden_size))
        self.flat_size = output_size * output_size
        self.num_layers = num_layers
        self.optim_typ = optim_typ
        self.mode = mode
        if self.mode == 'cnn':
            self.hidden_size = self.square_size * self.square_size  # change hidden size so it fits square definition
            self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
            self.maxpool = nn.MaxPool2d(3, stride=2)
            self.cnn2 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)
        elif self.mode == 'linear':
            self.fc = nn.Linear(hidden_size, self.flat_size)
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers,
                            dropout=drop_prob, batch_first=True)

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (final_h, final_c) = self.lstm(packed_x, (h0, c0))

        if self.mode == 'linear':  # linear, fully connected module
            out = self.fc(final_h[-1])
            out = torch.reshape(out, (-1, self.output_size, self.output_size))
            if self.optim_typ == 'binary':
                out = torch.sigmoid(out)  # torch.unsqueeze(out, 0)
        else:  # CNN module
            out = torch.reshape(final_h, (1, self.square_size, self.square_size))
            out = self.cnn1(out)
            out = F.tanh(out)
            out = self.maxpool(out)
            out = F.tanh(self.cnn2(out))
            out = F.interpolate(out.unsqueeze(0), size=(self.output_size, self.output_size), mode='bilinear',
                                align_corners=False)
            if self.optim_typ == 'binary':
                out = torch.sigmoid(out.squeeze(dim=1))
        return out


class BinaryTwoDimRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.5, mode='cnn'):
        super(BinaryTwoDimRNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.square_size = int(np.sqrt(hidden_size))
        self.flat_size = output_size * output_size
        self.num_layers = num_layers
        self.mode = mode
        if self.mode == 'cnn':
            self.hidden_size = self.square_size * self.square_size  # change hidden size so it fits square definition
            self.cnn = nn.Conv2d(1, 1, kernel_size=(5, 5), padding=1)
            # self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
            self.maxpool = nn.MaxPool2d(3, stride=2)
            # self.cnn2 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)
        elif self.mode == 'linear':
            self.fc = nn.Linear(hidden_size, self.flat_size)

        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers,
                          dropout=drop_prob, batch_first=True)  # Replaced LSTM with RNN

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, final_h = self.rnn(x, h0)  # Removed cell state (c0) from RNN forward

        """# Check the distribution of the final hidden state
        final_h_flat = np.squeeze(final_h.detach().numpy())"""

        if self.mode == 'linear':  # linear, fully connected module
            out = self.fc(final_h[-1])
            out = torch.reshape(out, (-1, self.output_size, self.output_size))
            out = torch.sigmoid(torch.unsqueeze(out, 0))

        else:  # CNN module
            out = torch.reshape(final_h, (-1, self.square_size, self.square_size))
            # out = F.relu(self.cnn1(out))
            # out = self.maxpool(out)
            # out = F.relu(self.cnn2(out))
            out = F.relu(self.cnn(out))
            out = self.maxpool(out)
            out = F.interpolate(out.unsqueeze(0), size=(self.output_size, self.output_size), mode='bilinear',
                                align_corners=False)
            out = torch.sigmoid(out.squeeze(dim=1))
        return out


class SentimentLSTM(nn.Module):
    """
    The LSTM model used for the dummy use case of sentiment analysis.
    """

    def __init__(self, vocab_size, hidden_size, num_layers, output_size, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        embedding_dim = 400

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)  # stack up outputs

        # dropout and fc-layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)  # reshape to be batch_size first
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden
