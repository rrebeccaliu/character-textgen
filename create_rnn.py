from lstm_cell import LSTMCell
from basic_rnn_cell import BasicRNNCell
from torch import nn, zeros, empty_like, torch


class CustomRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1, rnn_type='basic_rnn'):
        """
        Creates an recurrent neural network of type {basic_rnn, lstm_rnn}

        basic_rnn is an rnn whose layers implement a tanH activation function
        lstm_rnn is ann rnn whose layers implement an LSTM cell

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in each layer of the RNN.
        num_layers: (int), the number of RNN layers at each time step
        rnn_type: (string), the desired rnn type. rnn_type is a member of {'basic_rnn', 'lstm_rnn'}
        """
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # create a ModuleList self.rnn to hold the layers of the RNN
        # and append the appropriate RNN layers to it
        self.rnn = nn.ModuleList()
        #looping through each layer
        for i in range(num_layers):
            #adding basic rnn
            if rnn_type == "basic_rnn":
                #if first layer input size is vocab size, else input size is hidden size
                if i == 0:
                    self.rnn.append(BasicRNNCell(vocab_size, hidden_size))
                else:
                    self.rnn.append(BasicRNNCell(hidden_size, hidden_size))
            #adding lstm_rnn
            elif rnn_type == "lstm_rnn":
                #if first layer input size is vocab size, else input size is hidden size
                if i == 0:
                    self.rnn.append(LSTMCell(vocab_size, hidden_size))
                else:
                    self.rnn.append(LSTMCell(hidden_size, hidden_size))
 

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an RNN for a given sequence

        Arguments
        ----------
        x: (Tensor) of size (B x T x n) where B is the mini-batch size, T is the sequence length and n is the
            number of input features. x the mini-batch of input sequence
        h: (Tensor) of size (l x B x m) where l is the number of layers and m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (l x B x m). c is the cell state of the previous time step if the rnn is an LSTM RNN

        Return
        ------
        outs: (Tensor) of size (B x T x m), the final hidden state of each time step in order
        h: (Tensor) of size (l x B x m), the hidden state of the last time step
        c: (Tensor) of size (l x B x m), the cell state of the last time step, if the rnn is a basic_rnn, c should be
            the cell state passed in as input.
        """

        # compute the hidden states and cell states (for an lstm_rnn) for each mini-batch in the sequence

        #initialzing outs matrix
        outs = torch.zeros(x.shape[0], x.shape[1], self.hidden_size)
        #initialing h and c previous
        h_prev_t = h 
        c_prev_t = c
        #loop through sequence length T
        for i in range(x.size(1)):
            # matrix containg h and c for each time step
            h_t = torch.zeros((self.num_layers, x.size(0), h.size(2)))
            c_t = torch.zeros((self.num_layers, x.size(0), h.size(2)))
            #indexing the x_prev for this time step
            x_prev = x[:, i, :]

            for (idx, cell) in enumerate(self.rnn):
                #calcualting and updating h with basic rnn forward
                if self.rnn_type == "basic_rnn":
                    h_out = cell(x_prev, h_prev_t[idx,:,:])
                    h_t[idx,:,:] = h_out
                #calcualting and updating h and c with lstm rnn forward
                elif self.rnn_type == "lstm_rnn":
                    h_out, c_out = cell(x_prev, h_prev_t[idx,:,:], c_prev_t[idx,:,:])
                    h_t[idx,:,:] = h_out
                    c_t[idx,:,:] = c_out

                #updating x_prev with the calculated h_out
                x_prev = h_out

            #at the last layer add h to outs
            outs[:, i,:] = h_out
            #update h_prev and c_prev for the next time step
            h_prev_t = h_t
            c_prev_t = c_t



        #update c if lstm rnn
        if self.rnn_type == 'lstm_rnn':
            c = c_prev_t

        return outs, h_prev_t, c
