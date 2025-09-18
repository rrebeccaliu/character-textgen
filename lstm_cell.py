from torch import nn, sigmoid, tanh, Tensor, torch
from math import sqrt


class LSTMCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Creates an RNN layer with an LSTM activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        super(LSTMCell, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # W, the input weights matrix has size (n x (4 * m)) where n is
        # the number of input features and m is the hidden size
        # V, the hidden state weights matrix has size (m, (4 * m))
        # b, the vector of biases has size (4 * m)

        #calculating k
        k = 1/sqrt(hidden_size)
        #initializing parameters
        self.W = nn.Parameter((torch.empty(vocab_size,4*hidden_size)).uniform_(-k,k))
        self.V = nn.Parameter((torch.empty(hidden_size,4*hidden_size)).uniform_(-k,k))
        self.b = nn.Parameter((torch.empty(4*hidden_size)).uniform_(-k,k))

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an LSTM layer

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input-features.
            If the RNN has only one layer at each time step, x is the input data of the current time-step.
            In a multi-layer RNN, x is the previous layer's hidden state (usually after applying a dropout)
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m), the cell state of the previous time step

        Return
        ------
        h_out: (Tensor) of size (B x m), the new hidden
        c_out: (Tensor) of size (B x m), he new cell state

        """

        #calculating activation function
        a = self.b+ torch.mm(x, self.W) + torch.mm(h, self.V)

        #caluclating i,f, o,g
        i = sigmoid(a[:, 0:self.hidden_size])
        f = sigmoid(a[:,self.hidden_size:2*self.hidden_size])
        o = sigmoid(a[:,2*self.hidden_size:3*self.hidden_size])
        g = tanh(a[:,3*self.hidden_size: 4*self.hidden_size])

        #calculating c and h out
        c_out = i * g + f * c
        h_out = o * tanh(c) 

        return h_out, c_out


