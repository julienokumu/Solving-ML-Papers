import torch
import torch.nn as nn
import torch.functional as F

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FFNN).__init__()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.W1 = nn.Parameter(torch.randn(input_size, hidden_size1))
        self.b1 = nn.Parameter(torch.randn(hidden_size1))

        self.W2 = nn.Parameter(torch.randn(hidden_size1, hidden_size2))
        self.b2 = nn.Parameter(torch.randn(hidden_size2))

        self.W3 = nn.Parameter(torch.randn(hidden_size2, output_size))
        self.b3 = nn.Parameter(torch.randn(output_size))

    def forward(self, x):

        h1 = torch.matmul(x, self.W1.T)
        h1 = h1 + self.b1
        h1 = F.relu(h1)

        h2 = torch.matmul(h1, self.W2.T)
        h2 = h2 + self.b2
        h2 = F.relu(h2)

        out = torch.matmul(h2, self.W3.T)
        out = out + self.b3

        return out
