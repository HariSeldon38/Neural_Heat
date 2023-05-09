import torch.nn as nn

# Fully connected neural network with one hidden layer
# Basic model : no multitask
class NeuralNet(nn.Module):
    def __init__(self, hidden_size, input_size=10000, output_size=10000, xdim=100):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.xdim = xdim
        self.ydim = int(output_size/xdim)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = x.view(-1, self.input_size)
        out = nn.ReLU()(self.l1(out))
        out = self.l2(out)
        # no activation at the end
        out = out.view(-1, self.ydim, self.xdim)
        return out

# Convolutionnal neural network
# Basic model : no multitask
class ConvNet(nn.Module):
    def __init__(self, nb_channels, kernel_size, output_size=10000, xdim=100):
        super(ConvNet, self).__init__()
        self.xdim = xdim
        self.ydim = int(output_size/xdim)
        self.conv = nn.Conv2d(1,nb_channels,(kernel_size,kernel_size), padding=int((kernel_size-1)/2), padding_mode='replicate')
        self.act = nn.ReLU()

    def forward(self, x):
        out = x.view(-1,1,self.ydim,self.xdim) #we need the one because conv layer is expected a certain nb of channels
        out = self.act(self.conv(out))
        out = out.view(-1, self.ydim, self.xdim)
        return out
