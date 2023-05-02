import torch
import torch.nn as nn
import torchvision.transforms as transforms
from customDataset import HeatDiffusion
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#hyperparameters
epochs = 25
n_train = 200
batch_size = 20
lr = 0.001 #learning rate
hidden = 2000 #nb of neurons in the hidden layer
channels = (12,15) #nb output channel in the con layers
kernel_size = (5,5) #kernel size for the conv layers
input_folder = 'data/0_003/T0blocksMap/iteration_no0'
output_folder = 'data/0_003/T0blocksMap/iteration_no10'
schedule = False
sched_step = [15] #epoch steps when we want to decrease the lr by a factor 2

#access to tensorboard:
#cmd : tensorboard --logdir="C:\Users\gaia3\AppData\Local\Programs\Python\PythonProjects\Neural_Heat/runs/CNN/
#web = http://localhost:6006/
tensorboard = True
tensorboard_path = 'runs/CNN/1lr'
tensorboard_name = f"training{n_train}_batch{batch_size}_lr{lr}_hid{hidden}_chan1_{channels[0]}_chan2_{channels[1]}_ker1_{kernel_size[0]}_ker2_{kernel_size[1]}_ep{epochs}_ALLMAPS"


epoch_step_to_print = 10 #the loss will be displayed every _ epochs


#load data, nothing to touch here
dataset = HeatDiffusion(input_folder, output_folder, transform=transforms.ToTensor())
n_samples = len(dataset)
if n_train < n_samples:
    n_test = n_samples-n_train
else:
    raise Exception(f"not enough samples to train on {n_train} samples \n must be less")
train_set, test_set = torch.utils.data.random_split(dataset, [n_train,n_test])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True)
x_dim, y_dim = len(dataset[0][0][0]), len(dataset[0][0])
input_size = x_dim * y_dim #total nb of pixels of one image
#first [0] select one sample, second [0] select only the input 'T0' then nb of rows and cols
output_size = input_size #we want the nn to output images of the same size as input images

def train(num_epochs = epochs, learning_rate = lr, hidden_size = hidden):

    # Convolutionnal neural network
    class ConvNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(ConvNet, self).__init__()
            self.output_size_afterConv = int(((x_dim-kernel_size[0]+1)*0.5 - kernel_size[1]+1)*0.5) #size of images after convolution and maxpooling
            self.conv1 = nn.Conv2d(1,channels[0],(kernel_size[0],kernel_size[0]))
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(channels[0],channels[1],(kernel_size[1],kernel_size[1]))
            self.fc1 = nn.Linear(channels[1]*self.output_size_afterConv**2, hidden_size) #unhardcode that
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = F.relu

        def forward(self, x):
            out = x.view(-1,1,y_dim,x_dim) #we need the one because conv layer is expected a certain nb of channels
            out = self.pool(self.act(self.conv1(out)))
            out = self.pool(self.act(self.conv2(out)))
            out = out.view(-1, channels[1]*self.output_size_afterConv**2) #unhardcode that
            out = self.act(self.fc1(out))
            out = self.fc2(out)
            # no activation and no softmax at the end
            out = out.view(-1,y_dim,x_dim)

            return out

    model = ConvNet(input_size, hidden_size, output_size).to(device)

    #loss and optimizer, scheduler, writer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if schedule:
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=sched_step, gamma=0.5)
    if tensorboard:
        writer = SummaryWriter(tensorboard_path + '/' + tensorboard_name)
        step = 0

    #training loop
    model.train()
    for epoch in range(num_epochs):
        for inputs, true_outputs in train_loader:

            inputs = inputs.to(device)
            true_outputs = true_outputs.to(device)

            #forward
            pred_outputs = model(inputs)
            loss = criterion(pred_outputs, true_outputs)

            #backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if tensorboard:
                writer.add_scalar('Training loss', loss, global_step=step)
                step +=1
        if schedule:
            scheduler.step()

        if (epoch+1) % epoch_step_to_print == 0:
            print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.6f}')

    #test
    model.eval()
    with torch.no_grad():
        sum_test_loss = 0
        for inputs, true_outputs in test_loader:
            inputs = inputs.to(device)
            true_outputs = true_outputs.to(device)
            pred_outputs = model(inputs)

            sum_test_loss += criterion(pred_outputs, true_outputs)

        glob_err = sum_test_loss/n_test
        print("///////////////////////")
        print(f'global error = {glob_err}')
        print(f"epoch = {num_epochs}")

    return model

if __name__=='__main__':
    from test_models import visual_test_model, MMSE

    model = train()

    # load test data
    model.eval()

    test_input_folder = 'data/0_003/T0testMap/iteration_no0'
    test_output_folder = 'data/0_003/T0testMap/iteration_no10'
    test_dataset = HeatDiffusion(test_input_folder, test_output_folder, transform=transforms.ToTensor())
    visual_test_model(model, test_dataset[0][0], test_dataset[0][1])

    print(f'MMSE is : {MMSE(model, 10)}')
