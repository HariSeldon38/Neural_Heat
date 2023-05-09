"""
this is a res net based on a Convolution layer
the resnet is trained at each depth
--> means conv layer (each time, the same one)
input --> out1 --> out2 --> ... --> out_DEPTH
           |        |        |       |
        true_1    true2     ...    true_DEPTH   >> compute and sum each loss, backpropagat
                                                    to optimize params of the layer

by using only one layer invariant as well as depth is concerned
I hope the model will learn how to do a single iteration (of a fixed nb) of my heat transfert pb
that way, when the model is trained, I can apply it how many time I want to access the iteration I want
if it works it is a kind of multitasking cause I can do the same with any parameter instead of iteration
     with the condition that the parameter can be sorted !!!

Thus, when testing the precision of the model, we want that each iteration of the model compete
with an other model with only one output.
that way, we will compute the MSE on each test samples and each iteration regardless and finally do the mean of them all

The function MMSE could also be adapted in this direction (parameter iter accepting list)
"""


import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from tqdm import tqdm
from customDataset import HeatDiffusion_multi_outputs, HeatDiffusion
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_classes import ConvNet
#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning, cuda is not available")

#Physics
alpha = 0.003

#hyperparameters
epochs = 1#50 #nb of epochs to do when training
n_train = 1000
batch_size = 20
lr = 0.003 #learning rate
DEPTH = 10
kernel_size = 3
channels = 1
input_folder = f"data/{str(alpha).replace('.','_')}/ALLMAPS/iteration_no0"
output_folders = [f"data/{str(alpha).replace('.','_')}/ALLMAPS/iteration_no{i+1}" for i in range(DEPTH)]
schedule = False
sched_step = [] #epoch steps when we want to decrease the lr by a factor 2


epoch_step_to_print = 10 #the loss will be displayed every _ epochs


tensorboard = False
# access to tensorboard:
# cmd : tensorboard --logdir="C:\Users\gaia3\AppData\Local\Programs\Python\PythonProjects\Neural_Heat/runs/feedfwd/
# web = http://localhost:6006/
tensorboard_path = 'runs/RNN_multi_out/1'
tensorboard_name = f'trainingsamples{n_train}_batchsize{batch_size}_lr{lr}'
save_path =  f"trained_models/{str(alpha).replace('.','_')}_{n_train}trainsamples_depth={DEPTH}_kernel={kernel_size}_epoch={epochs}_RCNN"

#load data, nothing to touch here
dataset = HeatDiffusion_multi_outputs(input_folder, output_folders, transform=transforms.ToTensor())
n_samples = len(dataset)
if n_train < n_samples:
    n_test = n_samples - n_train
else:
    raise Exception(f"not enough samples to train on {n_train} samples \n must be less")
train_set, test_set = torch.utils.data.random_split(dataset, [n_train,n_test])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True)
x_dim, y_dim = len(dataset[0][0][0]), len(dataset[0][0])
#first [0] select one sample, second [0] select only the input 'T0' then nb of rows and cols
input_size = x_dim * y_dim
output_size = input_size #we want the nn to output images of the same size as input images

model = ConvNet(channels, kernel_size).to(device)

def train(num_epochs = epochs, learning_rate = lr, depth=DEPTH):

    #loss and optimizer, scheduler, writer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / depth)
    if schedule:
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=sched_step, gamma=0.5)
    if tensorboard:
        writer = SummaryWriter(tensorboard_path + '/' + tensorboard_name)
        step = 0

    #training loop
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for input_images, output_images in train_loader:

            inputs = input_images.to(device)
            true_outputs = [images.to(device) for images in output_images]

            #forward
            pred_outputs = inputs #tmp
            optimizer.zero_grad()
            for d in range(depth-1):
                pred_outputs = model(pred_outputs)
                loss = criterion(pred_outputs, true_outputs[d])
                loss.backward(retain_graph=True)
            pred_outputs = model(pred_outputs) #the following lines for the last iteration of depth is out of the loop
            loss = criterion(pred_outputs, true_outputs[depth-1]) #in order to not retain graph in the end

            #backward
            loss.backward()
            optimizer.step()
            if tensorboard:
                writer.add_scalar('Training loss', loss, global_step=step)
                step +=1
        if schedule:
            scheduler.step()

        if (epoch+1) % epoch_step_to_print == 0:
            print(f'\nepoch {epoch+1} / {num_epochs}, loss = {loss.item():.6f}')

    return model


if __name__=='__main__':
    from test_models import visual_test_model, MMSE

    model = train()

    #load test data
    model.eval()

    test_input_folder = 'data/0_003/T0testMap/iteration_no0'
    test_output_folders = [f'data/0_003/T0testMap/iteration_no{i+1}' for i in range(DEPTH)]
    test_dataset = HeatDiffusion_multi_outputs(test_input_folder, test_output_folders, transform=transforms.ToTensor())
    visual_test_model(model, test_dataset[0][0], test_dataset[0][1])
    print(f'MMSE is : {MMSE(model, [i for i in range(1,11)], depth=10)}')

    def save_model(save):
        torch.save(model.state_dict(),save_path + '_' + str(save))

    """load model
    loaded_model = ConvNet(channels, kernel_size).to(device)
    loaded_model.load_state_dict(torch.load(save_path + '_' + str(save)))
    loaded_model.eval()
    
    #see parameters:
    layers=[x.data for x in loaded_model.parameters()]
    """
