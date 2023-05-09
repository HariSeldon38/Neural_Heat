import torch
import torch.nn as nn
import torchvision.transforms as transforms
from customDataset import HeatDiffusion
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from model_classes import ConvNet
#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning, cuda is not available")


#Physics
alpha = 0.003
final_iteration = 10

#hyperparameters
epochs = 25
n_train = 300
batch_size = 20
lr = 0.005 #learning rate
channels = 1 #nb output channel in the con layer
kernel_size = 5 #kernel size for the conv layer
input_folder = f"data/{str(alpha).replace('.','_')}/2,5k_samples/iteration_no0"
output_folder = f"data/{str(alpha).replace('.','_')}/2,5k_samples/iteration_no{final_iteration}"
schedule = False
sched_step = [15] #epoch steps when we want to decrease the lr by a factor 2



tensorboard = False
#access to tensorboard:
#cmd : tensorboard --logdir="C:\Users\gaia3\AppData\Local\Programs\Python\PythonProjects\Neural_Heat/runs/CNN/1/
#web = http://localhost:6006/
tensorboard_path = 'runs/CNN/1'
tensorboard_name = f"trainingsamples{n_train}_batch{batch_size}_lr{lr}_chan_{channels}_ker_{kernel_size}_ep{epochs}"
save_path =  f"models/{str(alpha).replace('.','_')}_{final_iteration}_{n_train}trainsamples_CNN"


epoch_step_to_print = 5 #the loss will be displayed every _ epochs


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

model = ConvNet(channels, kernel_size).to(device)

def train(num_epochs = epochs, learning_rate = lr,):

    #loss and optimizer, scheduler, writer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if schedule:
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=sched_step, gamma=0.5)
    if tensorboard:
        writer = SummaryWriter(tensorboard_path + '/' + tensorboard_name)
        step = 0
    print("\n")
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

    def save_model(save):
        torch.save(model.state_dict(), save_path + '_' + str(save))

    """LOAD MODEL
    loaded_model = ConvNet(channels, kernel_size).to(device)
    loaded_model.load_state_dict(torch.load(save_path + '_' + str(save)))
    loaded_model.eval()
    
    #see parameters:
    layers=[x.data for x in loaded_model.parameters()]
    """