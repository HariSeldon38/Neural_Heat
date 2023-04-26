#Loss : sepearte it in two
#maybe try for myself to add k_fold cross validation for later
#see result
#monitor the loss with tensor board and find best hyperparam
# use lr scheduling
#use batch norm
# model.train() and .eval()
#

#access to tensorboard:
#cmd : tensorboard --logdir="C:\Users\gaia3\AppData\Local\Programs\Python\PythonProjects\Neural_Heat/runs/rnn_search/search2
#web = http://localhost:6006/

#pipreq pr requirements

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from customDataset import HeatDiffusion3, HeatDiffusion
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
input_folder = 'data/0_003/T0blocksMapV2_0_003(1)/iteration_no0/'
output_folders = ['data/0_003/T0blocksMapV2_0_003(1)/iteration_no1/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no2/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no3/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no4/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no5/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no6/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no7/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no8/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no9/',
                  'data/0_003/T0blocksMapV2_0_003(1)/iteration_no10/']

dataset = HeatDiffusion3(input_folder, output_folders, transform=transforms.ToTensor())

batch_size = 20 #for instance
train_set, test_set = torch.utils.data.random_split(dataset, [200,100])      #need to unardcode that if changing dataset size
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True)

# hyper parameters
input_size = 100 * 100
output_size = 100 * 100

example = False
if example:
    examples = iter(train_loader)
    for i, T in enumerate(examples):
        plt.figure()
        T0, T1 = T #T0 and T1 are representing a batch (20 images each)
        plt.subplot(1,2,1)
        plt.imshow(T0[0][0], cmap='hot')   #we just want a single image from the batch
        plt.subplot(1, 2, 2)                 # the second [0] is because T0[0] is in fact [T0[0]] but idk why
        plt.imshow(T1[0][0], cmap='hot')
        break #I don't know out to use the iter objet in a better way so that for loop will do sorry


write = "basic_rnn_depth10_lr0_00043_sched60_batch20_hidden5000"
def train(num_epochs = 90, lr = 0.00043, hidden_size = 5000, depth=10):

    # Fully connected neural network with one hidden layer
    class ConvNet(nn.Module):
        def __init__(self, hidden_size=hidden_size, output_size=output_size):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1,6,5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,9,5)
            self.fc1 = nn.Linear(100*100, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = F.relu

        def forward(self, x):
            #out = self.pool(self.act(self.conv1(x)))
            #out = self.pool(self.act(self.conv2(out)))
            out = x.view(-1, 100*100)
            out = self.act(self.fc1(out))
            out = self.fc2(out)
            # no activation and no softmax at the end
            return out.view(-1,1,100,100)

    model = ConvNet(hidden_size, output_size).to(device)

    #loss and optimizer, scheduler, writer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr/depth)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    if write:
        writer = SummaryWriter(f"runs/rnn_search/search2/"+write)

    #training loop
    model.train()
    n_total_steps = len(train_loader)
    step = 0
    for epoch in range(num_epochs):
        cummul_loss = 0
        for i, (input_image, output_image1, output_image2, output_image3, output_image4, output_image5, output_image6, output_image7, output_image8, output_image9, output_image10) in enumerate(train_loader):
            inputs = input_image.to(device)

            true_outputs = [output_image1.view(-1, 1, 100, 100).to(device),
                            output_image2.view(-1, 1, 100, 100).to(device),
                            output_image3.view(-1, 1, 100, 100).to(device),
                            output_image4.view(-1, 1, 100, 100).to(device),
                            output_image5.view(-1, 1, 100, 100).to(device),
                            output_image6.view(-1, 1, 100, 100).to(device),
                            output_image7.view(-1, 1, 100, 100).to(device),
                            output_image8.view(-1, 1, 100, 100).to(device),
                            output_image9.view(-1, 1, 100, 100).to(device),
                            output_image10.view(-1, 1, 100, 100).to(device)]

            #forward
            pred_outputs = inputs
            optimizer.zero_grad()
            for d in range(depth-1):
                pred_outputs = model(pred_outputs)
                loss = criterion(pred_outputs, true_outputs[d])
                cummul_loss += loss
                loss.backward(retain_graph=True)

            pred_outputs = model(pred_outputs) #the following lines for the last iteration of depth is out of the loop
            loss = criterion(pred_outputs, true_outputs[depth-1]) #in order to not retain indefinetly the graph
            cummul_loss += loss
            loss.backward()

            optimizer.step()
        if epoch<65:
            scheduler.step()

        if write:
            cummul_loss /= 200//batch_size
            writer.add_scalar('Training loss', cummul_loss, global_step=step)
            step += 1

        if (epoch+1) % 1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.6f}')

    #test
    """
    with torch.no_grad():
        n_samples = 0
        sum_test_loss = 0
        for inputs, true_outputs in test_loader:
            inputs = inputs.to(device)
            true_outputs = true_outputs.view(-1,100,100).to(device)
            pred_outputs = model(inputs)

            sum_test_loss += criterion(pred_outputs, true_outputs)
            n_samples += true_outputs.shape[0]

        precision = sum_test_loss/n_samples
        print("///////////////////////")
        print(f'precision = {precision}')
        print(f"epoch = {num_epochs}")
    """
    return model


model = train()


def test_model(model, T_init, true_T_final):
    true_T_final = true_T_final.numpy()

    machin = T_init.reshape(1,-1,100,100).to(device)
    for d in range(10):
        machin = model(machin)
    pred_T_final = machin.reshape(100,100).detach().to('cpu').numpy()

    T_init = T_init.numpy()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(T_init[0], cmap='hot', vmin=0.0, vmax=1.0)
    plt.subplot(2, 2, 2)
    plt.imshow(true_T_final[0], cmap='hot', vmin=0.0, vmax=1.0)
    plt.subplot(2, 2, 3)
    plt.imshow(pred_T_final, cmap='hot', vmin=0.0, vmax=1.0)
    plt.show()

#load test data
model.eval()

test_input_folder = 'data/0_003/T0blocksMap_0_003/iteration_no0/'
test_output_folder = 'data/0_003/T0blocksMap_0_003/iteration_no10/'
test_dataset = HeatDiffusion(test_input_folder, test_output_folder, transform=transforms.ToTensor())

test_input_folder = 'data/0_003/T0blocksMapV2_0_003/iteration_no0/'
test_output_folder = 'data/0_003/T0blocksMapV2_0_003/iteration_no10/'
test_datasetV2 = HeatDiffusion(test_input_folder, test_output_folder, transform=transforms.ToTensor())

test_input_folder = 'data/0_003/T0sinusMap_0_003/iteration_no0/'
test_output_folder = 'data/0_003/T0sinusMap_0_003/iteration_no10/'
test_dataset_sinus = HeatDiffusion(test_input_folder, test_output_folder, transform=transforms.ToTensor())

test_model(model, test_dataset[0][0], test_dataset[0][1])
test_model(model, test_datasetV2[1][0], test_datasetV2[1][1])
test_model(model, test_dataset_sinus[3][0], test_dataset_sinus[3][1])









"""
def classify(folder_name, model, negatif=True):
    import cv2
    from numpy import asarray
    import os

    image_names = os.listdir("data/"+folder_name)
    for elt in image_names:
        image = cv2.imread("data/"+folder_name+"/"+elt)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if negatif: #by default, model trained with digit in white on a black canvas
            image = list(asarray(image))
        else: #if digit in black above white
            image = list(asarray(~image))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.reshape(-1, 28 * 28).to('cuda')
        with torch.no_grad():
            _, output = torch.max(model(image), 1)
            print(f'file : {elt}, pred_digit = {output[0]}')



def short(folder_name, negatif=True):
    classify(folder_name, train(), negatif)

"""