#Loss : sepearte it in two
#maybe try for myself to add k_fold cross validation for later
#monitor the loss (with the website i dont remember the name

#access to tensorboard:
#cmd : tensorboard --logdir="C:\Users\gaia3\AppData\Local\Programs\Python\PythonProjects\Neural_Heat/runs
#web = http://localhost:6006/

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
from customDataset import HeatDiffusion
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
input_folder = 'data/0_003/T0blocksMap_0_003/iteration_no0/'
output_folder = 'data/0_003/T0blocksMap_0_003/iteration_no10/'
dataset = HeatDiffusion(input_folder, output_folder, transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [200,100])

batch_sizes = [1, 10, 50, 100, 200]
learning_rates = [0.001, 0.0001, 0.0005, 0.00005]

for batch in batch_sizes:
    for lr in learning_rates:

        train_loader = DataLoader(dataset=train_set, batch_size=batch, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch,shuffle=True)

        # hyper parameters
        input_size = 100 * 100
        output_size = 100 * 100

        def train(num_epochs = 60, lr = lr, hidden_size = 1000):

            # Fully connected neural network with one hidden layer
            class NeuralNet(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(NeuralNet, self).__init__()
                    self.input_size = input_size
                    self.l1 = nn.Linear(input_size, hidden_size)
                    self.l2 = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    out = nn.ReLU()(self.l1(x))
                    out = self.l2(out)
                    # no activation and no softmax at the end
                    return out

            model = NeuralNet(input_size, hidden_size, output_size).to(device)

            #loss and optimizer, scheduler, writer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)
            writer = SummaryWriter(f"runs/hypersearch2/MiniBatchSIze {batch} LR {lr}")

            #training loop
            model.train()
            n_total_steps = len(train_loader)
            step = 0
            for epoch in range(num_epochs):
                for i, (inputs, true_outputs) in enumerate(train_loader):

                    inputs = inputs.reshape(-1, 100*100).to(device)
                    true_outputs = true_outputs.reshape(-1, 100*100).to(device)

                    #forward
                    pred_outputs = model(inputs)
                    loss = criterion(pred_outputs, true_outputs)

                    #backwards
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar('Training loss', loss, global_step=step)
                    step +=1
                scheduler.step(metrics=loss)

                if (epoch+1) % 10 == 0:
                    print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.6f}')

            #test
            model.eval()
            with torch.no_grad():
                n_samples = 0
                sum_test_loss = 0
                for inputs, true_outputs in test_loader:
                    inputs = inputs.reshape(-1, 100*100).to(device)
                    true_outputs = true_outputs.reshape(-1, 100*100).to(device)
                    pred_outputs = model(inputs)

                    sum_test_loss += criterion(pred_outputs, true_outputs)
                    n_samples += true_outputs.shape[0]

                precision = sum_test_loss/n_samples
                print("///////////////////////")
                print(f'precision = {precision}')
                print(f"epoch = {num_epochs}")

            writer.add_scalar('precision*1000', precision*1000)

            return model

        model = train()
#load test data
test_input_folder = 'data/0_003/T0blocksMapV2_0_003/iteration_no0/'
test_output_folder = 'data/0_003/T0blocksMapV2_0_003/iteration_no10/'
test_dataset = HeatDiffusion(test_input_folder, test_output_folder, transform=transforms.ToTensor())
#test_model(model, test_dataset[0][0], test_dataset[0][1])
#test_model(model, test_dataset[1][0], test_dataset[1][1])


"""
Pyrorch common mistakes

1. you should take a single sample then a single batch and try overfitting it
before epoch :
data, targets = next(iter(train_loader)) #with batch size = 1 then normal
and comment the for enumerate(train_loader)
--> very low loss : ok the model is overfitting sanity check

2. Forgot toggle train/eval
model.eval() #when check accuracy and computing predictions in use case
model.train() #when in training (with dropout and batchnorm... etc)

3. Forgot .zero_grad()
decrease accuracy a lot
optimizer.zero_grad() then loss.backward() then optimizer.step()

4. Softmax when using CrossEntropy (it decrease a little bit the accuracy)

5. Bias term with BatchNorm
self.bn1 = nn.BatchNorm2d(out_channel_previous)
and then
x = F.relu(self.bn1(self.conv1(x)))
set bias=False in the conv layer (or the linear if batchnorm after a linear)
the bias is unnecessary

6. Using view as permute (not smth i would do)
transposition : x.permute(1,0)

7. Incorrect Data Augmentation

8. Not Shuffling Data

9. Not Normalizing Data
we want data to have mean=0 and std=1
ToTensor divide everything by 255
after ToTensor(), transforms.Normalize(mean=(...,) , std=(...,))
less important with batchnorm

10. Not Clipping Gradients for RNNs, GRUs, LSTMs
after loss.backward()
do 
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

also : 
Getting confused with tensor dimensions (as a new guy you can spend plenty of time before harnessing the power of unsqueeze())
Forgetting .cuda() or .to(device)
Getting confused with convnet dimensions after conv layer is applied
Not attempting to balance or disbalance the dataset on purpose, which can be useful

For Dimensions after applying Conv layer
you can use this formula [(W−K+2P)/S]+1  
W = Input Width
K = Kernel size
P = Padding
S = Stride

Einsum :
https://www.youtube.com/watch?v=pkVwUVEHmfI
"""