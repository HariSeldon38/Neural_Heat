"""

This is the same architecture model than feedforward_heat, except based on another database form
(use HeatDiffusion2)
this is used to test basic multitask learning where alpha is just another input among the 10 000 pixel already there

"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from customDataset import HeatDiffusion2
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
input_folder = 'data/T0blocksMap/iteration_no0/'
output_folder = 'data/T0blocksMap/iteration_no10/'
dataset = HeatDiffusion2(input_folder, output_folder, transform=transforms.ToTensor())

batch_size = 20 #for instance
train_set, test_set = torch.utils.data.random_split(dataset, [140,20])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True)

# hyper parameters
input_size = 100 * 100 + 1 #the alpha : diffusion coeficient
output_size = 100 * 100

example = False
if example:
    examples = iter(train_loader)
    for i, T in enumerate(examples):
        plt.figure()
        T0, T1 = T #T0 and T1 are representing a batch (20 images each)
        plt.subplot(1,2,1)
        plt.imshow(T0[0][0][0], cmap='hot')   #we just want a single image from the batch
        plt.subplot(1, 2, 2)                 # the second [0] is because T0[0] is in fact [T0[0]] but idk why
        plt.imshow(T1[0][0], cmap='hot')
        break #I don't know out to use the iter objet in a better way so that for loop will do sorry

def train(num_epochs = 130, lr = 0.0001, hidden_size = 1000):

    # Fully connected neural network with one hidden layer
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNet, self).__init__()
            self.input_size = input_size
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.to(torch.float32)
            out = nn.ReLU()(self.l1(x))
            out = self.l2(out)
            # no activation and no softmax at the end
            return out

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    #loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.2)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=6)

    #training loop
    model.train()
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (inputs, true_outputs) in enumerate(train_loader):
            image, alpha = inputs
            image = image.reshape(-1, 100*100).to(device)
            alpha = alpha.view(batch_size, 1).to(device)
            inputs = torch.concatenate((image,alpha),1)
            true_outputs = true_outputs.reshape(-1, 100*100).to(device)

            #forward
            pred_outputs = model(inputs)
            loss = criterion(pred_outputs, true_outputs)

            #backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(metrics=loss)

        if (epoch+1) % 10 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.6f}')

    #test
    model.eval()
    with torch.no_grad():
        n_samples = 0
        sum_test_loss = 0
        for inputs, true_outputs in test_loader:
            image, alpha = inputs
            image = image.reshape(-1, 100*100).to(device)
            alpha = alpha.view(batch_size, 1).to(device)
            inputs = torch.concatenate((image,alpha),1)

            true_outputs = true_outputs.reshape(-1, 100*100).to(device)
            pred_outputs = model(inputs)

            sum_test_loss += criterion(pred_outputs, true_outputs)
            n_samples += true_outputs.shape[0]

        precision = sum_test_loss/n_samples
        print("///////////////////////")
        print(f'precision = {precision}')
        print(f"epoch = {num_epochs}")

    return model

def test_model(model, T_init, true_T_final, fake_alpha=None):
    image, alpha = T_init
    image = image.reshape(-1, 100 * 100).to(device)
    if fake_alpha:

        alpha = fake_alpha
    alpha = torch.tensor(alpha).view(1, 1).to(device)
    T_init = torch.concatenate((image, alpha), 1)

    true_T_final = true_T_final.numpy()

    pred_T_final = model(T_init.reshape(-1, input_size).to(device)).reshape(100,100).detach().to('cpu').numpy()

    image = image.reshape(100,100).to('cpu').numpy()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='hot', vmin=0.0, vmax=1.0)
    plt.subplot(2, 2, 2)
    plt.imshow(true_T_final[0], cmap='hot', vmin=0.0, vmax=1.0)
    plt.subplot(2, 2, 3)
    plt.imshow(pred_T_final, cmap='hot', vmin=0.0, vmax=1.0)
    plt.show()

model = train()

#load test data
test_input_folder = 'data/T0blocksMap/iteration_no0/'
test_output_folder = 'data/T0blocksMap/iteration_no10/'
test_dataset = HeatDiffusion2(test_input_folder, test_output_folder, transform=transforms.ToTensor())


test_model(model, test_dataset[0][0], test_dataset[0][1]) #high diff coef
test_model(model, test_dataset[50][0], test_dataset[50][1]) #low diff coef


