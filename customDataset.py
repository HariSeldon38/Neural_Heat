import os
from torch.utils.data import Dataset
from skimage import io
from numpy import squeeze #it remove [ ] around single values in a matrix

class HeatDiffusion(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform

    def __len__(self):
        nb_input = os.listdir(self.input_dir)
        nb_output = os.listdir(self.output_dir)
        if nb_input == nb_output:
            return len(nb_input)
        else:
            print(f"WARNING : there is {nb_output} outputs (labels) for {nb_input} inputs.")

    def __getitem__(self, index):
        input_image = squeeze(io.imread(self.input_dir+f'{str(index)}.png')[:,:,0]) #we only want one of the RGB channel
        output_image = squeeze(io.imread(self.output_dir + f'{str(index)}.png')[:,:,0]) #as the image is grayscaled

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

class HeatDiffusion2(Dataset):
    """this is a class to load a dataset created with a different directory tree than upper"""
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform

    def __len__(self):
        nb_input = os.listdir(self.input_dir)
        nb_output = os.listdir(self.output_dir)
        if nb_input == nb_output:
            return len(nb_input)
        else:
            print(f"WARNING : there is {nb_output} outputs (labels) for {nb_input} inputs.")

    def __getitem__(self, index):
        alpha = ((index//20)+1)*0.0005
        name = f"{str(alpha).replace('.','')}_{index%20}.png"
        input_image = squeeze(io.imread(self.input_dir + name)[:,:,0]) #we only want one of the RGB channel
        output_image = squeeze(io.imread(self.output_dir + name)[:,:,0]) #as the image is grayscaled

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return (input_image, alpha), output_image

class HeatDiffusion3(Dataset):
    """this class is used to load a dataset with multiple output (nb of iterations)"""
    def __init__(self, input_dir, output_dirs, transform=None):
        self.input_dir = input_dir
        self.output_dirs = output_dirs
        self.transform = transform

    def __len__(self):
        nb_input = os.listdir(self.input_dir)
        nb_output = os.listdir(self.output_dirs[0])
        if nb_input == nb_output:
            return len(nb_input)
        else:
            print(f"WARNING : there is {nb_output} outputs (labels) for {nb_input} inputs.")

    def __getitem__(self, index):
        input_image = squeeze(io.imread(self.input_dir+f'{str(index)}.png')[:,:,0]) #we only want one of the RGB channel


        output_image1 = squeeze(io.imread(self.output_dirs[0] + f'{str(index)}.png')[:,:,0])
        output_image2 = squeeze(io.imread(self.output_dirs[1] + f'{str(index)}.png')[:,:,0])
        output_image3 = squeeze(io.imread(self.output_dirs[2] + f'{str(index)}.png')[:,:,0])
        output_image4 = squeeze(io.imread(self.output_dirs[3] + f'{str(index)}.png')[:,:,0])
        output_image5 = squeeze(io.imread(self.output_dirs[4] + f'{str(index)}.png')[:,:,0])
        output_image6 = squeeze(io.imread(self.output_dirs[5] + f'{str(index)}.png')[:,:,0])
        output_image7 = squeeze(io.imread(self.output_dirs[6] + f'{str(index)}.png')[:,:,0])
        output_image8 = squeeze(io.imread(self.output_dirs[7] + f'{str(index)}.png')[:,:,0])
        output_image9 = squeeze(io.imread(self.output_dirs[8] + f'{str(index)}.png')[:,:,0])
        output_image10 = squeeze(io.imread(self.output_dirs[9] + f'{str(index)}.png')[:,:,0])

        if self.transform:
            input_image = self.transform(input_image)
            output_image1 = self.transform(output_image1)
            output_image2 = self.transform(output_image2)
            output_image3 = self.transform(output_image3)
            output_image4 = self.transform(output_image4)
            output_image5 = self.transform(output_image5)
            output_image6 = self.transform(output_image6)
            output_image7 = self.transform(output_image7)
            output_image8 = self.transform(output_image8)
            output_image9 = self.transform(output_image9)
            output_image10 = self.transform(output_image10)

        return input_image, output_image1, output_image2, output_image3, output_image4, output_image5, output_image6, output_image7, output_image8, output_image9, output_image10




""" HOW TO USE
from customDataset import HeatDiffusion
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#load data
input_folder = 'data/T0blocksMap_0_003/iteration_no0'
output_folder = 'data/T0blocksMap_0_003/iteration_no10'
dataset = HeatDiffusion(input_folder, output_folder, transform=transforms.ToTensor())

batch_size = 20 #for instance
train_set, test_set = torch.utils.data.random_split(dataset, [200,100])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True)
"""





