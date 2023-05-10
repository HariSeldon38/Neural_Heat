import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from customDataset import *
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MMSE(trained_model, iteration_no, depth=None, coef_diff=0.003, batch_size=50, multi_alpha_model=False):
    """
    This function computes the Mean of the MSE (loss fct used during training)
    on a dataset previouly created with the function T0testMap avec the correct parameters coef_diff and iteration_no
    :param trained_model: the model we want to test
    :param iteration_no: the iteration nb on wich the model has been trained
    :param depth: nb of output images for each samples (also nb of iteration of the model on each input sample)
    :param coef_diff: the diffusivity on which the model has been trained
    :param batch_size: how samples are we predicting at once
    :param multi_alpha_model: specify if model is multi trained on alpha to still use this score function
    :return: a mean of the MSE loss of each samples in data/coef_diff/T0testMap/iteration_no
    """
    trained_model.eval()
    with torch.no_grad():

        if depth: #for multi output models
            iters = list(iteration_no)
            in_folder = f"data/{str(coef_diff).replace('.', '_')}/T0testMap/iteration_no0"
            out_folders = [f"data/{str(coef_diff).replace('.', '_')}/T0testMap/iteration_no{i}" for i in iters]
            dataset_test = HeatDiffusion_multi_outputs(in_folder, out_folders, transform=transforms.ToTensor())
            dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size)
            error = 0.0
            for batch in dataloader:
                prediction = batch[0] #tmp
                for i in range(depth):
                    prediction = trained_model(prediction.to(device))
                    ground_truth = batch[1][i].to(device)
                    error += float(torch.nn.MSELoss()(prediction, ground_truth))

            return error*batch_size / (len(dataset_test)*depth) # *batchsize cause already compute mean on each batch

        in_folder = f"data/{str(coef_diff).replace('.', '_')}/T0testMap/iteration_no0"
        out_folder = f"data/{str(coef_diff).replace('.', '_')}/T0testMap/iteration_no{iteration_no}"
        dataset_test = HeatDiffusion(in_folder, out_folder)
        if multi_alpha_model:
            dataset_test = HeatDiffusion_multi_alpha(in_folder, out_folder, single_alpha=coef_diff)
        dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size)
        error = 0.0
        for batch in dataloader:
            if multi_alpha_model:
                input=[elt.to(device) for elt in batch[0]]
            else:
                input = batch[0].to(device)
            prediction = trained_model(input)
            ground_truth = batch[1].to(device)
            error += float(torch.nn.MSELoss()(prediction, ground_truth))

    return error*batch_size / len(dataset_test) # *batchsize cause already compute mean on each batch

def visual_test_model(trained_model, T_init, true_T_final):
    """
    :param trained_model: the model to be used for prediction
    :param T_init: 2dTensor(100,100)
    :param true_T_final: 2dTensor(100,100)
    :return: None
    """
    if type(true_T_final) != list: #for single output models
        true_T_final = true_T_final.numpy()

        if type(T_init) == tuple: #for multi_alpha model (T = (img, alpha))
            T_init_img = T_init[0].numpy()
            T_init = [elt.to(device) for elt in T_init]
        else:
            T_init_img = T_init.numpy()
            T_init = T_init.to(device)

        pred_T_final = trained_model(T_init).detach().to('cpu').numpy()[0]  #[0]because only one sample in the batch

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(T_init_img, cmap='hot', vmin=0.0, vmax=1.0)
        plt.subplot(2, 2, 3)
        plt.imshow(true_T_final, cmap='hot', vmin=0.0, vmax=1.0)
        plt.subplot(2, 2, 4)
        plt.imshow(pred_T_final, cmap='hot', vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.show()
        return None

    else: #for multi output models
        depth = len(true_T_final)
        true_T_final = [T.numpy() for T in true_T_final]

        if type(T_init) == tuple: #for multi_alpha model (T = (img, alpha))
            T_init_img = T_init[0].numpy()
            T_init = [elt.to(device) for elt in T_init]
        else:
            T_init_img = T_init.numpy()
            T_init = T_init.to(device)

        pred_T_final = trained_model(T_init)

        plt.figure()
        plt.subplot(3, depth, 1)
        plt.imshow(T_init_img, cmap='hot', vmin=0.0, vmax=1.0)
        for i in range(depth):
            plt.subplot(3, depth, 1+i+depth)
            plt.imshow(true_T_final[i], cmap='hot', vmin=0.0, vmax=1.0)
            plt.subplot(3, depth, 1+i+2*depth)
            plt.imshow(pred_T_final.detach().to('cpu').numpy()[0], cmap='hot', vmin=0.0, vmax=1.0)
            pred_T_final = trained_model(pred_T_final)
        plt.show()
        return None

if __name__=='__main__':
    from training_scripts import feedfwd_heat

    model = feedfwd_heat.train()
    print(f'feedfwd model has a MMSE of : {MMSE(model,10)}')    #still need to address the iteration issue to make it autom



