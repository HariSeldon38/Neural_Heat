import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from customDataset import HeatDiffusion
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MMSE(trained_model, iteration_no, coef_diff=0.003, batch_size=50):
    """
    :param trained_model:
    :param iteration_no:
    :param coef_diff:
    :param batch_size:
    :return:
    """
    trained_model.eval()
    with torch.no_grad():
        in_folder = f"data/{str(coef_diff).replace('.','_')}/T0testMap/iteration_no0"
        out_folder = f"data/{str(coef_diff).replace('.','_')}/T0testMap/iteration_no{iteration_no}"
        dataset_test = HeatDiffusion(in_folder, out_folder, transform=transforms.ToTensor())
        dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size)
        error = 0.0
        for batch in dataloader:
            prediction = trained_model(batch[0].to(device))
            ground_truth = batch[1].to(device)
            for i in range(batch_size):
                error += float(torch.nn.MSELoss()(prediction[i],ground_truth[i]))
    return error/len(dataset_test)

def visual_test_model(model, T_init, true_T_final):
    """

    :param model:
    :param T_init: 2dTensor(100,100)
    :param true_T_final: 2dTensor(100,100)
    :return: None
    """
    true_T_final = true_T_final.numpy()

    pred_T_final = model(T_init.to(device)).detach().to('cpu').numpy()[0] #[0] because only one sample in the batch

    T_init = T_init.numpy()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(T_init, cmap='hot', vmin=0.0, vmax=1.0)
    plt.subplot(2, 2, 3)
    plt.imshow(true_T_final, cmap='hot', vmin=0.0, vmax=1.0)
    plt.subplot(2, 2, 4)
    plt.imshow(pred_T_final, cmap='hot', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.show()
    return None


if __name__=='__main__':
    import feedfwd_heat

    model = feedfwd_heat.train()
    print(f'feedfwd model has a MMSE of : {MMSE(model,10)}')    #still need to address the iteration issue to make it autom






