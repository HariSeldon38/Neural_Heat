#petite modélisation rapide du phénomène de diffusion de la chaleur
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Training set, input generation
def T0blocksMap(ny,nx):
    matrix = np.zeros((ny, nx))

    # Divide the matrix into 16 squares
    squares = np.split(matrix, 4, axis=0)
    squares = [np.split(square, 4, axis=1) for square in squares]

    # Randomly assign values to each square
    for row in range(4):
        for col in range(4):
            value = np.random.randint(2)
            squares[row][col][:, :] = value

    # Flatten the list of squares and return the matrix
    matrix = np.concatenate([np.concatenate(row, axis=1) for row in squares], axis=0)

    return matrix
# A rather simple test set input generation
# (a little bit different from the training set input generation but not so much)
def T0blocksMapV2(ny,nx):
    matrix = np.zeros((ny, nx))

    # Randomly choose split points for rows and columns
    row_split_points = np.sort(np.random.choice(range(1, ny), size=3, replace=False))
    col_split_points = np.sort(np.random.choice(range(1, nx), size=3, replace=False))

    # Divide the matrix into squares based on split points
    squares = np.split(matrix, row_split_points, axis=0)
    squares = [np.split(square, col_split_points, axis=1) for square in squares]

    # Randomly assign values to each square
    for row in range(4):
        for col in range(4):
            value = np.random.randint(2)
            squares[row][col][:, :] = value

    # Flatten the list of squares and return the matrix
    matrix = np.concatenate([np.concatenate(row, axis=1) for row in squares], axis=0)
    return matrix
# A more difficult one
def T0sinusMap(ny,nx):
    x, y = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx))
    a = np.random.normal()
    b = np.random.normal()
    matrix = np.sin(a*2*np.pi*x) * np.sin(b*2*np.pi*y) + np.random.normal(scale=0.05, size=(100, 100))
    matrix = np.clip(matrix, 0, 1)
    return matrix
#modelisation

def iterate(T_matrix, iterations, dt, dx, alpha):
    """
    Discretisation of the heat equation in 2D :
    Temperature = T(x,y,t)
    diff(T, t) = alpha * [ diff2(T, x) + diff2(T, y)
    --> T(t+dt) = T(t) + [alpha*dt/(dx)**2] * [ T(x+dx)+T(x-dx) + T(y+dy)+T(y-dy) - 4*T(x,y) ]

    :param T_matrix: temperature map at t initial (t_i)
    :param iterations: number of steps to evolve the temperature map
    :param dt : length of one iteration
    :param dx : size of the spatial discretization (dy is the same)
    :param alpha : diffusion coeficient

    :return: the new temperature map at t = t_i + iterations*dt
    """

    T = T_matrix.copy()
    ny,nx = T.shape #convention x is number of columns to fit with a basic plot (x,y)  (in fact -y cause matrices display)
    coef = alpha * dt / (dx * dx)
    for _ in range(iterations):
        for i in range(1,ny-1):
            for j in range(1,nx-1):
                T[i,j] = T[i,j] + coef * (T[i+1,j]+T[i-1,j]+T[i,j+1]+T[i,j-1]-4*T[i,j])
            # Edges
            T[i, 0] = T[i,0] + coef * (T[i+1,0]+T[i-1,0]+T[i,1]+T[i,0]-4*T[i,0])
            T[i, nx-1] = T[i,nx-1] + coef * (T[i+1,nx-1]+T[i-1,nx-1]+T[i,nx-1]+T[i,nx-2]-4*T[i,nx-1])
        for j in range(1,nx-1):
            T[0, j] = T[0,j] + coef * (T[1,j]+T[0,j]+T[0,j+1]+T[0,j-1]-4*T[0,j])
            T[ny-1, j] = T[ny-1,j] + coef * (T[ny-1,j]+T[ny-2,j]+T[ny-1,j+1]+T[ny-1,j-1]-4*T[ny-1,j])

        # Corners
        T[0, 0] = T[0,0] + coef * (T[1,0]+T[0,0]+T[0,1]+T[0,0]-4*T[0,0])
        T[0, nx-1] = T[0,nx-1] + coef * (T[1,nx-1]+T[0,nx-1]+T[0,nx-1]+T[0,nx-2]-4*T[0,nx-1])
        T[ny-1, 0] = T[ny-1,0] + coef * (T[ny-1,0]+T[ny-2,0]+T[ny-1,1]+T[ny-1,0]-4*T[ny-1,0])
        T[ny-1, nx-1] = T[ny-1,nx-1] + coef * (T[ny-1,nx-1]+T[ny-2,nx-1]+T[ny-1,nx-1]+T[ny-1,nx-2]-4*T[ny-1,nx-1])

    return T

test_diffusion = False
if __name__=="__main__" and test_diffusion:
    # Define parameters
    n = 100 # Number of grid points in each direction

    # all the following is just a scaling of the coef of diffusion nothing more
    # the discretization time however appears in the diffusion coef too but also on the nb of iteration (precision)
    size = 1 #in meter
    dx = size/n #spatial discretization step in meter
    alpha = 0.003 # Diffusion coefficient

    t1 = 1.0 # Total time
    dt = 0.01 # Time discretization step in sec
    N = int(t1 / dt) # Number of time steps

    # Initialize grid
    u0 = T0blocksMapV2(n,n)

    #make 3 matrices of diffusion.
    u1 = iterate(u0, N, dt, dx, alpha)
    u2 = iterate(u1, 5*N, dt, dx, alpha)
    u3 = iterate(u2, 10*N, dt, dx, alpha)


    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(u0, cmap='hot', origin='upper', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(u1, cmap='hot', origin='upper', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(u2, cmap='hot', origin='upper', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(u3, cmap='hot', origin='upper', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.show() #warning color scale change for each plot so do not wrongly analyse

def generate_data(samples, T0_func, coef_diffusion=0.003, iters=(10,40,50,450), n=100, dx=0.01, dt=0.01):
    #could change iters into times but create an error intrinsic to the modelisation
    #not good to check performances of the nn later, or we should quantizize this error
    import os
    data_folder = f"data/{str(coef_diffusion).replace('.','_')}/{T0_func.__name__}_{str(coef_diffusion).replace('.','_')}"
    while os.path.exists(data_folder):
        data_folder += "(1)"
    os.makedirs(data_folder)

    cummulative_iters = [0] + list(iters)
    os.makedirs(data_folder + f'/iteration_no{str(0)}')
    for i in range(1,len(cummulative_iters)):
        cummulative_iters[i]+=cummulative_iters[i-1]
        os.makedirs(data_folder+f'/iteration_no{str(cummulative_iters[i])}')

    for s in tqdm(range(samples)):

        T = T0_func(n,n)
        current_iter = 0
        plt.imsave(data_folder + f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')
        for it in iters:
            T = iterate(T, it, dt, dx, coef_diffusion)
            current_iter += it
            plt.imsave(data_folder+f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')

def generate_data2(samples, coef_diffusion=0.003, iters=(10,40), n=100, dx=0.01, dt=0.01):
    #this fct if to be used with the endeavour at multitasking alpha
    #could change iters into times but create an error intrinsic to the modelisation
    #not good to check performances of the nn later, or we should quantizize this error
    import os
    data_folder = f"data/{str(coef_diffusion).replace('.','_')}/ALLMAPS"
    while os.path.exists(data_folder):
        data_folder += "(1)"
    os.makedirs(data_folder)

    cummulative_iters = [0] + list(iters)
    os.makedirs(data_folder + f'/iteration_no{str(0)}')
    for i in range(1,len(cummulative_iters)):
        cummulative_iters[i]+=cummulative_iters[i-1]
        os.makedirs(data_folder+f'/iteration_no{str(cummulative_iters[i])}')

    MAPS = [T0blocksMapV2, T0sinusMap]
    for s in tqdm(range(samples)):
        T = MAPS[s%2](n,n)
        current_iter = 0
        plt.imsave(data_folder + f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')
        for it in iters:
            T = iterate(T, it, dt, dx, coef_diffusion)
            current_iter += it
            plt.imsave(data_folder+f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')

def generate_data3(samples, T0_func, coef_diffusion=0.003, iters=(10,40,50,450), n=100, dx=0.01, dt=0.01):
    #this fct is to be used with RNN and HeatDataset3

    import os
    data_folder = f"data/{str(coef_diffusion).replace('.','_')}/{T0_func.__name__}_{str(coef_diffusion).replace('.','_')}"
    while os.path.exists(data_folder):
        data_folder += "(1)"
    os.makedirs(data_folder)

    os.makedirs(data_folder + f'/iteration_no{str(0)}')
    for i in range(len(iters)):
        os.makedirs(data_folder+f'/iteration_no{str(iters[i])}')

    for s in tqdm(range(samples)):

        T = T0_func(n,n)
        current_iter = 0
        plt.imsave(data_folder + f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')
        for it in iters:
            T = iterate(T, 1, dt, dx, coef_diffusion)
            current_iter += 1
            plt.imsave(data_folder+f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')


def multiple_generation():
    generate_data(300, T0blocksMap, coef_diffusion=0.0035)
    generate_data(200, T0blocksMapV2, coef_diffusion=0.0035)
    generate_data(100, T0sinusMap, coef_diffusion=0.0035)


nb_samples_per_alphas = 20
alphas = [0.0005 * i for i in range(1,9)] #160 samples in total
def multitask_generation(samples_per_alpha, T0_func, coefs_diffusion, iter=10, n=100, dx=0.01, dt=0.01):

    import os
    data_folder = f"data/{T0_func.__name__}"
    while os.path.exists(data_folder):
        data_folder += "(1)"

    os.makedirs(data_folder + f"/iteration_no0")
    os.makedirs(data_folder + f"/iteration_no{str(iter)}")

    for i in tqdm(range(len(coefs_diffusion))):
        for s in range(samples_per_alpha):

            T = T0_func(n,n)
            plt.imsave(data_folder + f"/iteration_no0/{str(coefs_diffusion[i]).replace('.', '')}_{str(s)}.png", T, vmin=0.0, vmax=1.0, cmap='gray')
            T = iterate(T, iter, dt, dx, coefs_diffusion[i])
            plt.imsave(data_folder + f"/iteration_no{iter}/{str(coefs_diffusion[i]).replace('.', '')}_{str(s)}.png", T, vmin=0.0, vmax=1.0, cmap='gray')












# simplify the saving instead of a matrix with 4 channels that i delete after,
# save a singe list of matrices no readable by human
# ask gpt to do it
"""
import os
import pickle

# Main folder path
main_folder_path = "/path/to/main/folder"  # Replace with your main folder path

# Function to check if a file has an image file extension
def is_image_file(filename):
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif']  # Add more image extensions if needed
    return any(filename.lower().endswith(ext) for ext in image_extensions)

# Walk through the main folder
for root, dirs, files in os.walk(main_folder_path):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        # Check if the folder contains image files
        image_files = [filename for filename in os.listdir(dir_path) if is_image_file(filename)]
        if image_files:
            # Create a Python object to save
            data = {"folder_name": dir_name, "image_files": image_files}
            
            # Save the Python object as a pickle file with the same name as the original folder
            pickle_filename = os.path.join(root, dir_name + '.pickle')
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {dir_name} as {pickle_filename}")

"""







