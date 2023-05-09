"""
This is a 2d modelisation of the heat transfert. The equation used is known as the heat equation :
diff(T, t) - alpha[ diff2(T, x) + diff2(T, y)] = 0

First, we define some functions to create an initial temperature matrix (or map) T0.

Then we use the function 'diffuse' that implement a descretization of the heat transfert problem.
This function takes an matrix T0 and compute the temperature matrix T after a certain time.

Finally we define some functions that serve the purpose of saving a great number of T0 and T maps as images.

The data generated with these functions can be uploaded to a machine learning script through a torch.Dataset object
by using the module "customDataset.py"

NB : for physics purist, the spatial and time parameters have all been rigorously defines
with units specified in the function "diffuse".
Nonetheless the initial matrices set temperatures between 0 and 1 and thus need to be scaled to the desired physical values.
"""


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# /////////////////////////////////////// Defining T0 matrices ///////////////////////////////////////


def T0blocksMap(nx,ny):
    """
    Create a numpy.array of size ny,nx composed of 16 blocks (ny/4, nx/4) of all zeros or all ones randomly.
    (according to uniform law)
    """
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

def T0blocksMapV2(nx,ny):
    """
    Create a numpy.array of size ny,nx composed of blocks of all zeros or all ones randomly (according to uniform law).
    The size of the blocks are random too.
    """
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

def T0sinusMap(nx,ny):
    """
    Create a numpy.array of size ny,nx composed of values between 0 and 1
    following a sinus-like distribution along both axis.
    The sinus are randoms
    """
    x, y = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx))
    a = np.random.normal()
    b = np.random.normal()
    matrix = np.sin(a*2*np.pi*x) * np.sin(b*2*np.pi*y) + np.random.normal(scale=0.05, size=(ny, nx))
    matrix = np.clip(matrix, 0, 1)
    return matrix

def T0testMap(nx,ny):
    """
    Create a numpy.array of size ny,nx composed of values between 0 and 1.
    the creation of the array is based on T0sinusMap and T0blocksMapV2
    the numpy array will consist of randomly generated blocks of random sizes,
    each blocks composed either of data from a matrix T0sinusMap or T0blocksMapV2

    This function is used to make an ultimate test on trained trained_models
    to see if they can perform well on sharp and smooth T0 altogether
    """
    T0_blocks = T0blocksMapV2(nx,ny)
    T0_sinus = T0sinusMap(nx,ny)

    mask = T0blocksMapV2(nx,ny)
    negatif_mask = np.ones((nx,ny))-mask
    #blocks where value = 0 will be replaced by T0_blocks and where value = 1 by T0_sinus

    return T0_blocks*mask + T0_sinus*negatif_mask

# //////////////////////////////// Implementing the diffusion of heat ////////////////////////////////


def diffuse(T_matrix, iterations, dt, dx, coef_diff):
    """
    Takes a map of temperature as input and return a new map where temperature has evolve during a time step*dt.

    Discretisation of the heat diffusion equation in 2D :
    Temperature = T(x,y,t)
    diff(T, t) = alpha * [ diff2(T, x) + diff2(T, y)] = 0
        --> T(t+dt) = T(t) + [alpha*dt/(dx)**2] * [ T(x+dx)+T(x-dx) + T(y+dy)+T(y-dy) - 4*T(x,y) ]

    :param T_matrix: temperature map at t initial (t_i)
    :param iterations: number of steps to evolve the temperature map
    :param dt : duration of one iteration in second
    :param dx : size of the spatial discretization in meter (dy is the same)
    :param coef_diff : diffusivité thermique m2.s-1

    :return: the new temperature map at t = t_i + iterations*dt
    """

    T = T_matrix.copy()
    ny,nx = T.shape #convention x is number of columns to fit with a basic plot (x,y)  (in fact -y cause matrices display)
    coef = coef_diff * dt / (dx * dx)
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
    pixels = 100 # Number of grid points in each direction

    # all the following is just a scaling of the coef of diffusion nothing more
    # the discretization time however appears in the diffusion coef too but also on the nb of iteration (precision)
    size = 1 #in meter
    spatial_step = size/pixels #spatial discretization step in meter
    alpha = 0.003 # Diffusion coefficient

    t1 = 1.0 # Total time
    time_step = 0.01 # Time discretization step in sec
    N = int(t1 / time_step) # Number of time steps

    # Initialize grid
    u0 = T0blocksMapV2(pixels,pixels)

    #make 3 matrices of diffusion.
    u1 = diffuse(u0, N, time_step, spatial_step, alpha)
    u2 = diffuse(u1, 5*N, time_step, spatial_step, alpha)
    u3 = diffuse(u2, 10*N, time_step, spatial_step, alpha)


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
    plt.show()


# //////////////////////////////////// Generating data as images ////////////////////////////////////


def generate_data(n_samples, T0_func, n=100, iters=(10,40,50), coef_diffusion=0.003, dx=0.01, dt=0.01):
    """
    Create n_samples T0 matrices.
    For each of these, apply the function "diffuse" for the number of iterations specified.
    Save the results in the following path after cwd :
        data/{coef_diffusion}/{T0_func}/iteration_no0/ for the T0 matrices (T initial)
        data/{coef_diffusion}/{T0_func}/iteration_noN/ for the TN matrices (T0 after N iterations of the diffusion)

        samples are named nosample.png

    :param n_samples: number of samples wanted
    :param T0_func: func to use for creation of T0
    :param n: nb of row and cols of the matrices
    :param iters: if type int : nb of iterations to apply on T0 before saving the result.
                  if type list/tupple : T1 = diffusion of T0 during iters[0] time steps
                                        T2 = diffusion of T1 during iters[1] time steps
                                        ...and so on
                                in this case iters=(5,10,15) will produce the images for iteration_no0, no5, no15, no30
    :param coef_diffusion: coef alpha in heat equation in m^2.s^-1
    :param dx: spatial discretization step (horizontal and vertical is the same) in meter
    :param dt: temporal discretization step (duration of one iteration step) in second
    :return: None

    N.B. load the data with customDataset.HeatDiffusion
    """
    #could change iters into times but create an error intrinsic to the modelisation
    #not good to check performances of the nn later, or we should quantizize this error

    if type(iters) == int:
        iters = [iters]

    import os
    data_folder = f"data/{str(coef_diffusion).replace('.','_')}/{T0_func.__name__}"
    while os.path.exists(data_folder):
        data_folder += "(1)"
    os.makedirs(data_folder)

    cummulative_iters = [0] + list(iters)
    os.makedirs(data_folder + f'/iteration_no{str(0)}')
    for i in range(1,len(cummulative_iters)):
        cummulative_iters[i]+=cummulative_iters[i-1]
        os.makedirs(data_folder+f'/iteration_no{str(cummulative_iters[i])}')

    for s in tqdm(range(n_samples)):
        T = T0_func(n,n)
        current_iter = 0
        plt.imsave(data_folder + f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')
        for it in iters:
            T = diffuse(T, it, dt, dx, coef_diffusion)
            current_iter += it
            plt.imsave(data_folder+f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')
    return None

def generate_data2(n_samples, n=100, iters=(10,40,50), coef_diffusion=0.003, dx=0.01, dt=0.01):
    """
    Same as "generate_data" except we do not choose T0_func, it is set as a mix of T0blocksMapV2 and T0sinusMap
    """
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
    for s in tqdm(range(n_samples)):
        T = MAPS[s%2](n,n)
        current_iter = 0
        plt.imsave(data_folder + f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')
        for it in iters:
            T = diffuse(T, it, dt, dx, coef_diffusion)
            current_iter += it
            plt.imsave(data_folder+f'/iteration_no{str(current_iter)}/{str(s)}.png', T, vmin=0.0, vmax=1.0, cmap='gray')
    return None

def generate_data_multialpha(n_samples_per_alpha=20, T0_func=T0blocksMap, n=100, it=10, coefs_diffusion=(0.003,), dx=0.01, dt=0.01):
    """
    Create (samples_per_alpha*len(coefs_diffusion)) T0 matrices.
    For each of these, apply the function "diffuse" for the number of iterations specified.
    Save the results in the following path after cwd :
        data/{T0_func}/iteration_no0/ for the T0 matrices (T initial)
        data/{T0_func}/iteration_noN/ for the TN matrices (T0 after N iterations of the diffusion)

        samples are named {decimals_of_alpha}_nosample.png

    :param n_samples_per_alpha: number of samples wanted for each value of alpha (diffusion coef)
    :param T0_func: func to use for creation of T0
    :param n: nb of row and cols of the matrices
    :param it: type int, nb of iterations to apply on T0 before saving the result.
        (does not currently support iters as a list converselly to 'generate_data')
    :param coefs_diffusion: coefs alpha in heat equation in m^2.s^-1
        type : list, the dataset will contains n samples per alpha, all mixed
    :param dx: spatial discretization step (horizontal and vertical is the same) in meter
    :param dt: temporal discretization step (duration of one iteration step) in second
    :return: None

    N.B. load the data with customDataset.HeatDiffusion_multialpha
    """
    coefs_diffusion = list(coefs_diffusion)

    import os
    data_folder = f"data/{T0_func.__name__}"
    while os.path.exists(data_folder):
        data_folder += "(1)"

    os.makedirs(data_folder + f"/iteration_no0")
    os.makedirs(data_folder + f"/iteration_no{str(it)}")

    for i in tqdm(range(len(coefs_diffusion))):
        for s in range(n_samples_per_alpha):

            T = T0_func(n,n)
            plt.imsave(data_folder + f"/iteration_no0/{str(coefs_diffusion[i]).replace('.', '')}_{str(s)}.png", T, vmin=0.0, vmax=1.0, cmap='gray')
            T = diffuse(T, it, dt, dx, coefs_diffusion[i])
            plt.imsave(data_folder + f"/iteration_no{it}/{str(coefs_diffusion[i]).replace('.', '')}_{str(s)}.png", T, vmin=0.0, vmax=1.0, cmap='gray')
    return None



# simplify the saving instead of a matrix with 4 channels that i delete after,
# save a singe list of matrices no readable by human
# ask gpt to do it
