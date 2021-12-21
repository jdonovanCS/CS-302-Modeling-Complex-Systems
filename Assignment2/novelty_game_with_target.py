import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from tqdm import tqdm
import random
import math
import matplotlib.image as mpimg
import time
import scikits.bootstrap as bootstrap
import scipy.stats # for finding statistical significance
import sys


# update function
def update(grid, explored_grid):

    # make a copy of the grid
    newGrid = grid.copy()

    # track game
    global game

    # get average number of neighbors
    avgNbs = computeAverageNeighbors(grid)

    # compute neighbors
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = (grid[i, (j-1)%grid.shape[1]] + grid[i, (j+1)%grid.shape[1]] +
                grid[(i-1)%grid.shape[0], j] + grid[(i+1)%grid.shape[0], j] +
                grid[(i-1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i-1)%grid.shape[0], (j+1)%grid.shape[1]] +
                grid[(i+1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i+1)%grid.shape[0], (j+1)%grid.shape[1]])
            
            # subtract the target point if in neighbors
            if (target and ((i, (j-1)%grid.shape[1]) == target or (i, (j+1)%grid.shape[1]) == target
                or ((i-1)%grid.shape[0], j) ==target or ((i+1)%grid.shape[0], j) ==target
                or ((i-1)%grid.shape[0], (j-1)%grid.shape[1]) ==target or ((i-1)%grid.shape[0], (j+1)%grid.shape[1]) ==target
                or ((i+1)%grid.shape[0], (j-1)%grid.shape[1]) ==target or ((i+1)%grid.shape[0], (j+1)%grid.shape[1]) ==target)):
                neighbors -= neighbors%1.0

            # track explored cells
            if grid[i,j] == 1:
                explored_grid[i,j] = 1
                
            # apply rules
            if game == "novelty":
                
                # if turned on, turn off
                if grid[i,j] == 1:
                    newGrid[i,j] = 0
                    
                # if turned off, turn on if novel and connected
                elif grid[i,j] == 0:
                    if neighbors <= avgNbs and neighbors > 0:
                        if random.uniform(0,1) < stochasticity: newGrid[i,j] = 1                        # turn on if few neighbors but not above avg

                # if target point, novel, and connected, change to achieved value    
                else: 
                    if neighbors <= avgNbs and neighbors > 0:
                        if random.uniform(0,1) < stochasticity: newGrid[i,j] = .7
                

            # game of life
            if game == "life":
                # if on, turn off if more than 3 or less than 2 neighbors
                if (grid[i,j] == 1):
                    if (neighbors <2) or (neighbors > 3):
                        newGrid[i,j] = 0
                # if off, turn on if neighbors == 3
                else:
                    if neighbors ==3:
                        newGrid[i,j] = 1

            # following the trend / majority
            if game == "trendy":
                # if on
                if grid[i,j] == 1:
                    # turn off if connected, but not trendy
                    if neighbors > 0 and neighbors <= avgNbs:
                        if random.uniform(0,1) < stochasticity: newGrid[i,j] = 0                        # turn off if few neighbors but not above avg
                    # turn off if trendy with a very low probability
                    if neighbors > avgNbs:
                        # if random.uniform(0,1) < stochasticity**3 and stochasticity < 1: newGrid[i,j] = 0
                        # elif random.uniform(0,1) < (stochasticity-(random.uniform(0,1)))**3: newGrid[i,j] = 0
                        if random.uniform(0,1) < stochasticity*(1/8): newGrid[i,j] = 0

                # if off
                elif grid[i,j] == 0:
                    # turn on if not trendy with very low prob
                    if neighbors <= avgNbs and neighbors > 0:
                        # if random.uniform(0,1) < stochasticity**3 and stochasticity < 1: newGrid[i,j] = 1
                        # elif random.uniform(0,1) < (stochasticity-(random.uniform(0,1)))**3: newGrid[i,j] = 1
                        if random.uniform(0,1) < stochasticity*(1/8): newGrid[i,j] = 1

                    # turn on if trendy
                    elif (neighbors > avgNbs):
                        if random.uniform(0,1) < stochasticity: newGrid[i,j] = 1

                # if target
                else:
                    # turn to achieved if not trendy with very low prob
                    if neighbors <= avgNbs and neighbors > 0:
                        # if random.uniform(0,1) < stochasticity**3 and stochasticity < 1: newGrid[i,j] = .7 
                        # elif random.uniform(0,1) < (stochasticity-(random.uniform(0,1)))**3: newGrid[i,j] = .7
                        if random.uniform(0,1) < stochasticity*(1/8): newGrid[i,j] = .7

                    # turn achieved if trendy
                    elif (neighbors > avgNbs):
                        if random.uniform(0,1) < stochasticity: newGrid[i,j] = .7

    return newGrid


# compute average neighbors for every single cell in grid
def computeOverallAverageNeighbors(grid):
    count_on = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            count_on += grid[i,j]
    return (count_on*8.)/(grid.shape[0]*grid.shape[1])

# compute average neighbors for every cell that has at least one neighbor
def computeAverageNeighbors(grid):
    count_nbrs = 0
    count_cells_having_nbrs = 0
    
    # compute neighbors, same loop as used above with added skip for target
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if target and ((i,j) == target and grid[target[0], target[1]]) == 0.3:
                continue
            neighbors = (grid[i, (j-1)%grid.shape[1]] + grid[i, (j+1)%grid.shape[1]] +
                grid[(i-1)%grid.shape[0], j] + grid[(i+1)%grid.shape[0], j] +
                grid[(i-1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i-1)%grid.shape[0], (j+1)%grid.shape[1]] +
                grid[(i+1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i+1)%grid.shape[0], (j+1)%grid.shape[1]])
            
            # subtract target
            if (target and ((i, (j-1)%grid.shape[1]) == target or (i, (j+1)%grid.shape[1]) == target
                or ((i-1)%grid.shape[0], j) ==target or ((i+1)%grid.shape[0], j) ==target
                or ((i-1)%grid.shape[0], (j-1)%grid.shape[1]) ==target or ((i-1)%grid.shape[0], (j+1)%grid.shape[1]) ==target
                or ((i+1)%grid.shape[0], (j-1)%grid.shape[1]) ==target or ((i+1)%grid.shape[0], (j+1)%grid.shape[1]) ==target)):
                neighbors -= neighbors%1.0
            
            # if the cell has a neighbor, then use it in calculation of average
            if neighbors > 0:
                count_nbrs += neighbors
                count_cells_having_nbrs += 1
    
    return count_nbrs/(count_cells_having_nbrs*1.0)        

# for fun
def addCluster(i, j, size, grid):
 
    # add a cluster of size with randomized 1- or 0-valued cells
    for first in range(size):
        for second in range(size):
            grid[(i+first)%grid.shape[0], (j+second)%grid.shape[1]] = random.choice([0,1])
    if np.sum(grid) < 2:
        grid = addCluster(i, j, size, grid)
    return grid

# run
def run(grid, iterations, explored_grid):
    global achieved_exploration
    achieved_exploration = False
    achieved_target = False
    
    # loop for specificed iterations
    for i in (range(iterations)):
        # update grid
        grid = update(grid, explored_grid)
        
        # check if we achieved full exploration of space
        if not achieved_exploration:
            if np.sum(explored_grid) == (grid.shape[0] * grid.shape[1]):
                achieved_exploration = i
                print('Achieved full exploration at iteration:', i)
                # i = iterations
        
        # check if we achieved the target cell
        if not achieved_target:
            if target and (grid[target[0], target[1]] == .7):
                achieved_target = True
                print('Achieved target at iteration:', i)
                # i = iterations

        # yielding grid allows for video compilation
        yield grid

def plot_mean_and_bootstrapped_ci_over_time(input_data = None, name = "change me", x_label = "change me", y_label="change me", y_limit = None):
    """
    
    parameters: 
    input_data: (numpy array of shape (generations, num_repitions)) solution metric to plot
    name: (string) name for legend
    x_label: (string) x axis label
    y_label: (string) y axis label
    
    returns:
    None
    """

    generations = input_data.shape[0]

    CIs = []
    mean_values = []
    for i in range(generations):
        mean_values.append(np.mean(input_data[i]))
        CIs.append(bootstrap.ci(input_data[i], statfunction=np.mean))
    mean_values=np.array(mean_values)
    
    print(CIs)
    high = []
    low = []
    for i in range(len(CIs)):
        low.append(CIs[i][0])
        high.append(CIs[i][1])
    
    low = np.array(low)
    high = np.array(high)
    fig, ax = plt.subplots()
    ax.clear()
    y = range(0, generations)
    ax.plot(y, mean_values, label=name)
    ax.fill_between(y, high, low, color='b', alpha=.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if (name) and len(name)>0:
        ax.set_title(name)
    plt.show()

def plot_mean_and_bootstrapped_ci_multiple(input_data = None, title = 'overall', name = "change this", x_label = "x", y_label = "y", save_name=""):
    """ 
     
    parameters:  
    input_data: (numpy array of numpy arrays of shape (max_k, num_repitions)) solution met
    name: numpy array of string names for legend 
    x_label: (string) x axis label 
    y_label: (string) y axis label 
     
    returns: 
    None 
    """ 
 
    generations = len(input_data[0])
 
    fig, ax = plt.subplots() 
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label) 
    ax.set_title(title) 
    for i in range(len(input_data)): 
        CIs = [] 
        mean_values = [] 
        for j in range(generations): 
            mean_values.append(np.mean(input_data[i][j])) 
            CIs.append(bootstrap.ci(input_data[i][j], statfunction=np.mean)) 
        mean_values=np.array(mean_values) 
 
        print(CIs) 
        high = [] 
        low = [] 
        for j in range(len(CIs)): 
            low.append(CIs[j][0]) 
            high.append(CIs[j][1]) 
 
        low = np.array(low) 
        high = np.array(high) 

        y = range(0, generations) 
        ax.plot(y, mean_values, label=name[i]) 
        ax.fill_between(y, high, low, alpha=.2) 
        ax.legend()
    if len(save_name) > 0:
        plt.savefig(save_name)
    plt.show()
    
def plot_simple_bar_chart(input_data = None, title = 'overall', name= 'change this', x_label= "x", y_label= "y", save_name=""):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.bar(name, input_data)
    if len(save_name) > 0:
        plt.savefig(save_name)
    plt.show()

def main(clusters, stochasticity, size, iterations, initialization, target=False):
    # define grid
    N = size
    grid = np.zeros((N, N))
    explored_grid = np.zeros((size,size))
    if target:
        target = (random.choice(range(N)), random.choice(range(N)))
        grid[target[0], target[1]] = .3

    # define initial_states of grid
    # grid[N//2:(N//2+1), N//2:(N//2+1)] = 1
    # grid = np.random.choice([0,1], grid.shape[0]*grid.shape[1], p=[0.2,0.8]).reshape(grid.shape[0],grid.shape[1])

    # define other initial params
    iterations = iterations
    clusters = clusters
    stochasticity = stochasticity
    initialization = initialization

    global game
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Game of {}'.format(game), artist='Game: {}, Clusters: {}, Stochasticity: {}'.format(game, clusters, stochasticity))
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()
    fig.patch.set_facecolor('black')

    # loop for number of clusters in params
    if initialization == "cluster":
        for i in range(clusters):
            # calculate size of square cluster with size 2 - sqrt(N)
            size = random.choice(range(2, math.floor(math.sqrt(N))))
            # add cluster to grid
            grid = addCluster(random.choice(range(N)), random.choice(range(N)), size, grid)

    if initialization == 'random':
        grid = np.random.choice([0,1], grid.shape[0]*grid.shape[1], p=[0.2,0.8]).reshape(grid.shape[0],grid.shape[1])


    # record keeping
    discovery = []
    final_cells = []
    diversity = []
    # begin making video
    with writer.saving(fig, "game_of_{}_c{}_st{}_s{}_it{}_i{}.mp4".format(game, clusters, stochasticity, N, iterations, initialization), 300):  # last argument: dpi
        # plt.spy(grid, origin='lower')
        plt.imshow(grid, interpolation='none', cmap='binary')
        plt.axis('off')
        writer.grab_frame()
        plt.clf()
        # loop for the amount of iterations and grids after run function
        for i, x in enumerate(run(grid, iterations, explored_grid)):
            plt.title("iteration: {:03d}".format(i + 1))
            # plt.spy(grid, origin='lower')
            plt.imshow(grid, interpolation='none', cmap='binary')
            grid = x
            discovery.append(np.sum(explored_grid)/(grid.shape[0]*grid.shape[1]*1.0))
            final_cells.append(np.sum(grid)/(grid.shape[0]*grid.shape[1]*1.0))
            diversity.append(np.mean([np.std(grid, axis=0), np.std(grid, axis=1)]))
            plt.axis('off')
            writer.grab_frame()
            plt.clf()

    # output coverage grid of explored cells
    # if np.sum(explored_grid) < (grid.shape[0] * grid.shape[1]):
    #     plt.imshow(explored_grid)
    #     plt.show()
    return discovery, final_cells, diversity, achieved_exploration

games = ['trendy', 'novelty']
clusters = 1
stochasticity = 1
size = 200
iterations = 600
initialization = "cluster"
target=False
achieved_exploration = False
discovery_results = {}
final_cells_results = {}
diversity_results = {}
full_exploration_results = {}
num_runs = 10
for g in games:
    game = g
    discovery_results[g] = np.zeros((num_runs, iterations))
    final_cells_results[g] = np.zeros((num_runs, iterations))
    diversity_results[g] = np.zeros((num_runs, iterations))
    full_exploration_results[g] = np.zeros(num_runs)
    for i in range(num_runs):
        start_time = time.time()
        discovery_results[g][i], final_cells_results[g][i], diversity_results[g][i], full_exploration_results[g][i] = main(clusters,stochasticity,size,iterations,initialization,target)
        print('game: {}, initialization: {}, time: {}'.format(game, initialization, time.time()-start_time))
plt.close('all')
# plot_mean_and_bootstrapped_ci_over_time(input_data = np.transpose(diversity_results['novelty']), name = ["novelty"], x_label = "iteration", y_label = "Diversity")
# plot_mean_and_bootstrapped_ci_over_time(input_data = np.transpose(diversity_results['trendy']), name = ["trendy"], x_label = "iteration", y_label = "Diversity")
# plot_mean_and_bootstrapped_ci_over_time(input_data = np.transpose(discovery_results['novelty']), name = ["novelty"], x_label = "iteration", y_label = "Discovery")
# plot_mean_and_bootstrapped_ci_over_time(input_data = np.transpose(discovery_results['trendy']), name = ["trendy"], x_label = "iteration", y_label = "Discovery")
# plot_mean_and_bootstrapped_ci_over_time(input_data = np.transpose(final_cells_results['novelty']), name = ["novelty"], x_label = "iteration", y_label = "Cells")
# plot_mean_and_bootstrapped_ci_over_time(input_data = np.transpose(final_cells_results['trendy']), name = ["trendy"], x_label = "iteration", y_label = "Cells")


plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x) for k, x in diversity_results.items()], title="Diversity of cells over time", name=[x for x in games], x_label="iteration", y_label="Diversity", save_name="images/game_c{}_st{}_s{}_it{}_i{}_diversity.png".format(clusters, stochasticity, size, iterations, initialization))
plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x) for k, x in discovery_results.items()], title="Discovery of cells over time", name=[x for x in games], x_label="iteration", y_label="Discovery", save_name="images/game_c{}_st{}_s{}_it{}_i{}_discovery.png".format(clusters, stochasticity, size, iterations, initialization))
plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x) for k, x in final_cells_results.items()], title="Number of cells \"on\" over time", name=[x for x in games], x_label="iteration", y_label="Cells", save_name="images/game_c{}_st{}_s{}_it{}_i{}_cells.png".format(clusters, stochasticity, size, iterations, initialization))
plot_simple_bar_chart(input_data = [np.nanmean(np.where(x != 0, x, np.nan)) for k,x in full_exploration_results.items()], title="Iteration when full exploration achieved", name=[x for x in games], x_label="Game", y_label="Iteration", save_name="images/game_c{}_st{}_s{}_it{}_i{}_avgwhen_full_exploration.png".format(clusters, stochasticity, size, iterations, initialization))
plot_simple_bar_chart(input_data = [np.sum(np.where(x != 0,1,0)) for k,x in full_exploration_results.items()], title="Number of times full exploration achieved", name=[x for x in games], x_label="Game", y_label="Iteration", save_name="images/game_c{}_st{}_s{}_it{}_i{}_num_full_exploration.png".format(clusters, stochasticity, size, iterations, initialization))

# print(discovery_results)
# print(final_cells_results)
# print(diversity_results)
# print(full_exploration_results)