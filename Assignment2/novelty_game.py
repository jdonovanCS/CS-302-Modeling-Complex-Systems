import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from tqdm import tqdm
import random
import math
import matplotlib.image as mpimg

# define grid
N = 50
grid = np.zeros((N, N))
explored_grid = np.zeros((N,N))

# define initial_states of grid
# grid[N//2:(N//2+1), N//2:(N//2+1)] = 1
# grid = np.random.choice([0,1], grid.shape[0]*grid.shape[1], p=[0.05,0.95]).reshape(grid.shape[0],grid.shape[1])

# define other initial params
iterations = 250
clusters = 1
stochasticity = 1

game = "trendy"
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Game of life', artist='Game: {}, Clusters: {}, Stochasticity: {}'.format(game, clusters, stochasticity))
writer = FFMpegWriter(fps=5, metadata=metadata)
fig = plt.figure()
fig.patch.set_facecolor('black')

# update function
def update(grid):
    
    # make a copy of the grid
    newGrid = grid.copy()

    # track explored cells
    global explored_grid

    # get average number of neighbors
    avgNbs = computeAverageNeighbors(grid)

    # compute neighbors
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = (grid[i, (j-1)%grid.shape[1]] + grid[i, (j+1)%grid.shape[1]] +
                grid[(i-1)%grid.shape[0], j] + grid[(i+1)%grid.shape[0], j] +
                grid[(i-1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i-1)%grid.shape[0], (j+1)%grid.shape[1]] +
                grid[(i+1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i+1)%grid.shape[0], (j+1)%grid.shape[1]])
            
            # track explored cells
            if grid[i,j] == 1:
                explored_grid[i,j] = 1
                
            # apply rules
            if game == "novelty":
                if grid[i,j] == 1:
                    explored_grid[i,j] = 1
                    newGrid[i,j] = 0
                    # if neighbors > avgNbs or neighbors < 1:
                    #     if random.uniform(0,1) < stochasticity: newGrid[i,j] = 0                        # turn off if too many or too few neighbors
                    # elif neighbors > 0 and neighbors < avgNbs:
                    #     if random.uniform(0,2) < stochasticity: newGrid[i,j] = 1                        # leave on if few neighbors but not above avg
                        
                elif grid[i,j] == 0:
                    if neighbors < avgNbs and neighbors > 0:
                        if random.uniform(0,2) < stochasticity: newGrid[i,j] = 1                        # turn on if few neighbors but not above avg
                    # elif neighbors > avgNbs or neighbors < 1:
                    #     if random.uniform(0,2) < stochasticity: newGrid[i,j] = 0                      # leave off if too many or too few neighbors

            # game of life
            if game == "life":
                if (grid[i,j] == 1):
                    if (neighbors <2) or (neighbors > 3):
                        newGrid[i,j] = 0
                else:
                    if neighbors ==3:
                        newGrid[i,j] = 1

            # following the majority
            if game == "trendY":
                if grid[i,j] == 1:
                    if neighbors > 0 and neighbors < avgNbs:
                        if random.uniform(0,2) < stochasticity: newGrid[i,j] = 0
                    elif neighbors > avgNbs:
                        if random.uniform(0,1) < stochasticity*(1/8): newGrid[i,j] = 0
                        
                elif grid[i,j] == 0:
                    if neighbors < avgNbs and neighbors > 0:
                        # if random.uniform(0,2) < stochasticity**3 and stochasticity < 1: newGrid[i,j] = 1   # turn on very unlikely
                        # elif random.uniform(0,2) < (stochasticity-(random.uniform(0,2)))**3: newGrid[i,j] = 1
                        if random.uniform(0,1) < stochasticity*(1/8): newGrid[i,j] = 1
                    elif (neighbors > avgNbs):
                        if random.uniform(0,2) < stochasticity: newGrid[i,j] = 1

    return newGrid

# compute average neighbors
def computeOverallAverageNeighbors(grid):
    count_on = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            count_on += grid[i,j]
    return (count_on*8.)/(grid.shape[0]*grid.shape[1])

def computeAverageNeighbors(grid):
    count_nbrs = 0
    count_cells_having_nbrs = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = (grid[i, (j-1)%grid.shape[1]] + grid[i, (j+1)%grid.shape[1]] +
                grid[(i-1)%grid.shape[0], j] + grid[(i+1)%grid.shape[0], j] +
                grid[(i-1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i-1)%grid.shape[0], (j+1)%grid.shape[1]] +
                grid[(i+1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i+1)%grid.shape[0], (j+1)%grid.shape[1]])
            if neighbors > 0:
                count_nbrs += neighbors
                count_cells_having_nbrs += 1
    
    return count_nbrs/(count_cells_having_nbrs*1.0)        

# for fun
def addCluster(i, j, size, grid):
 
    """adds a glider with top left cell at (i, j)"""
    for first in range(size):
        for second in range(size):
            grid[(i+first)%grid.shape[0], (i+second)%grid.shape[1]] = random.choice([0,1])
    return grid

# run
def run(grid, iterations):
    achieved_exploration = False
    global explored_grid
    for i in (range(iterations)):
        grid = update(grid)
        
        if not achieved_exploration:
            if np.sum(explored_grid) == (grid.shape[0] * grid.shape[1]):
                achieved_exploration = True
                print('Achieved full exploration at iteration:', i)
                # i = iterations


        yield grid

for i in range(clusters):
    size = random.choice(range(2, math.floor(math.sqrt(N))))
    grid = addCluster(random.choice(range(N)), random.choice(range(N)), size, grid)

with writer.saving(fig, "game_of_{}_c{}_s{}_50_moving.mp4".format(game, clusters, stochasticity), 300):  # last argument: dpi
    plt.spy(grid, origin='lower')
    plt.axis('off')
    writer.grab_frame()
    plt.clf()
    for i, x in enumerate(run(grid, iterations)):
        plt.title("iteration: {:03d}".format(i + 1))
        plt.spy(x, origin='lower')
        grid = x
        plt.axis('off')
        writer.grab_frame()
        plt.clf()

if np.sum(explored_grid) < (grid.shape[0] * grid.shape[1]):
    plt.imshow(explored_grid)
    plt.show()



def life(X, steps):
    """
     Donovan's Game of Life.
     - X, matrix with the initial state of the game.
     - steps, number of generations.
    """
    def roll_it(x, y):
        # rolls the matrix X in a given direction
        # x=1, y=0 left;  x=-1, y=0 right;
        return np.roll(np.roll(X, y, axis=0), x, axis=1)

    for _ in range(steps):
        # count the number of neighbours
        # the universe is considered toroidal
        Y = roll_it(1, 0) + roll_it(0, 1) + \
            roll_it(-1, 0) + roll_it(0, -1) + \
            roll_it(1, 1) + roll_it(-1, -1) + \
            roll_it(1, -1) + roll_it(-1, 1)

        # count avgNeighbors
        avgNbs = ((np.count_nonzero(X == 1))*8)/(X.shape[0]*X.shape[1])

        # game of life rules
        X = np.logical_or(np.logical_and(X, Y==0), np.logical_and(X==0, Y < avgNbs, Y > 0))
        X = X.astype(int)
        yield X


# dimensions = (90, 160)  # height, width
# X = np.zeros(dimensions)  # Y by X dead cells
# middle_y = int(dimensions[0] / 2)
# middle_x = int(dimensions[1] / 2)

# print(middle_x, middle_y)

# N_iterations = 600

# acorn initial condition
# http://www.conwaylife.com/w/index.php?title=Acorn
# X[middle_y, middle_x:middle_x+2] = 1
# X[middle_y, middle_x+4:middle_x+7] = 1
# X[middle_y+1, middle_x+3] = 1
# X[middle_y+2, middle_x+1] = 1
# X = np.random.choice([0,1], dimensions[0]*dimensions[1], p=[0.5,0.5]).reshape(dimensions[0],dimensions[1])
# X[middle_y, middle_x] = 1

# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Game of life', artist='Acorn initial condition')
# writer = FFMpegWriter(fps=10, metadata=metadata)

# fig = plt.figure()
# fig.patch.set_facecolor('black')
# with writer.saving(fig, "game_of_life_onepixel.mp4", 300):  # last argument: dpi
#     plt.spy(X, origin='lower')
#     plt.axis('off')
#     writer.grab_frame()
#     plt.clf()
#     for i, x in enumerate(life(X, N_iterations)):
#         plt.title("iteration: {:03d}".format(i + 1))
#         plt.spy(x, origin='lower')
#         plt.axis('off')
#         writer.grab_frame()
#         plt.clf()