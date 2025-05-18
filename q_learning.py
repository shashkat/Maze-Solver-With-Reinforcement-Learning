# Here, I will implement q-learning myself, to get a better understanding about it, and make it 
# interesting by implementing a maze solver.

# imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from functions import *
import os

# hyperparameters
num_episodes = 1000 # this is the number of times we will try to go from start to goal, each time learning about which move is good/bad from each position in the maze
episodes_and_next_10_to_plot = [100, 990] # ensure that for each argument supplied, arg+9 <= num_episodes-1
gamma = 0.99 # coefficient of long term importance (it being more means that we give more importance to long term results than short term)
epsilon = 0.3 # coefficient of exploration (it being more means that in each move in the maze, the agent with more likely just explore with a random action instead of choosing the best known action so far from there)
# for grid, start is 1, goal is 2, obstructed is -1 and path is 0
grid = generate_maze(25, 25, 0.7) # generate the maze numpy array, with -1 as obstructions, 1 as start, 2 as goal and 0 as path.

# if want to make a manual grid (else, use the generate_maze function, below)
# grid = np.array([
# 	[0, 0, 0, 0, 0, 0, 0],
# 	[0, 0, 0,-1, 0, 0, 0],
# 	[0, 0, 0,-1, 0, 0, 0],
# 	[0, 1, 0,-1, 0, 2, 0],
# 	[0, 0, 0,-1, 0, 0, 0],
# 	[0, 0, 0,-1, 0, 0, 0],
# 	[0, 0, 0, 0, 0, 0, 0]
# ])

# obtain some information from the hyperparameters and specify the possible actions, which is generally not required to be edited by the user.
start = (np.where(grid == 1)[0][0], np.where(grid == 1)[1][0]) # get the index of the entry in grid which has 1. assumes only one location where 1 is there in grid
goal = (np.where(grid == 2)[0][0], np.where(grid == 2)[1][0]) # get the index of the entry in grid which has 2. similar assumptions as for start
possible_actions = [ # the possible actions that can be taken from each position in the maze
	(0,1),  # right
	(1,0),  # down
	(0,-1), # left
	(-1,0), # up
]
action_to_action_index = {action:itr for itr,action in enumerate(possible_actions)} # this is useful when we know which action, but not know which index it corresponds to, for the purpose of update the q-table

# initialize q-table with zeros. It has shape: (num_actions, grid_height, grid_width). 
# q-table is the table which stores the information about the q-score (score of goodness), of each action from each position
q_table = np.zeros((len(possible_actions), grid.shape[0], grid.shape[1]))

# actually run the training
all_positions = [] # list which will store dicts of individual datapoints, which will be transformed to a df
for i in tqdm(range(num_episodes)):
	# for current episode, set the current state to be the start position
	current_state = start
	all_positions.append({'iteration': i, 'step': 0, 'x': current_state[0], 'y': current_state[1]})
	# keep on computing the next move and move accordingly for current episode, until we reach the goal
	step = 1 # variable which stores which step we are at for current episode
	while current_state != goal: # currently, the only stopping condition for an episode is reaching the goal. But later, can add a max steps taken threshold too. 
		# Run one iteration to find the next action, make changes to the q-table according to the reward we find there, and update position if the new position was valid
		current_state, q_table = PerformStepIteration(current_state, q_table, grid, possible_actions, action_to_action_index, epsilon, gamma)
		all_positions.append({'iteration': i, 'step': step, 'x': current_state[0], 'y': current_state[1]})
		step += 1

# now convert the all_positions list to df
df = pd.DataFrame(all_positions)
# save in a csv
df.to_csv('./all_positions.csv', index = False)
# also save the grid so that when making images in R, the grid is also understood
np.savetxt('./grid.csv', grid, delimiter=',')

# now call the R script which makes the plots for the episodes
command = 'Rscript plot_images.R'
for i in range(len(episodes_and_next_10_to_plot)):
	command = command + ' ' + str(episodes_and_next_10_to_plot[i])
os.system(command) # the arguments are which timepoints we want to make the animation for (and the next 10 time points, as then we get a better idea of learning at a time point)


























