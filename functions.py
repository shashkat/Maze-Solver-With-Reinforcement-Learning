import numpy as np
import random

# compute which action to perform next, using the information about current state, q-table 
# in its current shape, and the epsilon value (exploration vs exploitation constant).
# possible_actions is a list of tuples, containing the possible actions
# epsilon tells how much we give importance to exploration vs to exploitation
def ComputeNextAction(current_state, q_table, epsilon, possible_actions):
	# generate a random number between 0 and 1, according to which we will take the decision of exploring vs exploiting
	random_num = np.random.rand()
	# if less then epsilon, then do exploration, else expoitation
	if random_num < epsilon:
		# do exploration: get a random number chosen from all possible action_indices
		action_index = np.random.randint(low = 0, high = len(possible_actions))
		action = possible_actions[action_index]
	else:
		# do exploitation: get the action which has max q-value in q-table for current position
		action_index = np.argmax(q_table[:, *current_state])
		action = possible_actions[action_index]
	return action

# compute what reward we will get for given action, from given state, given the q-table in its 
# current form
def ComputeRewardForAction(action, current_state, grid):
	# compute the next state using the current_state and action to be taken, so that we can compute the reward using the new position
	next_state = GetNextState(current_state, action)

	# if the next position is invalid (outside grid, or obstacle), then give heavy penalty
	if not IsValidState(next_state, grid):
		return -10
	
	# if the state was valid, then simply return a small penalty, which serves the purpose of 
	# trying to minimize the number of steps we take to reach the destination
	return -1

# get the next position an action will take us to from current state. We check validity of next 
# state later. This function just returns the position where the action will take us, 
# irrespective of its validity
def GetNextState(current_state, action):
	next_state = (current_state[0] + action[0], current_state[1] + action[1])
	return next_state

# get a boolean indicating if a pair of coordinates (position) is valid according to the supplied
# grid. Valid means inside boundary of grid, and not an obstacle position (identified by -1)
def IsValidState(state, grid):
	row_index = state[0]
	col_index = state[1]
	nrows = grid.shape[0]
	ncols = grid.shape[1]
	# check for being inside of row_index, col_index and presence of obstance in current position
	if row_index < 0 or row_index >= nrows or col_index < 0 or col_index >= ncols or grid[*state] == -1:
		return False
	return True

# update the q-table using the computed reward of an s,e pair, and current q-table.
# gamma tells how much weight we give to long-term impact of a decision
# NOTE: I am not using alpha in this function unlike the tutorial, because I think it doesn't have 
# much of an effect in the model learning, as it is basically how much we discount the reward for 
# each s,e pair, which just changes as actual values of the numbers, but not the end result.
# Also, I can possibly play around with the function to understand it better. Currently I dont 
# understand why we subtract the q-value of current position in the bracket term.
def UpdateQTable(action, current_state, q_table, grid, gamma, action_to_action_index):
	# formula: Q(s,a) += reward + gamma*(max(Q(s',a')) - Q(s,a))

	# make a copy of q_table so that we can make modifications without outside array getting affected (doing this way because I am more used to objects being passed as value in functions and not as reference)
	q_table_copy = q_table.copy()

	# get the reward that current action will yield
	reward = ComputeRewardForAction(action, current_state, grid)

	# also the the index corresponding to the action, as using that action_index, we can access the appropriate entries of q-table
	action_index = action_to_action_index[action]

	# get the best q-value obtainable from the next position as it is how we get an idea of long term gains (just one step)

	# get next valid position where we will be after this 
	if IsValidState(GetNextState(current_state, action), grid):
		next_state = GetNextState(current_state, action)
	else:
		next_state = current_state
	# now, get best q-value obtainable from next position
	best_q_value_from_next_position = np.max(q_table_copy[:, *next_state])

	# the term in bracket of the formula
	bracket_term = best_q_value_from_next_position - q_table_copy[action_index, *current_state]

	# update the q-table's appropriate entry appropriately
	q_table_copy[action_index, *current_state] += reward + gamma*bracket_term
	
	return q_table_copy

# find which step we should take next, compute the reward from the move, and update the 
# q-table accordingly, and finally make the change in current_state accordingly, if new 
# location is valid
def PerformStepIteration(current_state, q_table, grid, possible_actions, action_to_action_index, epsilon, gamma):
	# get which action to perform next
	action = ComputeNextAction(current_state, q_table, epsilon, possible_actions)

	# update q-table using the reward for the particular state-action pair
	q_table = UpdateQTable(action, current_state, q_table, grid, gamma, action_to_action_index)

	# update the current state according if valid, else stay at current position
	if IsValidState(GetNextState(current_state, action), grid):
		current_state = GetNextState(current_state, action)

	return current_state, q_table

# generate a maze (numpy array) of given height and width. -1 is obstruction, 1 is start, 2 is goal, and 0 is path
# percentage_of_remaining_hurdles_to_remove is the percentage of remaining obstructions we remove 
# after the initial maze is made. This was done because I realized that the maze was basically a 
# single path usually and not a bunch of path from which the algo would find the best one.
def generate_maze(height, width, percentage_of_remaining_hurdles_to_remove):
    '''
    Generate a maze (numpy array) of given height and width.

    Arguments:
        - height is nrows of numpy array
        - width is ncols of numpy array
        - percentage_of_remaining_hurdles_to_remove is percentage of remaining obstructions to remove after the initial maze is made (for having more than one paths to the goal, which better demonstrates learning).

    Returns: A numpy array of shape (height, width). -1 is obstruction, 1 is start, 2 is goal, and 0 is path.

    Example usage: 
        maze = generate_maze(13, 13, 0.1)
    '''

    # Ensure odd dimensions for proper maze structure
    height = height if height % 2 == 1 else height + 1
    width = width if width % 2 == 1 else width + 1
    maze = np.full((height, width), -1)  # -1 for walls
    # Start at (1,1)
    def carve(x, y):
        maze[y, x] = 0
        directions = [(0,2),(0,-2),(2,0),(-2,0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 1 <= nx < width-1 and 1 <= ny < height-1 and maze[ny, nx] == -1:
                maze[y+dy//2, x+dx//2] = 0
                carve(nx, ny)
    carve(1, 1)
    # Set start and goal
    maze[1, 1] = 1
    maze[height-2, width-2] = 2
    # finally, to have more paths, remove a certain percentage of -1s from the remaining maze
    for i,j in zip(np.where(maze == -1)[0], np.where(maze == -1)[1]):
        random_num = np.random.rand()
        if random_num < percentage_of_remaining_hurdles_to_remove:
            maze[i][j] = 0
    return maze



	









