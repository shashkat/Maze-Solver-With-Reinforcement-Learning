### Maze-Solver-With-Reinforcement-Learning
Using q-learning to learn how to solve a maze. q-learning (quality learning) is a reinforcement learning algorithm which is useful for cases where we have a fixed number of possible positions and a fixed number of steps to take from each possible position. What happens in simple terms is the following- 

We start at the starting position in the grid, and gradually take decisions according to the q-table, which stores the best decision known thus far from each position. As we make a decision, we get feedback about how good that decision was (it was a bad decision if its taking us to an obstructed position or out of the grid). Using this information we update the q-table at each step according to an equation (Bellman equation). We stop when we reach the goal/endpoint of the maze. As one can understand, the first time it would take a lot of time to reach the goal, as it is basically to be stumbled upon by chance. But gradually, the learnings start paying off, and the iterations complete faster.

### Requirements
**Python**: `numpy`, `pandas`, `random`, `tqdm`

**R** (used for plotting and making the gif): `tidyverse`, `ggplot2`, `reshape2`, `glue`, `gganimate`

### Usage

Clone the repo and set working directory as Maze-Solver-With-Reinforcement-Learning. Then, one just has to run the `q_learning.py` file to generate the gifs for the maze solver (stored by default in the images folder). User can specify the following parameters by editing the hyperparameters section of q_learning.py:

- **num_episodes**: the number of times we will try to reach from start to goal, while learning the best move from each position.
- **episodes_and_next_10_to_plot**: Which timepoints to make the gif for. For each timepoint, the gif will be made visulializing the timepoint and the next 9 timepoints, for a better idea of learning arount that timepoint. Hence, ensure that you supply some value less than equal to num_episodes - 10.
- **gamma**: coefficient of long term importance (it being more means that we give more importance to long term results than short term).
- **epsilon**: coefficient of exploration (it being more means that in each move in the maze, the agent with more likely just explore with a random action instead of choosing the best known action so far from there).
- **Maze** grid (or change the dimensions of the random grid).

### Example:

Learning at timepoint 100:

![Learning at timepoint 100](images/timepoint_centroid_100.gif)

Learning at timepoint 990:

![Learning at timepoint 990](images/timepoint_centroid_990.gif)

### References
- Adapted q-learning algo code from this youtube tutorial: https://youtu.be/qTY4Rr-x5q0?si=LbdH3lRjo3Otrk-s
- The notebook with some more information about the hyperparameters I have used (like epsilon, gamma): https://github.com/ALucek/three-RL-projects/blob/main/three_RL_projects.ipynb


