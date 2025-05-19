library(tidyverse)
library(ggplot2)
library(reshape2)
library(glue)
library(gganimate)

# get the arguments given as input (which are the timepoints for which for which we want to create an animation)
args <- commandArgs(trailingOnly = TRUE)

# read the df
df <- read.csv('all_positions.csv')

# obtain a df from grid, storing x, y, and identity of each grid-cell
grid <- as.matrix(read.csv('grid.csv', header = FALSE))
colnames(grid) <- 0:(ncol(grid)-1)
rownames(grid) <- 0:(nrow(grid)-1)
grid_df_long <- melt(grid)
colnames(grid_df_long) <- c('x', 'y', 'value')
grid_df_long$value <- as.factor(grid_df_long$value) # make into factor so that colors are better

# loop through each timepoint center, and make the corresponding 10 near timepoint animations in one
for (arg in args){
  timepoint_centroid <- as.numeric(arg)
  timepoints_in_cluster = seq(from = timepoint_centroid, to = timepoint_centroid + 9)
  
  df_subset <- df |> filter(iteration %in% timepoints_in_cluster)
  
  p <- ggplot(df_subset, aes(x = x, y = y)) +
    geom_tile(data = grid_df_long, mapping = aes(fill = value, x = x, y = y, width = 1, height = 1)) + 
    geom_point(size = 7, na.rm = TRUE, aes(group = iteration)) + 
    transition_time(step) + 
    labs(title = 'step: {frame_time}') + 
    ease_aes('cubic-in-out') + 
    scale_y_reverse() + 
    theme_void()
  
  # get maximum number of steps and set the nframes to that only
  nframes = 2*max(df_subset$step)
  animate(p, nframes = nframes)
  
  # create the images directory if not already there and save the animation 
  dir.create('images') # this will create the dir if it doesn't already exist, else do nothing
  anim_save(glue('images/timepoint_centroid_{timepoint_centroid}.gif'), height = 6, width = 6)
}


