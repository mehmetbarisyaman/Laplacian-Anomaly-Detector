import numpy as np

def cauchy_normalization(intermediate_dataset, average):
  return 1/(1 + pow(intermediate_dataset/average, 2))
  
def lad_training(X):
  r = X.shape[0] * X.shape[1]
  c = X.shape[2]
  X = X[0]
  average = np.mean(X)
  up_frequency = np.roll(X, -1, axis=0)
  up_frequency[(r-1):r, 0:c] = X[(r-1):r, 0:c]
  frequency_up_difference = X - up_frequency
  down_frequency = np.roll(X, 1, axis=0)
  down_frequency[0:1, 0:c] = X[0:1, 0:c]
  frequency_down_difference = X - down_frequency
  right_time = np.roll(X, 1)
  right_time[0:r, 0:1] = X[0:r, 0:1]
  time_right_difference = X -  right_time
  left_time = np.roll(X, -1)
  left_time[0:r, (c-1):c] = X[0:r, (c-1):c]
  time_left_difference = X - left_time
  frequency_up_difference = cauchy_normalization(frequency_up_difference, average)
  frequency_down_difference = cauchy_normalization(frequency_down_difference, average)
  time_left_difference = cauchy_normalization(time_left_difference, average)
  time_right_difference = cauchy_normalization(time_right_difference, average)
  weight_vector = [np.mean(time_left_difference), np.mean(time_right_difference), np.mean(frequency_down_difference), np.mean(frequency_up_difference)]
  np.savetxt('weight_vector', weight_vector)
  return np.mean(frequency_up_difference)

def lad_testing(X):
  weights = np.loadtxt('weight_vector')
  row = X.shape[0] * X.shape[1]
  column = X.shape[2]
  X = X[0]
  out = np.zeros((row, column))
  #Computation for each bin
  for r in range(0, row):
  	for c in range(0, column):
  		weightCounter = 0
  		totalDifference = 0
  		if c > 0: #time left difference
  			totalDifference += weights[0] * (X[r][c] - X[r][c-1])
  			weightCounter += 1
  		if c < column-1: #time right difference
  			totalDifference += weights[1] * (X[r][c] - X[r][c+1])
  			weightCounter += 1
  		if r > 0: #frequency down difference
  			totalDifference += weights[2] * (X[r][c] - X[r-1][c])
  			weightCounter += 1
  		if r < row-1: #frequency up difference
  			totalDifference += weights[3] * (X[r][c] - X[r+1][c])
  			weightCounter += 1
  		out[r][c] = totalDifference/weightCounter
  return out.reshape(1, out.shape[0], out.shape[1])