import numpy as np

def cauchy_normalization(intermediate_dataset, average):
  return 1/(1 + pow(intermediate_dataset/average, 2))

def lad_training_harmonics(X):
  r = X.shape[0] * X.shape[1]
  c = X.shape[2]
  X = X.reshape(r, c)
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
  #2nd Harmonics
  second_harmonic_list = np.zeros((int(X.shape[0]/2)-1, X.shape[1]))
  total_range = X.shape[0]
  for i in range(1, int(total_range/2)):
    second_harmonic_list[i-1] = X[total_range - 2*i-1] - X[total_range - i-1]
    print(second_harmonic_list[i-1])
  #3rd Harmonics
  third_harmonic_list = np.zeros((int(X.shape[0]/3)-1, X.shape[1]))
  for i in range(1, int(total_range/3)):
    third_harmonic_list[i-1] = X[total_range - 3*i-1] - X[total_range - i -1]
    
  frequency_up_difference = cauchy_normalization(frequency_up_difference, average)
  frequency_down_difference = cauchy_normalization(frequency_down_difference, average)
  time_left_difference = cauchy_normalization(time_left_difference, average)
  time_right_difference = cauchy_normalization(time_right_difference, average)
  second_harmonic_list = cauchy_normalization(second_harmonic_list, average)
  third_harmonic_list = cauchy_normalization(third_harmonic_list, average)
  weight_vector = [np.mean(time_left_difference), np.mean(time_right_difference), np.mean(frequency_down_difference), np.mean(frequency_up_difference), np.mean(second_harmonic_list), np.mean(third_harmonic_list)]
  np.savetxt('weight_vector_harmonics', weight_vector)
  return 

def lad_testing_harmonics(X):
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
        totalDifference += weights[0] * abs(X[r][c] - X[r][c-1])
        weightCounter += 1
      if c < column-1: #time right difference
        totalDifference += weights[1] * abs(X[r][c] - X[r][c+1])
        weightCounter += 1
      if r > 0: #frequency down difference
        totalDifference += weights[2] * abs(X[r][c] - X[r-1][c])
        weightCounter += 1
      if r < row-1: #frequency up difference
        totalDifference += weights[3] * abs(X[r][c] - X[r+1][c])
        weightCounter += 1
      if 2*r < row-1: #2nd harmonics
        totalDifference += weights[4] * abs(X[r][c] - X[2*r][c])
        weightCounter += 1
      if 3*r < row-1: #3rd harmonics
        totalDifference += weights[5] * abs(X[r][c] - X[3*r][c])
        weightCounter += 1
      out[r][c] = totalDifference/weightCounter
  return out.reshape(1, out.shape[0], out.shape[1])
