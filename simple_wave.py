from lad import lad_training
from lad import lad_testing
from lad_harmonics import lad_training_harmonics
import numpy as np
import sumpf
import sumpf_staging
import utilities
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

result = []
resultFormula = []
#resultMLS = [0.53, 0.56, 0.56]
#inputMLS = [2, 3, 4]
inputs = []
input_wave = np.zeros((1, 9, 10000))
for c in range(2, 1000):
	for i in range(0, 10000, c):
		input_wave[0, 0:9, i] = 1
	spectrogram = sumpf.Spectrogram(channels = input_wave)
	res = lad_training(spectrogram.magnitude())
	result.append(res) 
	resultFormula.append(pow(c-1, 2) / (pow(c, 2) +1) )
	input_wave = np.zeros((1, 9, 10000))
	inputs.append(c)

yellow_patch = mpatches.Patch(color='yellow', label='LAD')
black_patch = mpatches.Patch(color='black', label='Formula')
red_patch = mpatches.Patch(color='red', label='MLS')
plt.plot(inputs, result,'yo', inputs, resultFormula, 'k')
plt.title('Change in Time-Right Weight According to Consecutivity')
plt.ylabel('Time-Right Weight')
plt.xlabel('Consecutivity')
plt.legend(handles=[yellow_patch, black_patch])
plt.show()


#input_wave = np.zeros((1, 300, 21))
#for i in range(0, 21, 4):
	#input_wave[0, 0:300, i] = 1
#spectrogram = sumpf.Spectrogram(channels = input_wave)
#lad_training(spectrogram .magnitude())
#lad_training_harmonics(spectrogram.magnitude())

#lad_training(spectrogram .magnitude())
#result = lad_testing(spectrogram.magnitude())
#result_spectrogram = sumpf.Spectrogram(channels = result)
#plot = utilities.plot(result_spectrogram, log_magnitude=False) 
#plot = plot.plot(spectrogram, log_magnitude=False)
#plot.show()



