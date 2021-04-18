from lad import lad_training
from lad import lad_testing
import numpy as np
import sumpf
import sumpf_staging
import utilities
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

signal = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)
result = []
inputList = []
resultFormula = []
snrArray = []
bound = 0.001

#for i in range(100, 2049, 100):
	#signal = signal + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)
signal = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)

while(bound<1):
	f = int(bound*1000)
	for i in range(f, 2049, f):
		signal = signal + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)
	#noise = sumpf.UniformNoise(lower_boundary=-bound, upper_boundary=bound, length=16384)
	snr = -70
	#bound_value = 10/(f + 50) 
	#bound_value = 0.5
	#noise = sumpf.GaussianNoise(mean=0.0, standard_deviation=bound_value, sampling_rate=2048, length=16384)
	while snr >=1:
		noise = sumpf.GaussianNoise(mean=0.0, standard_deviation=bound_value, sampling_rate=2048, length=16384)
		level_1 = pow(signal.level(True), 2)
		level_2 = pow(noise.level(True), 2)
		snr = level_1/level_2
		snr = 10.0 * np.log10(snr)
		bound_value = bound_value + 0.001
	spectrogram = signal.short_time_fourier_transform()
	#level_1 = pow(signal.level(True), 2)
	#level_2 = pow(noise.level(True), 2)
	#snr = level_1/level_2
	#signal_noise = signal + noise
	#spectrogram_noise = signal_noise.short_time_fourier_transform()
	#print(snr)
	snr = 70
	if snr > 0:
		#snrArray.append(round(snr, 2))
		response = lad_training(spectrogram.magnitude())
		result.append(response)
		resultFormula.append(pow(f-1, 2) / (pow(f, 2) +1) )
		inputList.append(f)
	bound = bound + 0.001
	signal = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)

#snrArray.sort()
#snrArray = [snrArray[0], snrArray[199], snrArray[399], snrArray[599], snrArray[799], snrArray[998]]
yellow_patch = mpatches.Patch(color='yellow', label='LAD')
black_patch = mpatches.Patch(color='black', label='Formula')
plt.plot(inputList, result,'yo', inputList, resultFormula, 'k')
plt.title('CHANGE IN FREQUENCY WEIGHT WITH THE FREQUENCY DISTANCE')
plt.ylabel('Frequency-Up Weight')
plt.xlabel('Frequency Distance')
plt.legend(handles=[yellow_patch, black_patch])
#plt2 = plt.twiny()
#plt2.set_xticklabels(snrArray)
#plt.xlabel('Signal-Noise Ratio')
plt.show()



