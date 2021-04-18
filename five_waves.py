from lad import lad_training
from lad import lad_testing
import numpy as np
import sumpf
import sumpf_staging
import utilities
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

signal = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)
signal1 = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)
signal2 = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)
signal3 = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)
signal4 = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)
result = []
result1 = []
result2 = []
result3 = []
result4 = []
inputList = []
snrArray = []
snrArray1 = []
snrArray2 = []
snrArray3 = []
snrArray4 = []
bound = 0.001

for i in range(100, 2049, 100):
	signal = signal + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)

for i in range(200, 2049, 200):
	signal1 = signal1 + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)

for i in range(300, 2049, 300):
	signal2 = signal2 + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)

for i in range(400, 2049, 400):
	signal3 = signal3 + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)

for i in range(500, 2049, 500):
	signal4 = signal4 + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)


while(bound<1):
	f = int(bound*1000)
	noise = sumpf.GaussianNoise(mean=0.0, standard_deviation=bound, sampling_rate=2048, length=16384)
	signal_noise = signal + noise
	signal_noise1 = signal1 + noise
	signal_noise2 = signal2 + noise
	signal_noise3 = signal3 + noise
	signal_noise4 = signal4 + noise
	level = pow(signal.level(True), 2)
	level1 = pow(signal1.level(True), 2)
	level2 = pow(signal2.level(True), 2)
	level3 = pow(signal3.level(True), 2)
	level4 = pow(signal4.level(True), 2)
	level_noise = pow(noise.level(True), 2)
	snr = level/level_noise
	snr1 = level1/level_noise
	snr2 = level2/level_noise
	snr3 = level3/level_noise
	snr4 = level4/level_noise
	snr = 10.0 * np.log10(snr)
	snr1 = 10.0 * np.log10(snr1)
	snr2 = 10.0 * np.log10(snr2)
	snr3 = 10.0 * np.log10(snr3)
	snr4 = 10.0 * np.log10(snr4)
	spectrogram = signal.short_time_fourier_transform()
	spectrogram1 = signal1.short_time_fourier_transform()
	spectrogram2 = signal2.short_time_fourier_transform()
	spectrogram3 = signal3.short_time_fourier_transform()
	spectrogram4 = signal4.short_time_fourier_transform()
	spectrogram_noise = signal_noise.short_time_fourier_transform()
	spectrogram_noise1 = signal_noise1.short_time_fourier_transform()
	spectrogram_noise2 = signal_noise2.short_time_fourier_transform()
	spectrogram_noise3 = signal_noise3.short_time_fourier_transform()
	spectrogram_noise4 = signal_noise4.short_time_fourier_transform()
	snrArray.append(round(snr, 2))
	snrArray1.append(round(snr1, 2))
	snrArray2.append(round(snr2, 2))
	snrArray3.append(round(snr3, 2))
	snrArray4.append(round(snr4, 2))
	response = lad_training(spectrogram_noise.magnitude())
	response1 = lad_training(spectrogram_noise1.magnitude())
	response2 = lad_training(spectrogram_noise2.magnitude())
	response3 = lad_training(spectrogram_noise3.magnitude())
	response4 = lad_training(spectrogram_noise4.magnitude())
	result.append(response)
	result1.append(response1)
	result2.append(response2)
	result3.append(response3)
	result4.append(response4)
	inputList.append(snr)
	bound = bound + 0.001

yellow_patch = mpatches.Patch(color='yellow', label='100Hz')
green_patch = mpatches.Patch(color='green', label='200Hz')
magenta_patch = mpatches.Patch(color='magenta', label='300Hz')
red_patch = mpatches.Patch(color='red', label='400Hz')
blue_patch = mpatches.Patch(color='blue', label='500Hz')


plt.plot(inputList, result,'yellow', inputList, result1, 'green', inputList, result2,'magenta', inputList, result3, 'red', inputList, result4, 'blue')
plt.title('CHANGE IN FREQUENCY WEIGHT WITH THE FREQUENCY DISTANCE')
plt.ylabel('Frequency-Up Weight')
plt.xlabel('Signal-Noise Ratio')
plt.legend(handles=[yellow_patch, green_patch, magenta_patch, red_patch, blue_patch])
plt.show()




