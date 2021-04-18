from lad import lad_training
from lad import lad_testing
import numpy as np
import sumpf
import sumpf_staging
import utilities
import matplotlib.pyplot as plt


inputList = []
snrArray = []

bound = 0.6
signal = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)

for f in range(10, 1000, 10):
	straight = 0
	for i in range(f, 2049, f):
		signal = signal + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)
	spectrogram = signal.short_time_fourier_transform()
	straight = lad_training(spectrogram.magnitude())
	while bound<5:
		noise = sumpf.GaussianNoise(mean=0.0, standard_deviation=bound, sampling_rate=2048, length=16384)
		signal_noise = signal + noise
		spectrogram_noise = signal_noise.short_time_fourier_transform()
		response = lad_training(spectrogram_noise.magnitude())
		if response < straight * 0.95:
			level_1 = pow(signal.level(True), 2)
			level_2 = pow(noise.level(True), 2)
			snr = level_1/level_2
			snr = 10.0 * np.log10(snr)
			if snr > 0:
				snrArray.append(snr)
				inputList.append(f)
			print(f, snr, '	BOUND: ' ,bound)
			break
		bound = bound + 0.001
	signal = sumpf.SineWave(frequency=0.0, sampling_rate=2048, length=16384)
	bound=0.001

plt.plot(inputList, snrArray,'k')
plt.title('CUT-OFF SNR POINTS ACCORDING TO THE FREQUENCY DISTANCE')
plt.ylabel('Signal-Noise Ratio')
plt.xlabel('Frequency Distance')
plt.show()