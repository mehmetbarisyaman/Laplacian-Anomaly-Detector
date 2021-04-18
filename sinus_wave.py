from lad_harmonics import lad_training_harmonics
from lad import lad_training
from lad import lad_testing
import numpy as np
import sumpf
import sumpf_staging
import utilities
import matplotlib.pyplot as plt

signal = sumpf.SineWave(frequency=200.0, sampling_rate=2048, length=16384)
#for i in range(2, 2049, 2):
	#signal = signal + sumpf.SineWave(frequency=i, sampling_rate=2048, length=16384)
spectrogram = signal.short_time_fourier_transform()
#lad_training_harmonics(spectrogram.magnitude())
lad_training(spectrogram.magnitude())
spectrogram2 = lad_testing(spectrogram.magnitude())
result_spectrogram = sumpf.Spectrogram(channels = spectrogram2)


plot = utilities.plot(spectrogram, log_frequency=False, log_magnitude=True) 
plot = plot.plot(result_spectrogram)

#plot = plot.plot(added_signals)
#plot = plot.plot(spectrogram)
#plot[0].set_xlim(0, 5) 
#plot[1].set_xlim(0, 5)
plot.show()





