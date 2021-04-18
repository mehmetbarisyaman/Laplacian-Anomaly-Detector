from lad import lad_training
from lad import lad_testing
import numpy as np
import sumpf
import sumpf_staging
import utilities
import matplotlib.pyplot as plt


signal = sumpf.SquareWave(frequency=200.0, sampling_rate=2048, length=16384, phase=0.1)
spectrogram = signal.short_time_fourier_transform(window=8192)
lad_training(spectrogram.magnitude())
result = lad_testing(spectrogram.magnitude())

result_spectrogram = sumpf.Spectrogram(channels = result, resolution = spectrogram.resolution(), sampling_rate = spectrogram.sampling_rate())
plot = utilities.plot(result_spectrogram, log_frequency=False, log_magnitude=True, magnitude_range=-50) 
plot = plot.plot(spectrogram, magnitude_range=-50)
plot.show()