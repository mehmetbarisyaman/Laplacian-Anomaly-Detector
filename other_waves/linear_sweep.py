from lad import lad_training
from lad import lad_testing
import numpy as np
import sumpf
import sumpf_staging
import utilities

signal = sumpf.LinearSweep(length=2 ** 18)
spectrogram = signal.short_time_fourier_transform()


lad_training(spectrogram.magnitude())
result = lad_testing(spectrogram.magnitude())

result_spectrogram = sumpf.Spectrogram(channels = result, resolution = spectrogram.resolution(), sampling_rate = spectrogram.sampling_rate())

plot = utilities.plot(result_spectrogram, log_frequency=False, log_magnitude=False) 
plot = plot.plot(spectrogram)
plot.show()