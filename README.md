# AdaptiveMultibandFilter
This is a basic Python script to create a simple 3-band (fundamental, 1st and 2nd order formants of vocal range) adaptive bandpass filter by automatically detecting the fundamental frequency (F0) of the voice and adjusting the filter parameters with it.

Python Dependencies: 
- `pyaudio` - For real-time audio input/output
- `numpy` - For numerical operations
- `scipy` - For biquad filter design and application
- `librosa` - Fundamental frequency estimation (using `pyin`).

**Demo plotter screenshots with Q-Factor = 1.2**

Low-Pitched Voice (F0 ~ 150Hz)

![Multiband Low Pitch](https://github.com/Dybios/AdaptiveMultibandFilter/blob/main/docs/multiband_output_low_pitch.png?raw=true)

High-Pitched Voice (F0 ~ 400Hz)

![Multiband High Pitch](https://github.com/Dybios/AdaptiveMultibandFilter/blob/main/docs/multiband_output_high_pitch.png?raw=true)
