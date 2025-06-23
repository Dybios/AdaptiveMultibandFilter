# AdaptiveMultibandFilter
This is a basic Python script to create a simple 3-band (fundamental, 1st and 2nd order formants of vocal range) adaptive bandpass filter by automatically detecting the fundamental frequency (F0) of the voice and adjusting the filter parameters with it.

Python Dependencies: 
- `pyaudio` - For real-time audio input/output
- `numpy` - For numerical operations
- `scipy` - For biquad filter design and application
- `librosa` - Fundamental frequency estimation (using `pyin`).
- `matplotlib` - For plotting the realtime values of gain and F0
- `soundfile` - For writing audio output to a WAV file

**Demo Screengrab**

_Initialization and Caliberation Phase:_

![Caliberation & Init](https://github.com/Dybios/AdaptiveMultibandFilter/blob/main/docs/caliberation_and_init.gif?raw=true)


_After Caliberation:_

![After caliberation](https://github.com/Dybios/AdaptiveMultibandFilter/blob/main/docs/multiband_output.png?raw=true)
