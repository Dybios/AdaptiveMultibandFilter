import pyaudio
import numpy as np
import librosa
from scipy.signal import iirfilter, lfilter, freqz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import threading
import time

# --- Audio Stream Parameters ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4098

# --- Filter Parameters ---
TARGET_Q_FACTOR = 1.2
DEFAULT_F0_HZ = 150 # Default F0 when no voice is detected, for filter
MIN_VALID_F0 = 60   # Minimum F0 to consider valid for filtering
MAX_VALID_F0 = 500  # Maximum F0 to consider valid for filtering
MAKEUP_GAIN = 2.0   # Overall gain after filtering is done

# --- Pitch Estimation Parameters ---
FMIN_HZ = 50
FMAX_HZ = 600
HOP_LENGTH_F0 = CHUNK // 2

# Fixed Formant Filter Parameters
FIXED_FORMANT_FILTERS = [
    {'center_freq': 800, 'bandwidth_hz': 400}, # Band 1: Covers F1 for many vowels
    {'center_freq': 2000, 'bandwidth_hz': 800}, # Band 2: Covers F2/F3 for many vowels
    # {'center_freq': 3500, 'bandwidth_hz': 1000}, # Optional Band 3: For sibilance, higher formants
]

# --- Global Variables for Filter State and Plotting Data ---
# List to hold coefficients and states for ALL active filters
# Each element will be a tuple: (b_coeffs, a_coeffs, zi_state)
active_filters = [] # This will be initialized later

# Data for plotting
F0_HISTORY_LENGTH = 100 # Number of past F0 values to display
f0_history = collections.deque(np.full(F0_HISTORY_LENGTH, DEFAULT_F0_HZ), maxlen=F0_HISTORY_LENGTH)
time_history = collections.deque(np.arange(F0_HISTORY_LENGTH) * (CHUNK / RATE), maxlen=F0_HISTORY_LENGTH)

# A lock to prevent race conditions when updating global filter coeffs
filter_lock = threading.Lock()

# --- Filter Design Function ---
def design_bandpass_biquad(center_freq, bandwidth_hz, sampling_rate):
    """
    Designs a 2nd-order (biquad) bandpass filter using specified bandwidth.
    """
    # Calculate Q factor from center_freq and bandwidth_hz
    if bandwidth_hz <= 0: return None, None # Avoid division by zero
    q_factor = center_freq / bandwidth_hz

    # Ensure F0 is within valid range for the F0-tracking filter, and general valid freq for fixed filters
    if not (20 <= center_freq <= sampling_rate / 2 - 20): # Broader check for any filter
        # print(f"Debug: Center frequency {center_freq:.1f} Hz out of valid range. Skipping filter update.")
        return None, None

    # Ensure Q is reasonable (Q < 0.5 can lead to negative lower cutoff, very high Q is very narrow)
    if not (0.5 <= q_factor <= 100):
        # print(f"Debug: Calculated Q factor {q_factor:.2f} is extreme. Skipping filter update.")
        return None, None

    nyquist = sampling_rate / 2
    low_cutoff = (center_freq - bandwidth_hz / 2) / nyquist
    high_cutoff = (center_freq + bandwidth_hz / 2) / nyquist

    if not (0 < low_cutoff < high_cutoff < 1):
        # print(f"Debug: Calculated normalized cutoffs [{low_cutoff:.4f}, {high_cutoff:.4f}] out of range. Skipping.")
        return None, None

    try:
        b, a = iirfilter(2, [low_cutoff, high_cutoff], btype='bandpass', ftype='butter', output='ba')
        return b, a
    except ValueError as e:
        # print(f"Error designing filter: {e}. Check frequency parameters.")
        return None, None

# --- Initialization of Filters ---
def initialize_filters():
    global active_filters
    temp_filters = [] # Use a temporary list to build

    # 1. Initialize F0-tracking filter (initially passthrough or default)
    # Using a default bandwidth derived from DEFAULT_F0_HZ and TARGET_Q_FACTOR
    initial_f0_bandwidth = DEFAULT_F0_HZ / TARGET_Q_FACTOR
    b_f0, a_f0 = design_bandpass_biquad(DEFAULT_F0_HZ, initial_f0_bandwidth, RATE)
    if b_f0 is None or a_f0 is None: # Fallback to passthrough if initial design fails
        b_f0 = np.array([1.0, 0.0, 0.0])
        a_f0 = np.array([1.0, 0.0, 0.0])
    zi_f0 = np.zeros(max(len(b_f0), len(a_f0)) - 1)
    temp_filters.append({'type': 'F0_TRACKING', 'b': b_f0, 'a': a_f0, 'zi': zi_f0})

    # 2. Initialize Fixed Formant Filters
    for i, params in enumerate(FIXED_FORMANT_FILTERS):
        b_fixed, a_fixed = design_bandpass_biquad(params['center_freq'], params['bandwidth_hz'], RATE)
        if b_fixed is not None and a_fixed is not None:
            zi_fixed = np.zeros(max(len(b_fixed), len(a_fixed)) - 1)
            temp_filters.append({'type': f'FORMANT_FIXED_{i}', 'b': b_fixed, 'a': a_fixed, 'zi': zi_fixed})
        else:
            print(f"Warning: Failed to design fixed filter for {params['center_freq']}Hz. Skipping.")

    active_filters = temp_filters
    print(f"Initialized {len(active_filters)} filters.")

# Call initialization once at startup
initialize_filters()

# --- Audio Processing Function (to run in a separate thread) ---
def audio_callback(in_data, frame_count, time_info, status):
    global b_coeffs, a_coeffs, zi_state, f0_history, time_history

    audio_chunk = np.frombuffer(in_data, dtype=np.int16)
    audio_chunk_float = audio_chunk.astype(np.float32) / 32768.0

    # F0 Estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=audio_chunk_float,
        sr=RATE,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        frame_length=CHUNK,
        hop_length=HOP_LENGTH_F0
    )

    current_f0 = 0.0
    if np.any(voiced_flag == 1): # If any part of the chunk is voiced
        valid_f0s = f0[voiced_flag == 1]
        # Taking median to be more robust to outliers
        current_f0 = np.median(valid_f0s) if len(valid_f0s) > 0 else 0.0
        if np.isnan(current_f0):
            current_f0 = 0.0

    # Update F0 history for plotting
    f0_history.append(current_f0 if current_f0 > 0 else f0_history[-1]) # Keep last F0 if not voiced

    # Dynamic Filter Coefficient Update
    new_b, new_a = None, None
    if current_f0 > 0:
        new_b, new_a = design_bandpass_biquad(current_f0, TARGET_Q_FACTOR, RATE)

    # Update global filter coefficients safely
    with filter_lock:
        updated_f0_filter = None
        for i, f_data in enumerate(active_filters):
            if f_data['type'] == 'F0_TRACKING':
                if current_f0 > 0:
                    # Calculate bandwidth based on current F0 and fixed Q for F0-tracking filter
                    current_f0_bandwidth = current_f0 / TARGET_Q_FACTOR
                    new_b, new_a = design_bandpass_biquad(current_f0, current_f0_bandwidth, RATE)
                else:
                    new_b, new_a = None, None # No valid F0, don't update this filter

                if new_b is not None and new_a is not None:
                    # Update coefficients
                    f_data['b'] = new_b
                    f_data['a'] = new_a
                    # Re-initialize zi_state if filter order changed (for safety)
                    expected_zi_len = max(len(f_data['b']), len(f_data['a'])) - 1
                    if f_data['zi'].shape[0] != expected_zi_len:
                        f_data['zi'] = np.zeros(expected_zi_len)
                    # print(f"Debug: F0 filter updated to {current_f0:.1f}Hz.")
                # else: keep previous b, a, zi for the F0-tracking filter

                updated_f0_filter = f_data # Keep reference for plotting

    # --- Apply all filters in parallel and sum outputs ---
    summed_filtered_audio = np.zeros_like(audio_chunk_float)

    # Apply Filter
    with filter_lock: # Ensure coefficients don't change during filtering
        for f_data in active_filters:
            # Apply each filter. lfilter updates zi_state in place (if it's mutable like an array).
            # We pass the zi from the filter's data, and it updates it.
            filtered_band, f_data['zi'] = lfilter(f_data['b'], f_data['a'], audio_chunk_float, zi=f_data['zi'])
            summed_filtered_audio += filtered_band # Sum the outputs

    # Avoid clipping if summing results in values > 1.0 or < -1.0
    summed_filtered_audio = np.clip(summed_filtered_audio, -1.0, 1.0)
 
    audio_chunk_int16_filtered = np.int16((summed_filtered_audio * MAKEUP_GAIN) * 32767.0)

    return audio_chunk_int16_filtered.tobytes(), pyaudio.paContinue


# --- Matplotlib Plot Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.style.use('dark_background') # Or 'seaborn-v0_8' or 'ggplot' etc.

# Plot 1: F0 over time
x_f0_plot = np.linspace(0, F0_HISTORY_LENGTH * (CHUNK / RATE), F0_HISTORY_LENGTH)
line_f0, = ax1.plot(x_f0_plot, list(f0_history), label='Estimated F0 (Hz)', color='cyan')
ax1.set_title('Real-time Fundamental Frequency (F0) and Filter Response')
ax1.set_ylabel('F0 (Hz)')
ax1.set_ylim(FMIN_HZ - 10, FMAX_HZ + 100) # Adjust Y-axis for F0
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# Plot 2: Filter Magnitude Response
freq_axis = np.linspace(0, RATE / 2, 512)
# We'll plot all individual filter responses, and their sum
filter_lines = []
for i, f_data in enumerate(active_filters):
    color = 'lime' if f_data['type'] == 'F0_TRACKING' else 'red'
    label = f_data['type'].replace('_', ' ')
    line, = ax2.plot(freq_axis, np.zeros_like(freq_axis), label=label, color=color, linestyle='--')
    filter_lines.append(line)

composite_line, = ax2.plot(freq_axis, np.zeros_like(freq_axis), label='Composite Filter (dB)', color='white', linewidth=2)
filter_lines.append(composite_line) # Add composite line to the list for updates

# For faster updates in matplotlib
plt.ion() # Turn on interactive mode
plt.show(block=False)

# --- Animation Update Function ---
def update_plot(frame):
    global active_filters, f0_history, filter_lines

    # Update F0 plot (same as before)
    line_f0.set_ydata(list(f0_history))
    ax1.set_xlim(0, F0_HISTORY_LENGTH * (CHUNK / RATE))
    line_f0.set_xdata(np.linspace(0, F0_HISTORY_LENGTH * (CHUNK / RATE), F0_HISTORY_LENGTH))

    # Update Filter Magnitude Response plot
    with filter_lock:
        composite_h = np.zeros(512, dtype=np.complex128) # To sum complex responses
        for i, f_data in enumerate(active_filters):
            w, h = freqz(f_data['b'], f_data['a'], worN=512, fs=RATE)
            mag_db = 20 * np.log10(abs(h) + 1e-6)
            filter_lines[i].set_ydata(mag_db) # Update individual filter line
            composite_h += h # Sum the complex responses for composite

        # Update composite filter line
        composite_mag_db = 20 * np.log10(abs(composite_h) + 1e-6)
        filter_lines[-1].set_ydata(composite_mag_db)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return tuple(filter_lines) + (line_f0,) # Return all updated lines for blitting

# --- PyAudio Setup ---
p = pyaudio.PyAudio()

# Open stream with callback mode
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

print("Starting audio stream and monitor... Speak into your microphone.")
print(f"F0 filter Q: {TARGET_Q_FACTOR}. Added {len(FIXED_FORMANT_FILTERS)} fixed formant filters.")

# Start Matplotlib animation in the main thread
ani = animation.FuncAnimation(fig, update_plot, interval=1, blit=True) # interval in ms

try:
    stream.start_stream()
    # Keep the main thread alive for matplotlib to update
    while stream.is_active():
        plt.pause(0.01) # Small pause to allow GUI events to process
        time.sleep(0.01) # Sleep to prevent busy-waiting

except KeyboardInterrupt:
    print("\nStopping audio stream and monitor.")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # --- Cleanup ---
    if stream.is_active():
        stream.stop_stream()
    stream.close()
    p.terminate()
    plt.close(fig) # Close the matplotlib figure
    print("Audio streams and monitor closed.")