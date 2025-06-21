import pyaudio
import numpy as np
import librosa
from scipy.signal import iirfilter, lfilter, freqz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import threading
import time
import sys

# --- Audio Stream Parameters ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4098 # Samples per audio buffer (callback)

# --- Filter Parameters ---
TARGET_Q_FACTOR_F0 = 0.707 # Q factor for the F0-tracking filter
DEFAULT_F0_HZ = 150 # Default F0 when no voice is detected, for filter initialization/plotting
MIN_VALID_F0 = 60   # Minimum F0 to consider valid for general tracking
MAX_VALID_F0 = 500  # Maximum F0 to consider valid for general tracking
MAKEUP_GAIN = 2.0   # Overall gain applied after summing all filtered bands

# --- Pitch Estimation Parameters ---
FMIN_HZ = 50 # Lower bound for librosa.pyin search
FMAX_HZ = 600 # Upper bound for librosa.pyin search
HOP_LENGTH_F0 = CHUNK // 4 # A quarter of the chunk size for more frequent F0 estimates

# Fixed Formant Filter Parameters
FIXED_FORMANT_FILTERS = [
    {'center_freq': 800, 'bandwidth_hz': 400}, # Band 1: Covers F1 for many vowels
    {'center_freq': 2000, 'bandwidth_hz': 800}, # Band 2: Covers F2/F3 for many vowels
    {'center_freq': 3500, 'bandwidth_hz': 1000}, # Band 3: For sibilance, higher formants
]

# --- Global Variables for Filter State and Plotting Data ---
# List to hold coefficients, states, and metadata for ALL active filters
active_filters = []
# We'll plot all individual filter responses, and their summed response (with gains)
filter_lines = []
    
# Data for plotting
F0_HISTORY_LENGTH = 100 # Number of past F0 values to display
f0_history = collections.deque(np.full(F0_HISTORY_LENGTH, DEFAULT_F0_HZ), maxlen=F0_HISTORY_LENGTH)

# A lock to prevent race conditions when updating global filter coeffs and states
filter_lock = threading.Lock()

# Calibrated F0 range for user's voice (set during calibration phase)
CALIBRATED_MIN_F0 = None
CALIBRATED_MAX_F0 = None

# Initialize with unity gains (no attenuation)
current_gains = {
    'G_f0': 1.0,
    'G_f1': 1.0, # Corresponds to FORMANT_FIXED_0
    'G_f2': 1.0, # Corresponds to FORMANT_FIXED_1
    'G_f3': 1.0, # Corresponds to FORMANT_FIXED_2
}


# --- Filter Design Function ---
# This function now takes q_factor for consistency with F0 filter logic,
# but can still support bandwidth_hz externally.
def design_bandpass_biquad(center_freq, q_factor_or_bandwidth, sampling_rate, use_q_factor=True):
    """
    Designs a 2nd-order (biquad) bandpass filter.
    `q_factor_or_bandwidth` can be a Q factor or a bandwidth in Hz.
    `use_q_factor` determines interpretation of `q_factor_or_bandwidth`.
    """
    if use_q_factor:
        q_factor = q_factor_or_bandwidth
        bandwidth_hz = center_freq / q_factor if q_factor > 0 else 0
    else:
        bandwidth_hz = q_factor_or_bandwidth
        q_factor = center_freq / bandwidth_hz if bandwidth_hz > 0 else 0

    if bandwidth_hz <= 0 or q_factor <= 0:
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]) # Passthrough

    # Ensure F0 is within valid range for the F0-tracking filter, and general valid freq for fixed filters
    if not (20 <= center_freq < sampling_rate / 2 - 20):
        # print(f"Debug: Center frequency {center_freq:.1f} Hz out of valid range. Returning passthrough.")
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    nyquist = sampling_rate / 2
    low_cutoff = (center_freq - bandwidth_hz / 2) / nyquist
    high_cutoff = (center_freq + bandwidth_hz / 2) / nyquist

    if not (0 < low_cutoff < high_cutoff < 1):
        # print(f"Debug: Calculated normalized cutoffs [{low_cutoff:.4f}, {high_cutoff:.4f}] out of range. Returning passthrough.")
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    try:
        # 2nd order Butterworth bandpass
        b, a = iirfilter(2, [low_cutoff, high_cutoff], btype='bandpass', ftype='butter', output='ba')
        return b, a
    except ValueError as e:
        # print(f"Error designing filter: {e}. Returning passthrough.")
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

# --- Initialization of Filters ---
def initialize_filters():
    global active_filters
    temp_filters = [] # Use a temporary list to build

    # 1. Initialize F0-tracking filter
    # Using a default Q-factor for the F0 filter
    b_f0, a_f0 = design_bandpass_biquad(DEFAULT_F0_HZ, TARGET_Q_FACTOR_F0, RATE, use_q_factor=True)
    
    zi_f0 = np.zeros(max(len(b_f0), len(a_f0)) - 1)
    temp_filters.append({
        'type': 'F0_TRACKING',
        'b': b_f0,
        'a': a_f0,
        'zi': zi_f0,
        'gain_key': 'G_f0' # Key for its specific gain
    })

    # 2. Initialize Fixed Formant Filters
    for i, params in enumerate(FIXED_FORMANT_FILTERS):
        # For fixed formants, we use the specified bandwidth_hz
        b_fixed, a_fixed = design_bandpass_biquad(params['center_freq'], params['bandwidth_hz'], RATE, use_q_factor=False)
        
        # Ensure we successfully designed the filter before adding
        if b_fixed is not None and a_fixed is not None and len(b_fixed) > 0 and len(a_fixed) > 0:
            zi_fixed = np.zeros(max(len(b_fixed), len(a_fixed)) - 1)
            temp_filters.append({
                'type': f'FORMANT_FIXED_{i}', # e.g., FORMANT_FIXED_0, FORMANT_FIXED_1
                'b': b_fixed,
                'a': a_fixed,
                'zi': zi_fixed,
                'gain_key': f'G_f{i+1}' # e.g., G_f1, G_f2, G_f3
            })
        else:
            print(f"Warning: Failed to design fixed filter for {params['center_freq']}Hz. Skipping this filter.", file=sys.stderr)

    active_filters = temp_filters
    print(f"Initialized {len(active_filters)} filters.")

# --- F0 Calibration Function ---
def calibrate_f0_range(duration_seconds=5):
    global CALIBRATED_MIN_F0, CALIBRATED_MAX_F0
    print(f"\n--- F0 Calibration ---")
    print(f"Please speak normally for {duration_seconds} seconds to calibrate your F0 range.")
    print(f"Starting in 3 seconds...")
    time.sleep(3)
    print(f"Recording...")
    print(f"Please say 'A quick brown fox jumps over the lazy dog'.")
    
    audio_buffer = np.array([], dtype=np.float32)

    p_cal = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                audio_buffer = np.concatenate((audio_buffer, audio_chunk.astype(np.float32) / 32768.0))
            except IOError as e:
                print(f"IOError during calibration: {e}", file=sys.stderr)
                # Continue attempting to read
        
        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"Error during F0 calibration stream setup: {e}. Using default F0 range.", file=sys.stderr)
        return

    # Ensure buffer is not empty or too short for pyin
    if len(audio_buffer) == 0 or len(audio_buffer) < FMIN_HZ * 2: # Very rough minimum for pyin
        print("No audio captured during calibration or buffer too short. Using default F0 range.")
        return

    print("Analyzing F0 for calibration...")
    try:
        # pyin's frame_length and hop_length need to be carefully chosen.
        # For a full buffer, 2048/512 is common.
        f0_cal, voiced_flag_cal, _ = librosa.pyin(
            y=audio_buffer,
            sr=RATE,
            fmin=FMIN_HZ,
            fmax=FMAX_HZ,
            frame_length=2048, # Larger frame for better F0 estimation over calibration period
            hop_length=512,
            fill_na=0 # Fill NaN with 0 for easier processing
        )
    except Exception as e:
        print(f"Error during F0 calibration analysis (librosa.pyin): {e}. Using default F0 range.", file=sys.stderr)
        return

    valid_f0s = f0_cal[voiced_flag_cal == 1]
    
    if len(valid_f0s) > 0:
        # Calculate percentiles to get a robust range, ignoring extreme outliers
        CALIBRATED_MIN_F0 = np.percentile(valid_f0s, 5) # 5th percentile
        CALIBRATED_MAX_F0 = np.percentile(valid_f0s, 95) # 95th percentile

        # Add a small buffer to the range to account for natural voice variation
        # and ensure it doesn't go outside the overall FMIN/FMAX
        buffer_percent = 0.1 # 10% buffer
        CALIBRATED_MIN_F0 = max(FMIN_HZ, CALIBRATED_MIN_F0 * (1 - buffer_percent))
        CALIBRATED_MAX_F0 = min(FMAX_HZ, CALIBRATED_MAX_F0 * (1 + buffer_percent))

        print(f"Calibrated F0 range: {CALIBRATED_MIN_F0:.2f} Hz to {CALIBRATED_MAX_F0:.2f} Hz")
    else:
        print("Could not detect sufficient F0 during calibration. Using default F0 range.")
        CALIBRATED_MIN_F0 = None # Reset to use defaults
        CALIBRATED_MAX_F0 = None
        
    p_cal.terminate()
    print(f"--- Calibration Complete ---")


# --- Audio Processing Function ---
def audio_callback(in_data, frame_count, time_info, status):
    if status:
        print(status, file=sys.stderr)

    global f0_history, current_gains, active_filters

    audio_chunk_int16 = np.frombuffer(in_data, dtype=np.int16)
    audio_chunk_float = audio_chunk_int16.astype(np.float32) / 32768.0 # Normalize to -1.0 to 1.0

    # --- F0 Tracking for Current Chunk ---
    current_f0 = 0.0
    voiced_flag_for_block = False # True if a valid, calibrated F0 is found in this block
    try:
        # Ensure frame_length is suitable for chunk size.
        f0_pyin_frame, v_flag_pyin_frame, _ = librosa.pyin(
            y=audio_chunk_float,
            sr=RATE,
            fmin=FMIN_HZ,
            fmax=FMAX_HZ,
            frame_length=CHUNK, # Use the whole chunk for F0 estimation
            hop_length=HOP_LENGTH_F0, # Overlap to get more F0 estimates per chunk
            fill_na=0 # Fill NaN with 0 for easier processing
        )
        
        # Filter F0s based on voicing probability and calibrated range
        if np.any(v_flag_pyin_frame == 1):
            valid_f0s_in_block = f0_pyin_frame[v_flag_pyin_frame == 1]
            raw_current_f0_median = np.median(valid_f0s_in_block) if len(valid_f0s_in_block) > 0 else 0.0

            # Determine effective F0 range (calibrated or default)
            effective_min_f0 = CALIBRATED_MIN_F0 if CALIBRATED_MIN_F0 is not None else MIN_VALID_F0
            effective_max_f0 = CALIBRATED_MAX_F0 if CALIBRATED_MAX_F0 is not None else MAX_VALID_F0

            if not np.isnan(raw_current_f0_median) and \
               raw_current_f0_median >= effective_min_f0 and \
               raw_current_f0_median <= effective_max_f0:
                current_f0 = raw_current_f0_median
                voiced_flag_for_block = True
        
    except Exception as e:
        # print(f"F0 tracking error: {e}", file=sys.stderr) # Uncomment for debugging
        current_f0 = 0.0
        voiced_flag_for_block = False

    # Update F0 history and calculate smoothed F0
    # Add 0 if no valid F0 for current block, otherwise add current_f0
    f0_history.append(current_f0)
    
    # Calculate smoothed F0 by taking the mean of non-zero F0s in history
    # This helps stabilize the F0 filter frequency
    smoothed_f0 = np.median([f for f in f0_history if f > 0]) if any(f > 0 for f in f0_history) else 0.0

    # Update the current gains based on whether speech signal is found.
    if voiced_flag_for_block and current_f0 > 0:
        # If voiced speech is detected within calibrated range, prioritize passing
        current_gains['G_f0'] = 1.0 # High gain for F0 band
        current_gains['G_f1'] = 1.0 # High gain for Formant 1 band
        current_gains['G_f2'] = 1.0 # High gain for Formant 2 band
        current_gains['G_f3'] = 1.0 # High gain for Formant 3 band
    else:
        # No valid voice detected for this block (or F0 outside calibrated range)
        # Apply more aggressive attenuation to all speech-related bands
        current_gains['G_f0'] = 0.05 # Strong attenuation for F0 band
        current_gains['G_f1'] = 0.05
        current_gains['G_f2'] = 0.05
        current_gains['G_f3'] = 0.05

    # Ensure gains are always within 0.0 and 1.0
    for key in current_gains:
        current_gains[key] = np.clip(current_gains[key], 0.0, 1.0)

    # --- Apply all filters in parallel and sum outputs with individual gains ---
    summed_filtered_audio = np.zeros_like(audio_chunk_float)

    with filter_lock: # Ensure coefficients and states don't change during filtering
        for f_data in active_filters:
            # 1. Update filter coefficients for F0-tracking filter
            if f_data['type'] == 'F0_TRACKING':
                if smoothed_f0 > 0: # Only update if a valid (smoothed) F0 is present
                    current_f0_bandwidth_hz = smoothed_f0 / TARGET_Q_FACTOR_F0
                    f_data['b'], f_data['a'] = design_bandpass_biquad(
                        smoothed_f0, current_f0_bandwidth_hz, RATE, use_q_factor=False # Use bandwidth calc
                    )
                    # Reset zi if coefficients change. This can cause tiny clicks, but ensures stability.
                    # For perfectly smooth transitions, more advanced state management is needed.
                    f_data['zi'] = np.zeros(max(len(f_data['b']), len(f_data['a'])) - 1)
                else:
                    # If no valid F0, flatten the F0 filter response (passthrough for coefficients)
                    # The gain will then effectively mute it.
                    f_data['b'] = np.array([1.0, 0.0, 0.0])
                    f_data['a'] = np.array([1.0, 0.0, 0.0])
                    f_data['zi'] = np.zeros(max(len(f_data['b']), len(f_data['a'])) - 1)


            # 2. Apply filter to the input data
            filtered_band, f_data['zi'] = lfilter(f_data['b'], f_data['a'], audio_chunk_float, zi=f_data['zi'])

            # 3. Get individual gain and sum to the output buffer
            gain = current_gains.get(f_data['gain_key'], 1.0) # Default to 1.0 if key not found
            summed_filtered_audio += filtered_band * gain

    # Apply makeup gain and convert back to int16
    summed_filtered_audio = np.clip(summed_filtered_audio * MAKEUP_GAIN, -1.0, 1.0)
    audio_chunk_int16_filtered = np.int16(summed_filtered_audio * 32767.0)

    return audio_chunk_int16_filtered.tobytes(), pyaudio.paContinue


# --- Animation Update Function ---
def update_plot(frame):
    global f0_history, filter_lines, active_filters, current_gains

    # Update F0 plot
    line_f0.set_ydata(list(f0_history))
    ax1.set_xlim(0, F0_HISTORY_LENGTH * (CHUNK / RATE)) # Ensure X-axis updates with history length
    line_f0.set_xdata(np.linspace(0, F0_HISTORY_LENGTH * (CHUNK / RATE), F0_HISTORY_LENGTH))

    # Update Filter Magnitude Response plot
    with filter_lock:
        composite_h = np.zeros(512, dtype=np.complex128) # To sum complex responses (including gains)
        
        # Iterate over active_filters to update individual and composite lines
        for i, f_data in enumerate(active_filters):
            # Get current coefficients and gain for this filter
            b_current = f_data['b']
            a_current = f_data['a']
            gain_current = current_gains.get(f_data['gain_key'], 1.0)

            # Calculate frequency response
            w, h = freqz(b_current, a_current, worN=512, fs=RATE)
            
            # Apply gain to the individual filter response before plotting and summing
            h_gained = h * gain_current
            mag_db = 20 * np.log10(abs(h_gained) + 1e-6) # Add small epsilon to avoid log(0)

            # Update individual filter line
            filter_lines[i].set_ydata(mag_db)
            
            # Add to composite response (already gained)
            composite_h += h_gained

        # Update composite filter line
        composite_mag_db = 20 * np.log10(abs(composite_h) + 1e-6)
        filter_lines[-1].set_ydata(composite_mag_db)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Return all updated lines for blitting
    return tuple(filter_lines) + (line_f0,)

# --- Main execution block ---
#if __name__ == '__main_':
# --- PyAudio Setup ---
p = pyaudio.PyAudio()

# Initialize filters once at startup
initialize_filters()

# Perform F0 calibration for the user's voice
calibrate_f0_range(duration_seconds=5)

print("\nStarting real-time audio filter. Speak into your microphone.")
print("The system will try to filter based on your calibrated F0 range and simulated ML gains.")
print("Press Ctrl+C to stop.")

# --- Matplotlib Plot Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.style.use('dark_background')

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
for i, f_data in enumerate(active_filters):
    color = 'lime' if f_data['type'] == 'F0_TRACKING' else (
        'red' if 'FORMANT_FIXED_0' in f_data['type'] else (
        'orange' if 'FORMANT_FIXED_1' in f_data['type'] else 'yellow'
        )
    )
    label = f_data['type'].replace('_', ' ')
    line, = ax2.plot(freq_axis, np.zeros_like(freq_axis), label=label, color=color, linestyle='--')
    filter_lines.append(line)

# This line represents the *composite* (summed) filter response, including gains
composite_line, = ax2.plot(freq_axis, np.zeros_like(freq_axis), label='Composite Filter (dB)', color='white', linewidth=2)
filter_lines.append(composite_line) # Add composite line to the list for updates

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Gain (dB)')
ax2.set_xlim(0, RATE / 2)
ax2.set_ylim(-60, 10) # Typical range for filter plots
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

# For faster updates in matplotlib
plt.ion() # Turn on interactive mode
plt.show(block=False)

# Start Matplotlib animation in the main thread
ani = animation.FuncAnimation(fig, update_plot, interval=0.1, blit=True) # interval in ms

# Open stream with callback mode
try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    stream.start_stream()
    # Keep the main thread alive for matplotlib to update
    while stream.is_active():
        plt.pause(0.01) # Small pause to allow GUI events to process
        time.sleep(0.01) # Sleep to prevent busy-waiting / high CPU usage
except KeyboardInterrupt:
    print("\nStopping audio stream and monitor.")
except Exception as e:
    print(f"\nAn error occurred: {e}", file=sys.stderr)
    
finally:
    # --- Cleanup ---
    if 'stream' in locals() and stream.is_active(): # Check if stream was successfully opened and is active
        stream.stop_stream()
    if 'stream' in locals():
        stream.close()
    p.terminate()
    plt.close(fig) # Close the matplotlib figure
    print("Audio streams and monitor closed.")