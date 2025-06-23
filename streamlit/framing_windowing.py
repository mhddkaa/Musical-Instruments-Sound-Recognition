import numpy as np
from scipy.signal.windows import hamming

def framing_windowing(signal, sample_rate=44100, frame_duration=0.025, hop_duration=0.01):
    frame_length = int(frame_duration * sample_rate)
    hop_length = int(hop_duration * sample_rate)
    num_frames = 1 + int((len(signal) - frame_length) / hop_length)
    frames = np.zeros((num_frames, frame_length))
    window = hamming(frame_length, sym=False)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = signal[start:start + frame_length] * window
    return frames
