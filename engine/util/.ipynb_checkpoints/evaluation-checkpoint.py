import numpy as np


def geweke(trace, intervals=20, length=0.5, first=0.1):
    n_samples = len(trace)

    # Define the start and end indices for the early segment
    first_segment_end = int(first * n_samples)
    
    # Define the minimum length for the segments
    min_len = int(length * n_samples)

    z_scores = []

    # Early segment mean and variance
    first_segment_mean = np.mean(trace[:first_segment_end])
    first_segment_var = np.var(trace[:first_segment_end])

    for segment in range(intervals):
        # Starting index for the late segment
        start = n_samples - (segment + 1) * min_len
        # Late segment mean and variance
        late_segment_mean = np.mean(trace[start:])
        late_segment_var = np.var(trace[start:])

        # Geweke Z-score
        z_score = (first_segment_mean - late_segment_mean) / np.sqrt(first_segment_var + late_segment_var)
        z_scores.append(z_score)

    return z_scores