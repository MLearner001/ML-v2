import numpy as np
from sklearn.preprocessing import StandardScaler

# Replicate the behavior in `dataset_preprocessor.py` for target scaling
scaler = StandardScaler()
# Mock data (Duration_Sec values in seconds)
orig_durations = np.array([[0.004], [0.008], [0.1], [1.5], [10.0]])
# Transformation as per `_apply_log_transforms`
log_durations = np.log1p(np.maximum(0.0, orig_durations))
# Fit
scaler.fit(log_durations)

# Transform
scaled_durations = scaler.transform(log_durations)

# Inverse (Simulate `inference_pipeline.py`)
inv_scaled = scaler.inverse_transform(scaled_durations)
reconstructed = np.expm1(inv_scaled)

print("Original:\n", orig_durations.flatten())
print("Reconstructed:\n", reconstructed.flatten())
