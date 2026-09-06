import numpy as np
from sklearn.preprocessing import StandardScaler

target_scaler = StandardScaler()
# Let's say Duration_Sec goes from 0.0 to 10.0 seconds
durations = np.random.uniform(0.1, 10.0, size=(100, 1))
durations_log = np.log1p(durations)
target_scaler.fit(durations_log)

# Predict standard scaled values
scaled = target_scaler.transform(durations_log)
print("Mean:", np.mean(scaled), "Std:", np.std(scaled))

# Inverse transform
inv_log = target_scaler.inverse_transform(scaled)
orig = np.expm1(inv_log)

# print error
print("Max error:", np.max(np.abs(orig - durations)))
