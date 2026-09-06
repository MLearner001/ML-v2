import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
orig_durations = np.array([[0.004], [0.008], [0.1], [1.5], [10.0]])
log_durations = np.log1p(np.maximum(0.0, orig_durations))
scaler.fit(log_durations)

with open('dummy_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('dummy_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

scaled = loaded_scaler.transform(log_durations)

# Now imagine a slight negative prediction (common with NN outputs around 0)
# Wait! NN outputs scaled values (mean 0, std 1).
# If the prediction is -2 or something, inverse_transform will get a negative number.
pred_scaled = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])
inv_pred = loaded_scaler.inverse_transform(pred_scaled)
rec_pred = np.expm1(inv_pred)
print("Scaled input:\n", pred_scaled.flatten())
print("Inverse standard scaled (log1p space):\n", inv_pred.flatten())
print("expm1 reconstructed (seconds):\n", rec_pred.flatten())
