import numpy as np

# Suppose y_pred_scaled gives us some values.
# Is it possible that np.expm1 produces tiny negative numbers?
val = np.expm1(-2.398531e-06)
print("expm1(-2.398531e-06) =", val)
