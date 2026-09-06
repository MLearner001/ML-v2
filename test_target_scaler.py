import numpy as np

# Phase 2 targets are Duration_Sec
# We apply log1p(np.maximum(0.0, df_out['Duration_Sec'].values))
# Then we fit Standard Scaler to it.

import pickle

# Wait, Phase 2 dataset_preprocessor.py creates the scaler.pkl when we run train.
# Let's check how Phase 2 is actually written.
