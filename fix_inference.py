import re

with open('phase2_experimental/inference_pipeline.py', 'r') as f:
    code = f.read()

# Replace block_durations_sec assignment to clamp negative values to 0.0 before creating safe_durations,
# or simply replace `np.expm1(y_pred_log).flatten()` with `np.maximum(0.0, np.expm1(y_pred_log).flatten())`
code = code.replace('predicted_duration_sec = np.expm1(y_pred_log).flatten()',
                    'predicted_duration_sec = np.maximum(0.0, np.expm1(y_pred_log).flatten())')

with open('phase2_experimental/inference_pipeline.py', 'w') as f:
    f.write(code)
