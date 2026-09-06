import re

print("Reverting target to Target_Feedrate in dataset_preprocessor.py...")
with open("phase2_experimental/dataset_preprocessor.py", "r") as f:
    prep_code = f.read()

prep_code = prep_code.replace("self.target_scaler.fit(combined_df[['Duration_Sec']])", "self.target_scaler.fit(combined_df[['Target_Feedrate']])")
prep_code = prep_code.replace("df_out['Duration_Sec'] = np.log1p(np.maximum(0.0, df_out['Duration_Sec'].values))", "df_out['Target_Feedrate'] = np.log1p(np.maximum(0.0, df_out['Target_Feedrate'].values))")
prep_code = prep_code.replace("if is_training and 'Duration_Sec' in df_out.columns:", "if is_training and 'Target_Feedrate' in df_out.columns:")
prep_code = prep_code.replace("if 'Duration_Sec' in standstill_df.columns:", "if 'Target_Feedrate' in standstill_df.columns:")
prep_code = prep_code.replace("standstill_df['Duration_Sec'] = 0.0", "standstill_df['Target_Feedrate'] = 0.0")
prep_code = prep_code.replace("scaled_target = self.target_scaler.transform(df_prep[['Duration_Sec']]).astype(np.float32) if 'Duration_Sec' in df_prep.columns else None", "scaled_target = self.target_scaler.transform(df_prep[['Target_Feedrate']]).astype(np.float32) if 'Target_Feedrate' in df_prep.columns else None")

with open("phase2_experimental/dataset_preprocessor.py", "w") as f:
    f.write(prep_code)

print("Reverting inference_pipeline.py to predict feedrate...")
with open("phase2_experimental/inference_pipeline.py", "r") as f:
    infer_code = f.read()

# Replace block_durations_sec assignment and expm1 prediction
infer_code = re.sub(
    r"y_pred_log = preprocessor\.target_scaler\.inverse_transform\(y_pred_scaled\)\n\s+predicted_duration_sec = np\.maximum\(0\.0, np\.expm1\(y_pred_log\)\.flatten\(\)\)",
    "y_pred_log = preprocessor.target_scaler.inverse_transform(y_pred_scaled)\n    predicted_feedrate = np.maximum(1.0, np.expm1(y_pred_log).flatten())",
    infer_code
)

infer_code = re.sub(
    r"block_durations_sec = np\.where\(df_parsed\['Is_Motion_Block'\] == 1, predicted_duration_sec, 0\.0\)",
    "df_parsed['Predicted_Feedrate_mm_min'] = predicted_feedrate\n    block_durations_sec = np.where(df_parsed['Is_Motion_Block'] == 1, (effective_distance / predicted_feedrate) * 60.0, 0.0)",
    infer_code
)

# Remove the Phase 2 hard clipping explanation block if any
infer_code = re.sub(
    r"\s+# --- \[Fase 2\] MURNI AI PREDICTION.*?df_parsed\['Predicted_Feedrate_mm_min'\] = equivalent_feedrate",
    "",
    infer_code, flags=re.DOTALL
)

with open("phase2_experimental/inference_pipeline.py", "w") as f:
    f.write(infer_code)
