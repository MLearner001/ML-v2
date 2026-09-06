import pandas as pd
from trace_synchronizer import SinuTrainSynchronizer
from batch_gcode_parser import NCParser
import os

mpf_path = "phase2_experimental/data/ah8/60_c1f_sti28_ah8.mpf"
csv_path = "phase2_experimental/data/ah8/60_c1f_sti28_ah8.csv"

if os.path.exists(mpf_path) and os.path.exists(csv_path):
    print("Files found, testing synchronizer...")
    parser = NCParser()
    df_gcode = parser.parse_file(mpf_path)

    df_trace = pd.read_csv(csv_path, sep=';', on_bad_lines='skip', low_memory=False)

    syncer = SinuTrainSynchronizer()
    df_trace_valid = syncer.clean_and_attribute_trace(df_trace, None)

    df_sync = syncer.match_and_calculate_targets(df_gcode, df_trace_valid)

    print(f"Total G-Code lines processed: {len(df_sync)}")
    print(f"Total Duration (Sec) Extracted: {df_sync['Duration_Sec'].sum()}")
    print("Top 10 longest durations:")
    print(df_sync.sort_values('Duration_Sec', ascending=False)[['N_Number', 'Block_ID', 'Duration_Sec', 'Delta_3D', 'Cmd_F']].head(10))
    print("Success.")
else:
    print("Test files not available locally, skipping strict test.")
