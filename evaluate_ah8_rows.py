import pandas as pd
import sys
from batch_gcode_parser import NCParser

def analyze_trace_vs_gcode(mpf_file, csv_file):
    # Parse GCode
    print(f"Parsing {mpf_file}...")
    parser = NCParser()
    df_gcode = parser.parse_file(mpf_file)
    print(f"GCode DataFrame Rows: {len(df_gcode)}")
    print(f"GCode Unique Block_IDs: {df_gcode['Block_ID'].nunique()}")
    print(f"GCode Unique N_Numbers (excluding -1): {df_gcode[df_gcode['N_Number'] != -1]['N_Number'].nunique()}")
    print(f"GCode Max N_Number: {df_gcode['N_Number'].max()}")

    # Check max absolute_idx in gcode
    print(f"GCode Max Block_ID (absolute_idx): {df_gcode['Block_ID'].astype(int).max()}")

    # Read Trace
    print(f"\nReading {csv_file}...")
    try:
        df_trace = pd.read_csv(csv_file, sep=';', on_bad_lines='skip', low_memory=False)
        print(f"Trace Raw DataFrame Rows: {len(df_trace)}")
    except Exception as e:
        print(f"File reading error, looking for other files...: {e}")
        return

    # Try to find actLineNumber
    def find_column_by_substrings(df, substrings):
        for col in df.columns:
            if any(sub in col for sub in substrings):
                return col
        return None

    df_trace.columns = df_trace.columns.str.strip()
    col_line = find_column_by_substrings(df_trace, ['actLineNumber', 'f1\\s1', 'f1/s1', 'f1', 's1'])

    print(f"Detected actLineNumber Column: {col_line}")
    if col_line:
        # Convert to numeric, handle errors
        act_lines = pd.to_numeric(df_trace[col_line], errors='coerce').dropna().astype(int)

        print(f"Trace Rows with valid actLineNumber: {len(act_lines)}")

        # Output info about the N numbers in trace
        print(f"\nTrace N Numbers Statistics:")
        print(f"Min: {act_lines.min()}")
        print(f"Max: {act_lines.max()}")
        print(f"Unique Valid N Numbers in Trace: {act_lines[act_lines > 0].nunique()}")

        trace_unique_n = set(act_lines[act_lines > 0].unique())
        gcode_unique_n = set(df_gcode[df_gcode['N_Number'] > 0]['N_Number'].unique())

        intersection = trace_unique_n.intersection(gcode_unique_n)
        print(f"\nIntersection of N Numbers between Trace and GCode: {len(intersection)}")
        print(f"N Numbers in Trace but NOT in GCode: {len(trace_unique_n - gcode_unique_n)}")
        print(f"N Numbers in GCode but NOT in Trace: {len(gcode_unique_n - trace_unique_n)}")

if __name__ == "__main__":
    analyze_trace_vs_gcode("phase2_experimental/data/ah8/60_c1f_sti28_ah8.mpf", "phase2_experimental/data/ah8/60_c1f_sti28_ah8.csv")
