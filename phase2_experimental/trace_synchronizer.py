"""
trace_synchronizer.py

Tahap 2: Sinkronisasi data trace 4ms SinuTrain dengan G-code hasil parsing Tahap 1.
Menerapkan Harmonic Target, Distance-Weighted Interpolation, dan penanganan transisi CYCLE800.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class SinuTrainSynchronizer:
    def __init__(self, sample_interval_sec: float = 0.004):
        self.dt = sample_interval_sec  # 4ms = 0.004 s

    def clean_and_attribute_trace(self, df_trace: pd.DataFrame, gcode_blocks: List[str]) -> pd.DataFrame:
        """
        Membersihkan trace dan mengatribusikan block number negatif (CYCLE800 swiveling).
        """
        df = df_trace.copy()

        # Standarisasi nama kolom trace SinuTrain jika diperlukan
        # Bersihkan spasi kosong di kolom jika belum
        df.columns = df.columns.str.strip()

        # Kolom utama: actLineNumber, f2/s2 (X), f3/s3 (Y), f4/s4 (Z), f5/s5 (B), f6/s6 (C)
        # Pada beberapa export, nama kolom mengandung path panjang seperti '/Channel/!SPARP/actLineNumber [u1  1]'

        def find_column_by_substrings(substrings: List[str]) -> str:
            for col in df.columns:
                if any(sub in col for sub in substrings):
                    return col
            return None

        # Temukan kolom actLineNumber
        col_line = find_column_by_substrings(['actLineNumber', 'f1\\s1', 'f1/s1'])
        if not col_line:
            # Fallback agresif
            col_line = find_column_by_substrings(['f1', 's1'])
            if not col_line:
                raise KeyError(f"Kolom Line Number tidak ditemukan di file trace! Kolom yang tersedia: {list(df.columns)}")
        df.rename(columns={col_line: 'actLineNumber'}, inplace=True)

        # Temukan kolom f5/s5 (B) dan f6/s6 (C)
        # Pada file Siemens .csv raw sering digunakan backslash
        col_b = find_column_by_substrings(['f5\\s5', 'f5/s5', 'f5', 'actToolBasePos[3]'])
        col_c = find_column_by_substrings(['f6\\s6', 'f6/s6', 'f6', 'actToolBasePos[4]'])

        # Hitung diff posisi B dan C (Numerical Position Differentiation)
        if col_b and col_b in df.columns:
            delta_b = pd.to_numeric(df[col_b], errors='coerce').diff().abs().fillna(0)
        else:
            delta_b = pd.Series(0, index=df.index)

        if col_c and col_c in df.columns:
            delta_c = pd.to_numeric(df[col_c], errors='coerce').diff().abs().fillna(0)
        else:
            delta_c = pd.Series(0, index=df.index)

        is_rotary_moving = (delta_b > 1e-4) | (delta_c > 1e-4)

        mapped_blocks = []

        # SinuTrain actLineNumber corresponds perfectly to G-Code N_Number
        last_valid_n_number = None

        for idx, row in df.iterrows():
            try:
                # Handle possible NaN / empty string lines
                raw_line = int(float(row['actLineNumber']))
            except (ValueError, TypeError):
                mapped_blocks.append("IDLE")
                continue

            rot_moving = is_rotary_moving.iloc[idx]

            if raw_line > 0:
                # G-Code line explicitly mapped
                mapped_blocks.append(str(raw_line))
                last_valid_n_number = str(raw_line)

            elif raw_line < 0 and rot_moving:
                # Transisi CYCLE800 / Orientasi Bidang: Atribusikan ke blok parent CYCLE800 (the positive N_number)
                mapped_blocks.append(f"C800_{last_valid_n_number}" if last_valid_n_number else "INIT_IDLE")
            else:
                # Idle tanpa pergerakan signifikan
                mapped_blocks.append("IDLE")

        df['mapped_block'] = mapped_blocks

        # Buang baris trace yang tergolong IDLE murni (tidak ada eksekusi program benda kerja)
        df_valid = df[~df['mapped_block'].isin(["IDLE", "INIT_IDLE"])].copy()
        return df_valid

    def match_and_calculate_targets(self, df_parsed_gcode: pd.DataFrame, df_trace_valid: pd.DataFrame) -> pd.DataFrame:
        """
        Menghubungkan trace per blok dan menghitung Target Feedrate Harmonik (Y)
        serta menangani micro-blocks sub-4ms via Distance-Weighted Spatial Interpolation.
        """
        df_gcode = df_parsed_gcode.copy()

        # 1. Hitung jumlah tick dan rata-rata kecepatan terukur dari trace
        trace_counts = df_trace_valid['mapped_block'].value_counts().to_dict()

        # 2. Identifikasi blok yang tereksekusi langsung vs micro-blocks yang terlewati
        durations = []
        target_feedrates = []

        i = 0
        n_blocks = len(df_gcode)

        while i < n_blocks:
            row = df_gcode.iloc[i]

            # Identify the synchronization key using N_Number instead of Block_ID
            # If N_Number is -1 (missing N line), it won't be explicitly in the trace counts
            # and will fall into the cluster interpolation logic.
            n_number = str(int(row['N_Number'])) if row['N_Number'] != -1 else "-1"

            # Determine if this row is a CYCLE800 transition
            if row.get('Is_Cycle800', 0) == 1:
                sync_key = f"C800_{n_number}"
            else:
                sync_key = n_number

            delta_3d = row['Delta_3D']
            delta_rot = row['Delta_Rot']

            # Hitung jarak ekuivalen (translasi mm atau rotasi deg)
            dist = delta_3d if delta_3d > 1e-4 else delta_rot

            # Cek apakah N_Number ini tercatat di trace dan KITA PERTAMA KALI memprosesnya
            # Note: Multiple G-Code actions can have the SAME N_Number (e.g. MCALL Expansion).
            # To prevent double-counting the trace time for each sub-block, we must group them.

            # Start clustering
            cluster_indices = [i]
            j = i + 1

            # Find all subsequent rows that share this exact SAME N_Number (sub-blocks)
            # OR blocks that have NO N_Number (-1) which fall under the same execution window
            while j < n_blocks:
                next_n = str(int(df_gcode.iloc[j]['N_Number'])) if df_gcode.iloc[j]['N_Number'] != -1 else "-1"
                # Group if it's the exact same explicit N_number, or if it's an N-less block
                if next_n == n_number or next_n == "-1":
                    cluster_indices.append(j)
                    j += 1
                else:
                    # Next explicit N_number found, break cluster
                    break

            # Now we look at the ticks assigned to this entire N_Number cluster
            ticks = trace_counts.get(sync_key, 0)

            if ticks > 0:
                # KASUS A: Cluster ini tereksekusi dan tercatat di trace
                cluster_dt = ticks * self.dt
            else:
                # KASUS B: Micro-blocks Kluster yang sama sekali melompati trace (0 ticks)
                # Jendela waktu kluster ini minimal dialokasikan 1 interval (4ms)
                cluster_dt = 1.0 * self.dt

            # Distance-Weighted Spatial Interpolation di dalam kluster
            cluster_dists = [
                df_gcode.iloc[k]['Delta_3D'] if df_gcode.iloc[k]['Delta_3D'] > 1e-4 else df_gcode.iloc[k]['Delta_Rot']
                for k in cluster_indices
            ]
            total_cluster_dist = sum(cluster_dists)

            if total_cluster_dist > 1e-6:
                f_group = (total_cluster_dist / cluster_dt) * 60.0
                for k, d in zip(cluster_indices, cluster_dists):
                    weight = d / total_cluster_dist
                    t_sub = weight * cluster_dt
                    durations.append(t_sub)
                    target_feedrates.append(f_group)
            else:
                # Gerakan diam tapi memakan waktu (misal Dwell atau eksekusi logika)
                for k in cluster_indices:
                    t_sub = cluster_dt / len(cluster_indices)
                    durations.append(t_sub)
                    # Jika diam, target feedrate fallback ke Command
                    target_feedrates.append(df_gcode.iloc[k]['Cmd_F'])

            i = j  # Lompat ke blok / N_Number berikutnya

        df_gcode['Duration_Sec'] = durations
        df_gcode['Target_Feedrate'] = target_feedrates

        return df_gcode


if __name__ == "__main__":
    # Contoh verifikasi modul
    syncer = SinuTrainSynchronizer()
    print("[INFO] Trace Synchronizer Module siap digunakan.")
