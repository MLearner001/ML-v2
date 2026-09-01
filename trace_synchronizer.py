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
        # Kolom utama: actLineNumber (atau f1/s1), f2/s2 (X), f3/s3 (Y), f4/s4 (Z), f5/s5 (B), f6/s6 (C), acVactB, acVactC
        if 'actLineNumber' not in df.columns and 'f1/s1' in df.columns:
            df.rename(columns={'f1/s1': 'actLineNumber'}, inplace=True)
        if 'acVactB' not in df.columns and 'f5/s5' in df.columns: # fallback jika kecepatan diestimasi
            pass # We strictly rely on acVactB/acVactC or calculate it if missing, but let's assume it exists or use f5/s5 diff

        mapped_blocks = []
        last_valid_block = None

        for idx, row in df.iterrows():
            raw_line = int(row['actLineNumber'])
            
            # Cek apakah ada pergerakan rotasi aktif (sumbu B/C)
            rot_velocity = 0.0
            if 'acVactB' in row and 'acVactC' in row:
                rot_velocity = abs(float(row['acVactB'])) + abs(float(row['acVactC']))

            if raw_line > 0:
                last_valid_block = str(raw_line)
                mapped_blocks.append(str(raw_line))
            elif raw_line < 0 and rot_velocity > 1e-3:
                # Transisi CYCLE800 / Orientasi Bidang: Atribusikan ke blok aktif berikutnya / blok terakhir
                mapped_blocks.append(f"C800_{last_valid_block}" if last_valid_block else "INIT_IDLE")
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
            block_id = str(row['Block_ID'])
            delta_3d = row['Delta_3D']
            delta_rot = row['Delta_Rot']

            # Hitung jarak ekuivalen (translasi mm atau rotasi deg)
            dist = delta_3d if delta_3d > 1e-4 else delta_rot

            # Cek apakah blok ini tercatat di trace
            ticks = trace_counts.get(block_id, 0)

            if ticks > 0:
                # KASUS A: Blok Standar (Tercatat 1 atau lebih ticks di trace)
                t_exec = ticks * self.dt  # Durasi dalam detik
                # Harmonic Target Formula: Y = (Jarak / Waktu) * 60 (mm/min atau deg/min)
                f_target = (dist / t_exec) * 60.0 if t_exec > 0 and dist > 0 else row['Cmd_F']
                
                durations.append(t_exec)
                target_feedrates.append(f_target)
                i += 1
            else:
                # KASUS B: Micro-blocks Kluster (Beberapa baris tereksekusi dalam 1 jendela sampling 4ms)
                # Kumpulkan seluruh micro-blocks berturutan hingga menemukan blok yang tercatat di trace
                cluster_indices = [i]
                j = i + 1
                while j < n_blocks and trace_counts.get(str(df_gcode.iloc[j]['Block_ID']), 0) == 0:
                    cluster_indices.append(j)
                    j += 1
                
                # Jendela waktu kluster ini minimal dialokasikan 1 interval (4ms) atau dialokasikan proporsional
                cluster_ticks = 1.0  # Jendela 4ms
                cluster_dt = cluster_ticks * self.dt
                
                cluster_dists = [
                    df_gcode.iloc[k]['Delta_3D'] if df_gcode.iloc[k]['Delta_3D'] > 1e-4 else df_gcode.iloc[k]['Delta_Rot']
                    for k in cluster_indices
                ]
                total_cluster_dist = sum(cluster_dists)

                # Distance-Weighted Spatial Interpolation
                if total_cluster_dist > 1e-6:
                    f_group = (total_cluster_dist / cluster_dt) * 60.0
                    for k, d in zip(cluster_indices, cluster_dists):
                        weight = d / total_cluster_dist
                        t_sub = weight * cluster_dt
                        durations.append(t_sub)
                        target_feedrates.append(f_group)
                else:
                    for _ in cluster_indices:
                        durations.append(cluster_dt / len(cluster_indices))
                        target_feedrates.append(row['Cmd_F'])

                i = j  # Lompat ke blok berikutnya

        df_gcode['Duration_Sec'] = durations
        df_gcode['Target_Feedrate'] = target_feedrates

        return df_gcode


if __name__ == "__main__":
    # Contoh verifikasi modul
    syncer = SinuTrainSynchronizer()
    print("[INFO] Trace Synchronizer Module siap digunakan.")
