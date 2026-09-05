"""
inference_pipeline.py
Tahap 5: Eksekusi Inferensi Standalone pada File Program NC Baru (.mpf).
(Updated: V2 dengan Physics-Informed Hard-Clipping)
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from batch_gcode_parser import NCParser
from dataset_preprocessor import DatasetPreprocessor
from typing import Dict

import os

def predict_nc_file(mpf_filepath: str,
                     model_path: str = "bilstm_feedrate_model.keras",
                     scaler_path: str = "scaler.pkl",
                     out_dir: str = ".") -> Dict[str, float]:
    """Memproses file .mpf baru dan menghitung estimasi waktu pemesinan total."""

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"[INFO] 1. Mem-parsing file NC: {mpf_filepath}")
    parser = NCParser()
    df_parsed = parser.parse_file(mpf_filepath)

    print(f"[INFO] 2. Memuat Scaler & Menyiapkan Tensor Jendela W=201...")
    preprocessor = DatasetPreprocessor(window_size=201)
    preprocessor.load_scalers(scaler_path)

    X_windows = preprocessor.transform_file(df_parsed, is_training=False)

    print(f"[INFO] 3. Memuat Model Bi-LSTM & Menjalankan Inferensi...")
    model = tf.keras.models.load_model(model_path, compile=False)
    y_pred_scaled = model.predict(X_windows, batch_size=256, verbose=0)

    # 4. Inverse Transform untuk Mendapatkan Waktu Aktual (Detik)
    y_pred_log = preprocessor.target_scaler.inverse_transform(y_pred_scaled)
    predicted_duration_sec = np.expm1(y_pred_log).flatten()

    # 5. Integrasi Kinematika Fisik & Pengecekan Validitas
    # Gunakan Delta_3D untuk pergerakan linier atau Delta_Rot untuk pergerakan putar
    effective_distance = np.where(df_parsed['Delta_3D'] > 1e-4,
                                  df_parsed['Delta_3D'],
                                  df_parsed['Delta_Rot'])

    # Jika blok non-motion (Delta=0), beri durasi 0 agar aman
    block_durations_sec = np.where(df_parsed['Is_Motion_Block'] == 1, predicted_duration_sec, 0.0)

    # --- [Fase 2] Menghitung ulang Feedrate yang setara (Opsional untuk pelaporan CSV) ---
    # F = (Jarak / Waktu) * 60
    # Hindari ZeroDivision jika model meprediksi waktu 0 detik pada blok gerak
    safe_durations = np.maximum(block_durations_sec, 1e-6)
    equivalent_feedrate = (effective_distance / safe_durations) * 60.0
    # Jika itu blok non-motion, set feedrate laporan jadi 0
    equivalent_feedrate = np.where(df_parsed['Is_Motion_Block'] == 1, equivalent_feedrate, 0.0)

    # --- Clipping pelaporan agar feedrate laporan tidak melebihi kecepatan limit G-code/Mesin ---
    limit_f = df_parsed['Cmd_F'].values
    equivalent_feedrate = np.minimum(equivalent_feedrate, limit_f)

    df_parsed['Predicted_Feedrate_mm_min'] = equivalent_feedrate
    df_parsed['Estimated_Duration_Sec'] = block_durations_sec

    total_time_sec = float(np.sum(block_durations_sec))
    total_time_min = total_time_sec / 60.0

    # Simpan hasil analisis profil feedrate ke CSV
    base_name = os.path.basename(mpf_filepath)
    output_filename = base_name.replace(".mpf", "_predicted_profile.csv").replace(".nc", "_predicted_profile.csv")
    output_csv = os.path.join(out_dir, output_filename)
    df_parsed.to_csv(output_csv, index=False)

    print("\n" + "="*50)
    print("HASIL PREDIKSI MACHINING TIME (DIGITAL TWIN PHASE 2 - V3)")
    print("="*50)
    print(f"Total Blok Program     : {len(df_parsed)} baris")
    print(f"Total Estimasi Waktu   : {total_time_sec:.2f} detik ({total_time_min:.2f} menit)")
    print(f"Profil Lengkap Disimpan: {output_csv}")
    print("="*50 + "\n")

    return {
        "total_seconds": total_time_sec,
        "total_minutes": total_time_min,
        "block_count": len(df_parsed)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_nc_file(sys.argv[1])
    else:
        print("[USAGE] python inference_pipeline.py <path_file_nc.mpf>")
