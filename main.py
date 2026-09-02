"""
main.py
Orchestrator utama untuk menjalankan End-to-End Pipeline Prediksi Machining Time CNC
Menggunakan Dual-Layer Bi-LSTM.
"""

import argparse
import os
import sys
import pandas as pd
from batch_gcode_parser import NCParser
from trace_synchronizer import SinuTrainSynchronizer
from dataset_preprocessor import DatasetPreprocessor
from train_bi_lstm import run_training
from inference_pipeline import predict_nc_file

import glob

def run_training_pipeline(data_dir: str, out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output_model = os.path.join(out_dir, "bilstm_feedrate_model.keras")
    scaler_path = os.path.join(out_dir, "scaler.pkl")

    print(f"\n{'='*50}\n[MEMULAI BATCH TRAINING]\nMencari pasangan file G-Code (.mpf) dan Trace (.csv) di folder: {data_dir}\n{'='*50}")

    # Cari semua file G-Code (.mpf atau .nc)
    gcode_files = glob.glob(os.path.join(data_dir, "*.mpf")) + glob.glob(os.path.join(data_dir, "*.nc"))
    if not gcode_files:
        print("[ERROR] Tidak ditemukan file .mpf atau .nc di folder tersebut.")
        sys.exit(1)

    synced_dfs = []

    for gcode_file in gcode_files:
        base_name = os.path.splitext(os.path.basename(gcode_file))[0]
        # Cari pasangan file trace (.csv)
        trace_file = os.path.join(data_dir, f"{base_name}.csv")

        if not os.path.exists(trace_file):
            print(f"[WARNING] Melewati {base_name}: Tidak ditemukan file trace pasangannya ({trace_file})")
            continue

        print(f"\n--- Memproses Pasangan: {base_name} ---")

        print("[TAHAP 1] Parsing NC & Geometri 3D...")
        parser = NCParser()
        df_parsed = parser.parse_file(gcode_file)

        print("[TAHAP 2] Sinkronisasi Trace SinuTrain...")
        df_trace = pd.read_csv(trace_file)
        syncer = SinuTrainSynchronizer()
        df_trace_clean = syncer.clean_and_attribute_trace(df_trace, df_parsed['Block_ID'].tolist())
        df_synced = syncer.match_and_calculate_targets(df_parsed, df_trace_clean)

        synced_filename = f"{base_name}_synced.csv"
        synced_output = os.path.join(out_dir, synced_filename)
        df_synced.to_csv(synced_output, index=False)
        print(f"-> Tersinkronisasi ({len(df_synced)} baris), disimpan ke: {synced_output}")

        synced_dfs.append(df_synced)

    if not synced_dfs:
        print("\n[ERROR] Tidak ada satupun pasangan file yang berhasil disinkronisasi.")
        sys.exit(1)

    print(f"\n{'='*50}\n[TAHAP 3] Scaling, Padding & Sequence Windowing (Batch)\n{'='*50}")
    preprocessor = DatasetPreprocessor(window_size=201)

    # Preprocessor menerima list dataframe dari berbagai file untuk di-fit scaler dan diubah ke window 3D
    X_all, Y_all = preprocessor.fit_transform_dataset(synced_dfs)
    preprocessor.save_scalers(scaler_path)
    print(f"Scaler parameters saved to {scaler_path}")

    split_idx = int(0.8 * len(X_all))
    X_train, Y_train = X_all[:split_idx], Y_all[:split_idx]
    X_val, Y_val = X_all[split_idx:], Y_all[split_idx:]
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

    print(f"\n{'='*50}\n[TAHAP 4] Pelatihan Model Dual-Layer Bi-LSTM\n{'='*50}")
    model, history = run_training(X_train, Y_train, X_val, Y_val, model_save_path=output_model)
    print(f"Model berhasil dilatih dan disimpan di: {output_model}")
    print("End-to-End Training Pipeline selesai.\n")

def run_inference_pipeline(data_dir: str, out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_path = os.path.join(out_dir, "bilstm_feedrate_model.keras")
    scaler_path = os.path.join(out_dir, "scaler.pkl")

    print(f"\n{'='*50}\n[TAHAP 5] Standalone Inference (Batch)\n{'='*50}")

    # Cari semua file G-Code (.mpf atau .nc) di folder test
    gcode_files = glob.glob(os.path.join(data_dir, "*.mpf")) + glob.glob(os.path.join(data_dir, "*.nc"))
    if not gcode_files:
        print(f"[ERROR] Tidak ditemukan file .mpf atau .nc di folder: {data_dir}")
        sys.exit(1)

    print(f"Ditemukan {len(gcode_files)} file program untuk diinferensi.")

    for idx, gcode_file in enumerate(gcode_files):
        print(f"\n[{idx+1}/{len(gcode_files)}] Memproses Prediksi: {os.path.basename(gcode_file)}")
        predict_nc_file(gcode_file, model_path=model_path, scaler_path=scaler_path, out_dir=out_dir)

    print("\n[INFO] Seluruh proses inferensi batch selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Digital Twin CNC Bi-LSTM")

    subparsers = parser.add_subparsers(dest="mode", help="Mode eksekusi: train atau infer")

    # Subparser untuk mode TRAINING
    train_parser = subparsers.add_parser("train", help="Jalankan Pipeline Pelatihan End-to-End (Batch)")
    train_parser.add_argument("--data-dir", type=str, required=True, help="Path folder data mentah (isi .mpf dan .csv)")
    train_parser.add_argument("--out-dir", type=str, default="output", help="Path folder hasil pipeline (default: output)")

    # Subparser untuk mode INFERENCE
    infer_parser = subparsers.add_parser("infer", help="Jalankan Pipeline Prediksi Standalone (Batch)")
    infer_parser.add_argument("--data-dir", type=str, required=True, help="Path folder G-Code baru (.mpf) untuk testing")
    infer_parser.add_argument("--out-dir", type=str, default="output", help="Path folder berisi model/scaler (dan tempat menyimpan hasil prediksi)")

    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.exists(args.data_dir) or not os.path.isdir(args.data_dir):
            print("[ERROR] Pastikan argumen --data-dir adalah folder yang valid.")
            sys.exit(1)
        run_training_pipeline(args.data_dir, args.out_dir)

    elif args.mode == "infer":
        if not os.path.exists(args.data_dir) or not os.path.isdir(args.data_dir):
            print("[ERROR] Pastikan argumen --data-dir adalah folder yang valid.")
            sys.exit(1)

        model_path = os.path.join(args.out_dir, "bilstm_feedrate_model.keras")
        scaler_path = os.path.join(args.out_dir, "scaler.pkl")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"[ERROR] Model atau Scaler tidak ditemukan di {args.out_dir}. Lakukan train terlebih dahulu.")
            sys.exit(1)

        run_inference_pipeline(args.data_dir, args.out_dir)

    else:
        parser.print_help()
