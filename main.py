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

def run_training_pipeline(gcode_file: str, trace_file: str, output_model: str, scaler_path: str):
    print(f"\n{'='*50}\n[TAHAP 1] Parsing NC & Geometri 3D\n{'='*50}")
    parser = NCParser()
    df_parsed = parser.parse_file(gcode_file)
    print(f"Berhasil mengekstrak {len(df_parsed)} baris.")

    print(f"\n{'='*50}\n[TAHAP 2] Sinkronisasi Trace SinuTrain 4ms\n{'='*50}")
    df_trace = pd.read_csv(trace_file)
    syncer = SinuTrainSynchronizer()
    df_trace_clean = syncer.clean_and_attribute_trace(df_trace, df_parsed['Block_ID'].tolist())
    df_synced = syncer.match_and_calculate_targets(df_parsed, df_trace_clean)

    synced_output = gcode_file.replace('.mpf', '_synced.csv')
    df_synced.to_csv(synced_output, index=False)
    print(f"Dataset disinkronisasi dan disimpan ke: {synced_output}")

    print(f"\n{'='*50}\n[TAHAP 3] Scaling, Padding & Sequence Windowing\n{'='*50}")
    preprocessor = DatasetPreprocessor(window_size=201)

    # Untuk contoh ini kita membagi data train dan val dari satu file secara sederhana (80/20)
    # Di pipeline produksi, df_list biasanya terdiri dari beberapa file tersinkronisasi.
    X_all, Y_all = preprocessor.fit_transform_dataset([df_synced])
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

def run_inference_pipeline(gcode_file: str, model_path: str, scaler_path: str):
    print(f"\n{'='*50}\n[TAHAP 5] Standalone Inference\n{'='*50}")
    predict_nc_file(gcode_file, model_path=model_path, scaler_path=scaler_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Digital Twin CNC Bi-LSTM")

    subparsers = parser.add_subparsers(dest="mode", help="Mode eksekusi: train atau infer")

    # Subparser untuk mode TRAINING
    train_parser = subparsers.add_parser("train", help="Jalankan Pipeline Pelatihan End-to-End")
    train_parser.add_argument("--gcode", type=str, required=True, help="Path ke file G-Code (.mpf)")
    train_parser.add_argument("--trace", type=str, required=True, help="Path ke file trace SinuTrain (.csv)")
    train_parser.add_argument("--model-out", type=str, default="bilstm_feedrate_model.keras", help="Output model (.keras)")
    train_parser.add_argument("--scaler-out", type=str, default="scaler.pkl", help="Output scaler (.pkl)")

    # Subparser untuk mode INFERENCE
    infer_parser = subparsers.add_parser("infer", help="Jalankan Pipeline Prediksi Standalone")
    infer_parser.add_argument("--gcode", type=str, required=True, help="Path ke file G-Code (.mpf)")
    infer_parser.add_argument("--model", type=str, default="bilstm_feedrate_model.keras", help="Path model Keras (.keras)")
    infer_parser.add_argument("--scaler", type=str, default="scaler.pkl", help="Path scaler (.pkl)")

    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.exists(args.gcode) or not os.path.exists(args.trace):
            print("[ERROR] Pastikan file G-code dan file Trace valid/ada.")
            sys.exit(1)
        run_training_pipeline(args.gcode, args.trace, args.model_out, args.scaler_out)

    elif args.mode == "infer":
        if not os.path.exists(args.gcode):
            print("[ERROR] File G-code tidak ditemukan.")
            sys.exit(1)
        if not os.path.exists(args.model) or not os.path.exists(args.scaler):
            print(f"[ERROR] Model ({args.model}) atau Scaler ({args.scaler}) tidak ditemukan. Lakukan train terlebih dahulu.")
            sys.exit(1)
        run_inference_pipeline(args.gcode, args.model, args.scaler)

    else:
        parser.print_help()
