"""
dataset_preprocessor.py
Tahap 3: Penskalaan Fitur Hybrid, Boundary Padding, dan Sliding Window Generator W=201.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict

class DatasetPreprocessor:
    def __init__(self, window_size: int = 201):
        self.window_size = window_size
        self.half_w = (window_size - 1) // 2  # 100 blocks
        
        # Inisialisasi Scaler
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.feature_cols = [
            'Cmd_F', 'Cmd_S', 'Is_G01', 'Is_G02', 'Is_G03', 'Is_Traori',
            'Is_Cycle800', 'Is_MCALL_Sub', 'C832_Tol', 'C832_Mode',
            'Delta_3D', 'Delta_Rot', 'Tool_Vector_Delta', 'Sharpness_Angle',
            'Is_Motion_Block', 'Is_Reversal_X', 'Is_Reversal_Y', 'Is_Reversal_Z'
        ]

    def _apply_log_transforms(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df_out = df.copy()
        # Kompresi logaritmik untuk meredam rentang ekstrem
        df_out['Delta_3D'] = np.log1p(np.maximum(0.0, df_out['Delta_3D'].values))
        df_out['Cmd_F'] = np.log1p(np.maximum(0.0, df_out['Cmd_F'].values))
        df_out['Sharpness_Angle'] = df_out['Sharpness_Angle'].values / np.pi
        
        if is_training and 'Target_Feedrate' in df_out.columns:
            df_out['Target_Feedrate'] = np.log1p(np.maximum(0.0, df_out['Target_Feedrate'].values))
            
        return df_out

    def fit_transform_dataset(self, df_list: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler pada kumpulan data training dan kembalikan tensor (X, Y)."""
        combined_df = pd.concat([self._apply_log_transforms(df, is_training=True) for df in df_list], axis=0)
        
        # Fit scaler
        self.feature_scaler.fit(combined_df[self.feature_cols])
        self.target_scaler.fit(combined_df[['Target_Feedrate']])
        
        X_all, Y_all = [], []
        for df in df_list:
            X_part, Y_part = self.transform_file(df, is_training=True)
            X_all.append(X_part)
            Y_all.append(Y_part)
            
        return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0)

    def transform_file(self, df: pd.DataFrame, is_training: bool = True):
        df_prep = self._apply_log_transforms(df, is_training=is_training)
        
        # Standarisasi fitur
        scaled_features = self.feature_scaler.transform(df_prep[self.feature_cols])
        
        # Terapkan Standstill Edge Padding (100 di awal, 100 di akhir)
        first_row = scaled_features[0:1, :]
        last_row = scaled_features[-1:, :]
        
        padded_features = np.vstack([
            np.repeat(first_row, self.half_w, axis=0),
            scaled_features,
            np.repeat(last_row, self.half_w, axis=0)
        ])
        
        # Bentuk Jendela Sekuens W=201
        num_samples = len(df_prep)
        num_features = len(self.feature_cols)
        X_windows = np.zeros((num_samples, self.window_size, num_features), dtype=np.float32)
        
        for i in range(num_samples):
            X_windows[i] = padded_features[i : i + self.window_size]
            
        if is_training:
            scaled_target = self.target_scaler.transform(df_prep[['Target_Feedrate']])
            return X_windows, scaled_target.astype(np.float32)
            
        return X_windows

    def save_scalers(self, path: str = "scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "feature_cols": self.feature_cols
            }, f)

    def load_scalers(self, path: str = "scaler.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.feature_scaler = data["feature_scaler"]
            self.target_scaler = data["target_scaler"]
            self.feature_cols = data["feature_cols"]
