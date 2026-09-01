"""
train_bi_lstm.py
Tahap 4: Pelatihan Model Dual-Layer Bi-LSTM untuk Prediksi Profil Kecepatan CNC.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np
from typing import Tuple

def build_bilstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Membangun arsitektur Dual-Layer Bi-LSTM."""
    inputs = layers.Input(shape=input_shape, name="NC_Sequence_Input")
    
    # Layer 1: Bidirectional LSTM dengan Feature Dropout
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, name="BiLSTM_L1"))(inputs)
    x = layers.SpatialDropout1D(0.2)(x)
    
    # Layer 2: Bidirectional LSTM memadat ke konteks target tengah
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False, name="BiLSTM_L2"))(x)
    x = layers.BatchNormalization()(x)
    
    # Dense Regressor Head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="linear", name="Normalized_Feedrate_Output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CNC_Kinematics_BiLSTM")
    
    # Optimizer AdamW dengan Huber Loss
    # We must use tf.keras.optimizers.AdamW in recent Keras / TF versions
    optimizer = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0), metrics=["mae", "mse"])
    
    return model

def run_training(X_train: np.ndarray, Y_train: np.ndarray, 
                 X_val: np.ndarray, Y_val: np.ndarray, 
                 model_save_path: str = "bilstm_feedrate_model.keras"):
    
    model = build_bilstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    training_callbacks = [
        callbacks.ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
    ]
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=128,  # Sesuai keputusan spesifikasi
        callbacks=training_callbacks,
        verbose=1
    )
    
    return model, history
