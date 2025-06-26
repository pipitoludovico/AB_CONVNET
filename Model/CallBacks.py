from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

lr_reduction = ReduceLROnPlateau(
    monitor='val_gbsa_prediction_mae',
    factor=0.2,
    patience=5,
    min_lr=1e-10
)

checkpoint = ModelCheckpoint("best_model.keras", monitor='val_gbsa_prediction_mae', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_gbsa_prediction_mae', patience=20, restore_best_weights=True, mode='min', verbose=1)
