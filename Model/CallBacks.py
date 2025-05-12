from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

lr_reduction = ReduceLROnPlateau(
    monitor='loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7
)

checkpoint = ModelCheckpoint("model.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
