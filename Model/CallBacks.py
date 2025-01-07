from keras.callbacks import ReduceLROnPlateau

lr_reduction = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# from keras.callbacks import ModelCheckpoint, Callback
# import numpy as np
#
#
# class PredictionHistory(Callback):
#     def __init__(self, X_scaled, y_original, feature_scaler=None, args_=None):
#         super().__init__()
#         self.X_train = X_scaled
#         self.y_original = y_original
#         self.differences = []
#         self.name = args_['name']
#         self.feature_scaler = feature_scaler
#         self.totalEpochs = args_['epoch']
#         self.last_saved_epoch = -1
#         self.batch_size = args_.get('batch', 32)  # Changed to 'batch' to match args
#
#     @staticmethod
#     def calculate_batchwise_mae(predictions, y_scaled, batch_size):
#         """Calcola la MAE batch-wise per allinearsi al comportamento di Keras."""
#         num_samples = predictions.shape[0]
#         num_batches = (num_samples + batch_size - 1) // batch_size  # Numero di batch totali
#         weighted_mae = 0
#         total_samples = 0
#
#         for i in range(num_batches):
#             start = i * batch_size
#             end = min(start + batch_size, num_samples)
#
#             batch_predictions = predictions[start:end]
#             batch_true = y_scaled[start:end]
#             batch_mae = np.mean(np.abs(batch_predictions - batch_true.reshape(-1, 1)))
#
#             # Pondera la MAE del batch per la dimensione del batch
#             weighted_mae += batch_mae * (end - start)
#             total_samples += (end - start)
#
#         # Calcola la MAE totale ponderata
#         batchwise_mae = weighted_mae / total_samples
#         return batchwise_mae
#
#     def on_epoch_end(self, epoch, logs=None):
#         if self.last_saved_epoch == epoch:
#             with open(f'report_{self.name}.txt', 'a') as report:
#                 report.write(f"\nEpoch {epoch} - Predictions and Differences:\n")
#
#                 # Get raw predictions
#                 predictions = self.model.predict(self.X_train, verbose=0)
#
#                 # Ensure predictions are properly shaped before inverse transform
#                 pred_reshaped = predictions.reshape(-1, 1)
#                 pred_orig = self.feature_scaler.inverse_transform(pred_reshaped)
#
#                 absolute_errors = []
#                 for i in range(len(self.X_train)):
#                     error = abs(pred_orig[i][0] - self.y_original[i])
#                     absolute_errors.append(error)
#
#                     report.write(
#                         f"Sample {i}: Raw Prediction = {predictions[i][0]:.6f}, "
#                         f"Rescaled = {pred_orig[i][0]:.4f}, "
#                         f"Actual = {self.y_original[i]:.4f}, "
#                         f"Error = {error:.4f}\n"
#                     )
#
#                 mae = np.mean(absolute_errors)
#                 scaled_mae = self.calculate_batchwise_mae(predictions, self.y_original, self.batch_size)
#
#                 report.write(
#                     f"Original Scale MAE: {mae:.4f}\n"
#                     f"Scaled MAE: {scaled_mae:.6f}\n"
#                     f"Keras MAE: {logs['mean_absolute_error']:.6f}\n"
#                 )
#
#                 self.differences.append(mae)
#
#
# class SaveEpochTracker(ModelCheckpoint):
#     def __init__(self, *args, prediction_history=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.prediction_history = prediction_history
#
#     def on_epoch_end(self, epoch, logs=None):
#         super().on_epoch_end(epoch, logs)
#         # Se il modello viene salvato, aggiorna l'epoch salvato
#         if self.best == logs.get(self.monitor):  # Controlla se l'attuale è il migliore
#             if self.prediction_history is not None:
#                 self.prediction_history.last_saved_epoch = epoch
#
#
# def MakeCheckPoint(name_, prediction_history):
#     checkpoint_path = f'model_{name_}.keras'
#     checkpoint = SaveEpochTracker(
#         filepath=checkpoint_path,
#         monitor='mean_absolute_error',
#         save_best_only=True,
#         mode='min',
#         verbose=1,
#         prediction_history=prediction_history  # Passa il callback per sincronizzare
#     )
#     return checkpoint
