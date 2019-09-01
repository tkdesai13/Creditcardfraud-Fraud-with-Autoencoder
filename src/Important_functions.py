import math
import numpy as np

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense


def threshold_setter(threshold_val, result_df):
    result_df['predicted_class'] = np.where(result_df['mse'] >= threshold_val, 1, 0)

    tp = np.sum((result_df.predicted_class == 1) & (result_df.actual_class == 1))
    fp = np.sum((result_df.predicted_class == 1) & (result_df.actual_class == 0))
    fn = np.sum((result_df.predicted_class == 0) & (result_df.actual_class == 1))
    tn = np.sum((result_df.predicted_class == 0) & (result_df.actual_class == 0))

    precision = float(tp / (tp + fp))
    recall = float(tp / (tp + fn))
    f1_score = (2 * precision * recall) / (precision + recall)

    return {'thresold': threshold_val, 'precision': precision, 'recall': recall, 'f1': f1_score}


def lr_decay(epochs):
    initial_lrrate = 0.01
    drop = 0.1
    epoch_drop = 200
    lr_rate = initial_lrrate * math.pow(drop, math.floor((1 + epochs) / epoch_drop))
    return lr_rate


def encoder_model(input_dim, dense_neurons):
    model = models.Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim,), activation='tanh'))
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Dense(dense_neurons / 3, activation='relu'))
    model.add(Dense(dense_neurons, activation='tanh'))
    model.add(Dense(input_dim, activation='linear'))
    print(model.summary())

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model
