import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from src.Important_functions import threshold_setter, lr_decay, encoder_model

# importing data
df = pd.read_csv(r'/Users/tushardesai/Documents/kaggle/Creditcard/Data/creditcard.csv')
print(df.isnull().values.any())
df.drop('Time', axis=1, inplace=True)
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

# Train and Test data
X_train, X_test = train_test_split(df, test_size=0.1, random_state=0)
X_train = X_train[X_train['Class'] == 0]
X_train.drop('Class', axis=1, inplace=True)
Y_test = X_test['Class']
X_test = X_test.drop('Class', axis=1)

# Model(29,12,4,12,29)
input_shape = X_train.shape[1]
hidden_neurons = 12
model = encoder_model(input_shape, hidden_neurons)

# filename
model_path = '/Users/tushardesai/Documents/kaggle/Creditcard'
model_name = time.strftime('%d_%m_%Y_%H_%M_%S') + '_creditcard_fraud.hdf5'
full_path = os.path.join(model_path, model_name)
tensorboard_logs = model_path + '/logs'

# Training of the model

checkpoint = ModelCheckpoint(full_path, verbose=0, monitor='val_loss', save_best_only=True, load_weights_on_restart=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
tensorboard = TensorBoard(write_graph=True, write_images=True)
learning_rate = LearningRateScheduler(lr_decay, verbose=1)
history = model.fit(X_train, X_train, epochs=1000, batch_size=2048, validation_split=0.1, verbose=1,
                    callbacks=[checkpoint, earlystopping, tensorboard])

# plotting train_test loss
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(loc='upper right')
plt.show()

# Predictions
X_test_prediction = model.predict(X_test)
mse = np.mean(np.power(X_test - X_test_prediction, 2), axis=1)
recon_df = pd.DataFrame({'mse': mse, 'actual_class': Y_test})

# getting f1 scores for different threshold values
threshold_val_list = list(np.linspace(1, 100, 100))
threshold_score_dataframe = pd.DataFrame(columns=['thresold', 'precision', 'recall', 'f1'])
for threshold_val in threshold_val_list:
    score_dict = threshold_setter(threshold_val=threshold_val, result_df=recon_df)
    threshold_score_dataframe = threshold_score_dataframe.append(score_dict, ignore_index=True)

threshold_score_dataframe.to_excel(model_path + '/threshold_score.xlsx')


