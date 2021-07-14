import pandas
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sys


import tensorflow as tf
import tensorflow.keras.models
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer



df = pd.read_csv('regression lib/admissions_data.csv')

print(df.head())
print(df.describe())

labels = df.iloc[:, -1]
features = df.iloc[:, 0:-1]

print(len(labels))
print(len(features))

features = pandas.get_dummies(df)

features_train, features_test, labels_train, labels_test = (
    sklearn.model_selection.train_test_split(
        features,
        labels,
        test_size = 0.2,
        random_state = 23
    )
)

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer(
    [("only numeric", StandardScaler(), numerical_columns)],
    remainder='passthrough'
)

features_train_scaled = ct.fit_transform(features_train)
features_scaled_test = ct.transform(features_test)

model = tensorflow.keras.models.Sequential()

my_input = InputLayer(input_shape = (features.shape[1], ))
model.add(my_input)

model.add(Dense(16, activation = 'relu'))

model.add(Dense(1))
model.summary()


opt = Adam(learning_rate= 0.01)
model.compile(loss = 'mse', metrics=['mae'], optimizer= opt)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(features_train_scaled, labels_train, epochs= 40, batch_size= 1, verbose= 1, callbacks=[tensorboard_callback])


res_mse, res_mae = model.evaluate(features_scaled_test, labels_test, verbose =0)
print(res_mae)
print(res_mse)

predicted_labels = model.predict(features_scaled_test)
rs = r2_score(labels_test, predicted_labels)
print(rs)
