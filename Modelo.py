import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam
import joblib  

# Cargamos el dataset a partir del csv creado previamente.
data_set = pd.read_csv('DataSet.csv')
print(data_set.head())

# Prepocesamiento del dataset

X = data_set.iloc[:, :-1].values
Y = data_set['Comando'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, random_state=0, shuffle=True, train_size=0.9)
print(x_train.shape)
print(x_test.shape)

X_train = x_train.reshape(x_train.shape[0] , x_train.shape[1] , 1)
X_test = x_test.reshape(x_test.shape[0] , x_test.shape[1] , 1)

# Creamos el modelo y lo compilamos
modelo = Sequential()
modelo.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128))
modelo.add(Dropout(0.2))
modelo.add(Dense(12, activation='softmax'))
optimizador = Adam()
modelo.compile(loss='categorical_crossentropy',
               optimizer=optimizador, metrics=['accuracy'],)

# Entrenamos el modelo
modelo.fit(X_train, y_train, epochs=250, validation_data=(X_test, y_test),batch_size=64,verbose=1)

# Guardamos el modelo
joblib.dump(modelo, 'modelo_entrenado.pkl')
