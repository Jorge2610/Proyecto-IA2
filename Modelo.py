import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
import joblib  

# Cargamos el dataset a partir del csv creado previamente.
data_set = pd.read_csv('DataSet.csv')
print(data_set.head())

# Prepocesamiento del dataset

X = data_set.iloc[:, :-1].values
Y = data_set['Comando'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

print(encoder.categories_)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, random_state=0, shuffle=True, train_size=0.9)
print(x_train.shape)
print(x_test.shape)

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train =np.expand_dims(x_train, axis=2)
x_test= np.expand_dims(x_test, axis=2)

# Creamos el modelo y lo compilamos
modelo = Sequential()
modelo.add(Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X.shape[1], 1)))
modelo.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.3))

          
modelo.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
modelo.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.3))

modelo.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
modelo.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.3))
          
modelo.add(LSTM(128, return_sequences=True)) 
modelo.add(LSTM(128, return_sequences=True)) 
modelo.add(Dropout(0.3))
modelo.add(LSTM(128))
modelo.add(Dropout(0.3))

modelo.add(Dense(128, activation='relu'))

modelo.add(Dense(64, activation='relu'))

modelo.add(Dense(32, activation='relu'))

modelo.add(Dense(12, activation='softmax'))
optimizador = Adam(learning_rate=0.00045) #Default: 0.001
modelo.compile(loss='categorical_crossentropy',
               optimizer=optimizador, metrics=['accuracy'],)

# Entrenamos el modelo
modelo.fit(x_train, y_train, epochs=125, validation_data=(x_test, y_test),batch_size=64,verbose=1)

# Guardamos el modelo
joblib.dump(modelo, 'modelo125.pkl')
# Guardamos el escalador
joblib.dump(scaler, 'scaler.joblib')
# Guardamos el codificador
joblib.dump(encoder, 'encoder.joblib')

pred_test = modelo.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Etiquetas predichas', 'Etiquetas actuales'])
df['Etiquetas predichas'] = y_pred.flatten()
df['Etiquetas actuales'] = y_test.flatten()

print(df.head(10))