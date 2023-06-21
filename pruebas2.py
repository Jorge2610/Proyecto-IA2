# Importacion las librerias necesarias

import pandas as pd
import numpy as np

import os
import sys

# Librosa es un libreria para analizar audio y musica, esta libreria tambien puede ser usada para extraer caracteristicas de un audio.

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Librerias para reproducir audio.

import IPython.display as ipd
from IPython.display import Audio
from playsound import playsound
import keras
import joblib  
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.optimizers import Adam



import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
print ("Done")

# Guardamos la dirrecion de los archivos del dataset en una variable

ravdess = "audio_speech_actors_01-24/"
ravdess_directory_list = os.listdir(ravdess)
print(ravdess_directory_list)

# Creamos 2 arreglos para almacenar las emociones y las rutas

file_emotion = []
file_path = []
for i in ravdess_directory_list:
    # Como existen 24 actores diferentes en el directorio previo, necesitamos extraer los archivos de cada actor.
    actor = os.listdir(ravdess + i)
    for f in actor:
        part = f.split('.')[0].split('-')
    # El tercer numero de cada archivo representa la emocion asociada al archivo.
        file_emotion.append(int(part[2]))
        file_path.append(ravdess + i + '/' + f)

print(file_path[0])
print(actor[0])
print(part[0])
print(int(part[2]))

# Dataframe para las emociones.
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
# Dataframe para las rutas.
path_df = pd.DataFrame(file_path, columns=['Path'])
# Dataframe con ambos dataframes
ravdess_df = pd.concat([emotion_df, path_df], axis=1)
# Cambiamos los enteros por sus respectivas emociones
ravdess_df.Emotions.replace({1:'Crear tarea', 
                            2:'Ver tareas',
                            3:'Borrar tarea', 
                            4:'Editar tarea', 
                            5:'Guardar tarea',
                            6:'Reintentar', 
                            7:'Confirmar',
                            8:'Cancelar',
                            9:'Familiar',
                            10:'Social',
                            11:'Educativo',
                            12:'Todos'},
                            inplace=True)
print(ravdess_df.head())
print("______________________________________________")
print(ravdess_df.Emotions.value_counts())

fRA1= ravdess + 'Actor_01/03-01-01-01-01-01-01.wav'
data, sr = librosa.load(fRA1)
playsound(fRA1)
#ipd.Audio(fRA1)

# Funciones para aplicar distintas tecnicas para generar nuevos audios

# Ruido
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data
# Cambio de velocidad
def stretch(data, rate1):
    return librosa.effects.time_stretch(data, rate = rate1)
# Poner delay al audio y mezclarlo con el audio original
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# Tono
def pitch(data, sampling_rate1, pitch_factor1):
    return librosa.effects.pitch_shift(data, sr = sampling_rate1, n_steps = pitch_factor1)

def feat_ext(data):
    # ZCR Caracteristicas acusticas de bajo nivel
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # Apilamos los datos horizontalmente
    # MFCC Caracteristicas del dominio de frecuencia
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr,n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc)) # Apilamos los datos horizontalmente
    return result

def get_feat(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    # Dato normal
    res1 = feat_ext(data)

    result = np.array(res1)
    # Dato con ruido
    #noise_data = noise(data)
    #res2 = feat_ext(noise_data)
    #result = np.vstack((result, res2))
    # Dato con la velocidad cambiada y el tono
    new_data = stretch(data, 0.8)
    data_stretch_pitch = pitch(new_data, 22050, 0.7)
    res3 = feat_ext(data_stretch_pitch)
    result = np.vstack((result, res3))
    return result

X , Y = [], []
for path, emotion in zip(ravdess_df['Path'] , ravdess_df['Emotions']):
    feature = get_feat(path)
    for ele in feature:
        X.append(ele)
        Y.append(emotion)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Creamos y guardamos el dataset.

Emotions = pd.DataFrame(X)
Emotions['Emotions'] = Y
Emotions.to_csv('emotion.csv', index=False)

# Cargamos el dataset a partir del csv creado previamente.

Emotions = pd.read_csv('./emotion.csv')
Emotions.head(10)

X = Emotions.iloc[: ,:-1].values
Y = Emotions['Emotions'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True, train_size=0.9)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

X_train = x_train.reshape(x_train.shape[0] , x_train.shape[1] , 1)
X_test = x_test.reshape(x_test.shape[0] , x_test.shape[1] , 1)

# Creamos el modelo y lo compilamos

modelo=Sequential()
modelo.add(LSTM(128,return_sequences=True,input_shape=(x_train.shape[1],1)))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128,return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128,return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128,return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128,return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128,return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128))
modelo.add(Dropout(0.2))
modelo.add(Dense(12,activation = 'softmax'))
optimizador = Adam(learning_rate=0.005)
modelo.compile(loss='categorical_crossentropy',optimizer=optimizador,metrics=['accuracy'],)

hist=modelo.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test),batch_size=64,verbose=1)

pred_test = modelo.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

# Mostramos los resultados de las predicciones

df = pd.DataFrame(columns=['Etiquetas predichas', 'Etiquetas actuales'])
df['Etiquetas predichas'] = y_pred.flatten()
df['Etiquetas actuales'] = y_test.flatten()

print(df.head(10))

joblib.dump(modelo, 'modelo_entrenado.pkl') # Guardo el modelo.