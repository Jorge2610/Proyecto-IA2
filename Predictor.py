import keras
import joblib  
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import librosa
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np

import sounddevice as sd
from scipy.io.wavfile import write

modelo = joblib.load('modelo_entrenado.pkl') # Carga del modelo.

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

print("Grabando audio...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('E:\Proyectos\Proyecto IA2\output.wav', fs, myrecording)  # Save as WAV file 

ravdess = "audio_speech_actors_01-24/"

# Creamos 2 arreglos para almacenar las emociones y las rutas

file_emotion = []
file_emotion.append(1)
file_path = []
file_path.append('output.wav')

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

data, sr = librosa.load('output.wav')

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
    res = np.array(res1)
    return res

X , Y = [], []
feature = get_feat('output.wav')
for ele in feature:
    X.append(ele)
Y.append(ravdess_df['Emotions'][0])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Creamos y guardamos el dataset.

Emotions = pd.DataFrame(X)
Emotions['Emotions'] = Y
Emotions.to_csv('emotion.csv', index=False)
print(Emotions.head())

# Cargamos el dataset a partir del csv creado previamente.

Emotions = pd.read_csv('./emotion.csv')
Emotions.head(10)

X = Emotions.iloc[: ,:-1].values
Y = Emotions['Emotions'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

X_test = X.reshape(X.shape[0] , X.shape[1] , 1)

pred_test = modelo.predict(X_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(Y)

# Mostramos los resultados de las predicciones

df = pd.DataFrame(columns=['Etiquetas predichas', 'Etiquetas actuales'])
df['Etiquetas predichas'] = y_pred.flatten()
df['Etiquetas actuales'] = y_test.flatten()

print(df.head(10))