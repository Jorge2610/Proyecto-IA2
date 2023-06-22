import os

import pandas as pd
import numpy as np

import librosa

# Guardamos la dirrecion de los archivos del dataset en una variable
grabaciones = "DataSet/"
directorios = os.listdir(grabaciones)
print(directorios)

# Creamos 2 arreglos para almacenar los comandos y las rutas
comando_audio = []
ruta_audio = []
for i in directorios:
    # Extraemos los archivos de cada actor.
    actor = os.listdir(grabaciones + i)
    for f in actor:
        part = f.split('.')[0].split('-')
    # El tercer numero de cada archivo representa la emocion asociada al archivo.
        comando_audio.append(int(part[2]))
        ruta_audio.append(grabaciones + i + '/' + f)

print(comando_audio[0])
print(ruta_audio[0])

# Dataframe para los comandos.
comandos_df = pd.DataFrame(comando_audio, columns=['Comando'])

# Dataframe para las rutas.
rutas_df = pd.DataFrame(ruta_audio, columns=['Ruta'])

# Dataframe con ambos dataframes
df = pd.concat([comandos_df, rutas_df], axis=1)

print(df.head())

# Cambiamos los enteros por sus respectivas emociones
df.Comando.replace({1: 'Crear tarea',
                     2: 'Ver tareas',
                     3: 'Borrar tarea',
                     4: 'Editar tarea',
                     5: 'Guardar tarea',
                     6: 'Reintentar',
                     7: 'Confirmar',
                     8: 'Cancelar',
                     9: 'Familiar',
                     10: 'Social',
                     11: 'Educativo',
                     12: 'Todos'},
                    inplace=True)

print(df.Comando.value_counts())

audio = grabaciones + 'Jorge/03-01-01-01-01-01-01.wav'
data, sr = librosa.load(audio)
# playsound(audio)

# Funciones para aplicar distintas tecnicas para generar nuevos audios
# Cambio de velocidad
def stretch(data, rate1):
    return librosa.effects.time_stretch(data, rate=rate1)

# Tono
def pitch(data, sampling_rate1, pitch_factor1):
    return librosa.effects.pitch_shift(data, sr=sampling_rate1, n_steps=pitch_factor1)

# Funciones para extraer caracteristicas del dataset
def feat_ext(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result


def get_feat(path):
    data, sample_rate = librosa.load(path, duration=3.0, offset=0.6)
    # Dato normal
    res1 = feat_ext(data, sample_rate)
    result = np.array(res1)
    # Dato generado con cambio de velocidad y tono
    new_data = stretch(data, 0.8)
    data_stretch_pitch = pitch(new_data, sample_rate, 0.7)
    res3 = feat_ext(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))
    return result

X , Y = [], []
for path, emotion in zip(df['Ruta'] , df['Comando']):
    feature = get_feat(path)
    for ele in feature:
        X.append(ele)
        Y.append(emotion)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Creamos y guardamos el dataset.
df = pd.DataFrame(X)
df['Comando'] = Y
df.to_csv('DataSet.csv', index=False)
print(df.head())