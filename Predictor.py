import joblib
import numpy as np
import pandas as pd

import librosa

from scipy.io.wavfile import write
import sounddevice as sd

# Carga del modelo.
modelo = joblib.load('modelo250.pkl')
# loading our scaler
scaler = joblib.load('scaler.joblib')
# loading our encoder
encoder = joblib.load('encoder.joblib')
print(encoder.categories_)

# Funciones para extraer caracteristicas del audio


def feat_ext(data, s_rate):
    # ZCR Caracteristicas acusticas de bajo nivel
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # Apilamos los datos horizontalmente
    # MFCC Caracteristicas del dominio de frecuencia
    mfcc = np.mean(librosa.feature.mfcc(
        y=data, sr=s_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))  # Apilamos los datos horizontalmente
    return result


def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = feat_ext(d, s_rate)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 41))
    i_result = scaler.transform(result)
    final_result = np.expand_dims(i_result, axis=2)

    return final_result


def prediccion(path1):
    res = get_predict_feat(path1)
    predictions = modelo.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    return y_pred[0][0]

control = 1

while control == 1:
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    print("Grabando...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('E:\Proyectos\Proyecto IA2\Grabaciones\output.wav',
          fs, myrecording)  # Save as WAV file
    res = prediccion('Grabaciones/output.wav')
    print(res)
    control = int(input("C: "))
