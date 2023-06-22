import joblib
import numpy as np
import pandas as pd

import librosa

from scipy.io.wavfile import write
import sounddevice as sd

# Carga del modelo.
modelo = joblib.load('./modelo125.pkl')
# loading our scaler
scaler = joblib.load('./scaler.joblib')
# loading our encoder
encoder = joblib.load('./encoder.joblib')
print(encoder.categories_)

# Funciones para extraer caracteristicas del audio


def feat_ext(data, s_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=s_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=s_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=s_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result


def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=3, offset=0.6)
    res = feat_ext(d, s_rate)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 162))
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
    write('./output.wav',
          fs, myrecording)  # Save as WAV file
    res = prediccion('Grabaciones/output.wav')
    print(res)
    control = int(input("C: "))
