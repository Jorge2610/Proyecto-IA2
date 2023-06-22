import speech_recognition as sr
import random
import time

def p(ruta_archivo):
    r = sr.Recognizer()

    import soundfile
    data, samplerate = soundfile.read(ruta_archivo)    
    soundfile.write('output2.wav', data, samplerate, subtype='PCM_16')

    try:
    
      with sr.AudioFile('output2.wav') as fuente:
        try:
          audio = r.record(fuente)
        except Exception as e:
          pass

    except  Exception as e:
      pass
    
    try:
        texto = r.recognize_google(audio, language='es')
        time.sleep(1)
        return texto
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        pass



