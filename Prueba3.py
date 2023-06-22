import speech_recognition as sr
r = sr.Recognizer()

hellow=sr.AudioFile('E:\Proyectos\Proyecto IA2\Grabaciones\output.wav')
with hellow as source:
    audio = r.record(source)
try:
    s = r.recognize_google(audio,language='es-BO')
    print("Text: "+s)
except Exception as e:
    print("Exception: "+str(e))