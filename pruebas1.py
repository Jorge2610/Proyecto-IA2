import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

print("Grabando...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('E:\Proyectos\Proyecto IA2\Grabaciones\output.wav', fs, myrecording)  # Save as WAV file 