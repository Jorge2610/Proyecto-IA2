import sounddevice as sd
from scipy.io.wavfile import write
import os

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

comandos = {
  1: "crear tarea",
  2: "ver tareas",
  3: "borrar tarea",
  4: "editar tarea",
  5: "guardar tarea",
  6: "reintentar",
  7: "confirmar",
  8: "cancelar",
  9: "familiar",
  10: "social",
  11: "educativo",
  12: "todos"
}

for codigo in comandos:
  print("---------------------------------------------------------")
  print(f"COMANDO: {comandos[codigo]}")
  input("Presione una tecla para comenzar...")
  muestra = 1
  while muestra <= 5:
    input(f"Muestra {muestra}: presione una tecla para comenzar a grabar...")
    print("Escuchando...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(f'./03-01-{ "0" if codigo < 10 else ""}{codigo}-01-01-01-0{muestra}.wav', fs, myrecording)
    print("Grabacion finalizada")
    input("Presione una tecla para continuar...")
    muestra += 1
  os.system('cls' if os.name == 'nt' else 'clear')
