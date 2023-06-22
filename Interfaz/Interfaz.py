import customtkinter as ctk
import tkinter as tk
import os
from PIL import Image, ImageTk
import sounddevice as sd
from scipy.io.wavfile import write
import threading as th

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

def hash():
    return os.urandom(2).hex()

class AppWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.tipo_tarea_actual = "familiar"
        self.title_tarea_actual = "visitar a la abuela"

        self.title("Asistente de Tareas")
        self.geometry("400x400+0+0")
        self.resizable(False, False)
        self.update_idletasks()
        width = 400
        height = 500
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{x+150}+{y}")

        self.container = ctk.CTkFrame(self, width=400, height=400)
        self.container.columnconfigure(0, weight=1)
        self.container.rowconfigure(1, weight=1)
        self.container.rowconfigure(2, weight=3)
        self.container.rowconfigure(3, weight=1)

        self.title_frame = ctk.CTkFrame(self.container, width=400, height=100, border_width=2)
        self.main_frame = ctk.CTkFrame(self.container, width=400, height=200, border_width=2)
        self.microfono_frame = ctk.CTkFrame(self.container, width=400, height=100, border_width=2)

        self.title_frame.grid(row=0, column=0, sticky=tk.NSEW, ipadx=10, ipady=10, padx=(10, 10), pady=(10, 10))
        self.title_frame.columnconfigure(0, weight=1)

        self.main_frame.grid(row=1, column=0, sticky=tk.NSEW, ipadx=10, ipady=10, padx=(10, 10))

        self.microfono_frame.grid(row=2, column=0, sticky=tk.NSEW, ipadx=10, ipady=10, padx=(10, 10), pady=(10, 10))

        self.label_title = ctk.CTkLabel(self.title_frame, text="Home", font=ctk.CTkFont(size=15, weight="bold"),
                                        padx=10, pady=10
                                        )
        self.label_title.pack(side=tk.LEFT, padx=(10, 0))

        self.label_status = ctk.CTkLabel(self.main_frame, text="Diga un Comando", font=ctk.CTkFont(size=15, weight="bold"))
        self.label_status.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.imagen_tk = Image.open("./off_micro.png")
        self.imagen_tk.resize((50, 50))
        self.imagen_tk = ImageTk.PhotoImage(self.imagen_tk)
        self.button_microfono = tk.Button(self.microfono_frame, image=self.imagen_tk, command=self.escuchar, width=60, height=60)
        self.button_microfono.pack(pady=(20, 0))

        self.caja_texto = ctk.CTkEntry(self.microfono_frame, width=100, height=10)
        self.caja_texto.pack(pady=(10, 0))
        self.enter_button = ctk.CTkButton(self.microfono_frame, text="Enter", command=self.enter)
        self.enter_button.pack(pady=(10, 0))

        self.container.pack(expand=True, fill=tk.BOTH)

        self.comando_ant = ""
        self.respuesta = ""
        self.comandos = {
            "crear tarea": self.crear_tarea,
            "ver tareas" : self.ver_tareas_general,
            "crear tarea familiar" : lambda: self.tipo_tarea("familiar"),
            "crear tarea social" : lambda: self.tipo_tarea("social"),
            "crear tarea educativo" : lambda: self.tipo_tarea("educativo"),
            "ver tareas familiar" : lambda: self.ver_tareas("familiar"),
            "ver tareas social" : lambda: self.ver_tareas("social"),
            "ver tareas educativo" : lambda: self.ver_tareas("educativo"),
            "ver tareas todos" : lambda: self.ver_tareas("todos"),
            "todos" : lambda: self.tipo_tarea("todos"),
            "cancelar" : self.cancelar,
            "guardar tarea" : self.guardar_tarea,
            "reintentar" : self.reintentar,
            "borrar tarea" : self.ver_tareas_borrar,
            "confirmar" : self.confirmar,
        }

    def enter(self):
        self.respuesta = self.caja_texto.get()
        self.actuar()

    def escuchar(self):
        self.imagen_tk = Image.open("./on_micro.png")
        self.imagen_tk.resize((50, 50))
        self.imagen_tk = ImageTk.PhotoImage(self.imagen_tk)
        self.button_microfono.configure(image=self.imagen_tk)
        self.label_status.configure(text="Escuchando...")
        print("Iniciando grabación...")
        self.hilo = th.Thread(target=self.grabar)
        self.hilo.start()

    def grabar(self):
        fs = 44100
        seconds = 4
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        self.imagen_tk = Image.open("./off_micro.png")
        self.imagen_tk.resize((50, 50))
        self.imagen_tk = ImageTk.PhotoImage(self.imagen_tk)
        self.button_microfono.configure(image=self.imagen_tk)
        write("ouput.wav", fs, myrecording)
        print("Grabacion finalizada")
        self.actuar()

    def actuar(self):
        #self.respuesta = self.predecir()
        if self.respuesta == "crear tarea" or self.respuesta == "ver tareas" or self.respuesta == "cancelar":
            self.comando_ant = ""
        self.comandos[f"{self.comando_ant}{self.respuesta}"]()
    
    def reintentar(self):
        self.tipo_tarea_actual = ""
        self.title_tarea_actual = ""
        self.comando_ant = ""
        self.crear_tarea()

    def crear_tarea(self):
        try:
            self.tareas_frame.destroy()
        except:
            pass
        self.label_title.configure(text="Crear Tarea")
        self.comando_ant = "crear tarea "
        self.label_status.configure(text="¿Qué tipo de tarea desea crear?")

    def tipo_tarea(self, tipo):
        self.comando_ant = ""
        self.tipo_tarea_actual = tipo
        self.label_status.configure(text="¿Ingrese la tarea?")
        self.caja_titulo = ctk.CTkEntry(self.main_frame, width=100, height=10)
        self.caja_titulo.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
    
    def cancelar(self):
        try:
            self.tareas_frame.destroy()
        except:
            pass
        try:
            self.caja_titulo.destroy()
        except:
            pass
        try:
           self.codigo_borrar.destroy()
        except:
            pass
        self.comando_ant = ""
        self.label_title.configure(text="Home")
        self.label_status.configure(text="Diga un Comando")
        self.label_status.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def guardar_tarea(self):
        self.title_tarea_actual = self.caja_titulo.get()
        self.caja_titulo.destroy()
        self.comando_ant = ""
        if self.title_tarea_actual == "":
            return
        cantTareas = len(os.listdir(f"./tareas/{self.tipo_tarea_actual}")) + 1
        new_tarea = open(f"./tareas/{self.tipo_tarea_actual}/{self.tipo_tarea_actual[0:2]}{hash()}-{cantTareas}.txt", "w")
        new_tarea.write(self.title_tarea_actual)
        new_tarea.close()
        self.cancelar()
      
    def ver_tareas_general(self):
        try:
            self.tareas_frame.destroy()
        except:
            pass
        self.comando_ant = "ver tareas "
        self.label_title.configure(text="Ver Tareas")
        self.label_status.configure(text="¿Qué tipo de tareas desea ver?")

    def ver_tareas(self, tipo):
        self.comando_ant = ""
        self.label_title.configure(text=f"Tareas: {tipo}")
        self.tareas_frame = ctk.CTkFrame(self.main_frame, width=300, height=50)
        self.label_status.configure(text="")
        scrollbar = tk.Scrollbar(self.tareas_frame, orient=tk.VERTICAL)
        self.obj_list = tk.Listbox(self.tareas_frame, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.obj_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.obj_list.pack(expand=True, fill=tk.BOTH, pady=5, padx=5)

        self.obj_list.delete(0, tk.END)

        if tipo != "todos":
          for tarea in os.listdir(f"./tareas/{tipo}"):
            cont_tarea = open(f"./tareas/{tipo}/{tarea}", "r")
            self.obj_list.insert(tk.END, cont_tarea.read())
            cont_tarea.close()
        else:
          for tipo in os.listdir(f"./tareas"):
            for tarea in os.listdir(f"./tareas/{tipo}"):
              cont_tarea = open(f"./tareas/{tipo}/{tarea}", "r")
              self.obj_list.insert(tk.END, cont_tarea.read())
              cont_tarea.close()

        self.tareas_frame.pack(expand=True, fill=tk.BOTH, pady=5, padx=5)

    def ver_tareas_borrar(self):
        self.label_status.configure(text="Ingrese el codigo para borrar y diga CONFIRMAR")
        self.label_status.pack()
        self.codigo_borrar = ctk.CTkEntry(self.main_frame, width=100, height=10)
        self.codigo_borrar.pack()
        self.tareas_frame = ctk.CTkFrame(self.main_frame, width=300, height=50)
        scrollbar = tk.Scrollbar(self.tareas_frame, orient=tk.VERTICAL)
        self.obj_list = tk.Listbox(self.tareas_frame, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.obj_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.obj_list.pack(expand=True, fill=tk.BOTH, pady=5, padx=5)

        self.obj_list.delete(0, tk.END)

        for tipo in os.listdir(f"./tareas"):
          for tarea in os.listdir(f"./tareas/{tipo}"):
            cont_tarea = open(f"./tareas/{tipo}/{tarea}", "r")
            self.obj_list.insert(tk.END, cont_tarea.read()+"  [codigo: "+tarea[:-4]+"]")
            cont_tarea.close()
        
        self.tareas_frame.pack(expand=True, fill=tk.BOTH, pady=5, padx=5)

    def confirmar(self):
        codigo = self.codigo_borrar.get()
        self.codigo_borrar.destroy()
        self.comando_ant = ""
        if codigo == "":
            return
        for tipo in os.listdir(f"./tareas"):
          for tarea in os.listdir(f"./tareas/{tipo}"):
            if codigo == tarea[:-4]:
              os.remove(f"./tareas/{tipo}/{tarea}")
              self.cancelar()
              return
        self.cancelar()
        self.label_status.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def predecir(self):
        # ejecutar modelo
        # path = "output.wav"
        respuesta = "crear tarea"
        return respuesta



if __name__ == "__main__":
    ventana = AppWindow()
    ventana.mainloop()