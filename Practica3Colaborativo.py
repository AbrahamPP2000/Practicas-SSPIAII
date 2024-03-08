import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
import matplotlib.pyplot as plt
import numpy as np

# Obtención de los datos del archivo de entradas
try:
    with open("entradas.csv", "r") as file:
        lines = file.readlines()
        entradas = []
        for line in lines:
            x, y = map(float, line.strip().split(","))
            entradas.append((x, y))
        x_entradas = np.array([i for i in entradas])
except FileNotFoundError:
    print("Archivo de entradas no encontrado.")

# Obtención de los datos del archivo de salidas
try:
    with open("salidas.csv", "r") as file:
        lines = file.readlines()
        y_salidas = [float(line.strip()) for line in lines]
except FileNotFoundError:
    print("Archivo de salidas no encontrado.")


def mostrar_grafica_inicial():
    pass

def entrenar_Adaline_para_clasificacion():
    pass

def entrenar_Adaline_para_regresion():
    pass


# Configuración de la interfaz gráfica
interfaz_inicial = tk.Tk()
interfaz_inicial.geometry("320x370")
interfaz_inicial.title("Adaline")

peso1_label = ttk.Label(interfaz_inicial, width=30, text="Peso 1:")
peso1_label.grid(row=0, column=0)
peso1_entry = ttk.Entry(interfaz_inicial)
peso1_entry.grid(row=0, column=1)

peso2_label = ttk.Label(interfaz_inicial, width=30, text="Peso 2:")
peso2_label.grid(row=1, column=0)
peso2_entry = ttk.Entry(interfaz_inicial)
peso2_entry.grid(row=1, column=1)

bias_label = ttk.Label(interfaz_inicial, width=30, text="Bias:")
bias_label.grid(row=2, column=0)
bias_entry = ttk.Entry(interfaz_inicial)
bias_entry.grid(row=2, column=1)

learning_rate_label = ttk.Label(interfaz_inicial, width=30, text="Parámetro de aprendizaje:")
learning_rate_label.grid(row=3, column=0)
learning_rate_entry = ttk.Entry(interfaz_inicial)
learning_rate_entry.grid(row=3, column=1)
learning_rate_entry.insert(tk.END, "0.01")

epochs_label = ttk.Label(interfaz_inicial, width=30, text="Épocas:")
epochs_label.grid(row=4, column=0)
epochs_entry = ttk.Entry(interfaz_inicial)
epochs_entry.grid(row=4, column=1)
epochs_entry.insert(tk.END, "100")

error_objetivo_label = ttk.Label(interfaz_inicial, width=30, text="Error objetivo:")
error_objetivo_label.grid(row=5, column=0)
error_objetivo_entry = ttk.Entry(interfaz_inicial)
error_objetivo_entry.grid(row=5, column=1)
error_objetivo_entry.insert(tk.END, "0.01")

# Etiquetas para mostrar precisión, matriz de confusión y F1 Score
precision_label = ttk.Label(interfaz_inicial, text="Precisión: ")
precision_label.grid(row=9, columnspan=2)

matriz_confusion_label = ttk.Label(interfaz_inicial, text="Matriz de confusión:\n")
matriz_confusion_label.grid(row=10, columnspan=2)

f1_score_label = ttk.Label(interfaz_inicial, text="F1 Score: ")
f1_score_label.grid(row=11, columnspan=2)

mostrar_grafica_button = ttk.Button(interfaz_inicial, text="Mostrar gráfica inicial", command=mostrar_grafica_inicial)
mostrar_grafica_button.grid(row=6, columnspan=2)

entrenar_clasificacion_button = ttk.Button(interfaz_inicial, text="Entrenar para clasificacion", command=entrenar_Adaline_para_clasificacion)
entrenar_clasificacion_button.grid(row=7, columnspan=2)

entrenar_regresion_button = ttk.Button(interfaz_inicial, text="Entrenar para regresión", command=entrenar_Adaline_para_regresion)
entrenar_regresion_button.grid(row=8, columnspan=2)

interfaz_inicial.mainloop()
