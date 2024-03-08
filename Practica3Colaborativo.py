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
