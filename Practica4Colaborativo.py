# Práctica 4 Red neuronal unicapa


# Red unicapa. Podemos tener “n” entradas y “m” capas. Podemos hacer una clasificación de valores. Podemos hacer clasificaciones múltiples. Cada neurona tiene sus propios pesos, su propio bias, sus propias conexiones y no se comparten. La cooperación entre todas es clave. Es una red lineal. Hay una forma de implementar una red unicapa:
# Cada neurona aprende a diferenciar la información que tiene. Cuando más de una neurona se enciende, puede haber ambigüedad.
# El sobre entrenamiento ocurre cuando tenemos datos no linealmente separables.
# Un sub-ajuste es cuando no todos los datos se han podido clasificar.

# Red unicapa. Permite hacer multi-clasificación.
# Es recomendable convertir el valor de la salida en binario y ese valor será “z” en el barrido.
# La red unicapa genera más de una salida, por lo que se puede organizar en un vector.

# Preferible hacer diferentes ejemplos con diferentes archivos de salidas. El número de neuronas va en función al número de columnas del archivo de salidas.

# Se puede utilizar un archivo de dos entradas.

# El archivo de salidas puede tener varias columnas. Estas columnas indican las salidas de cada neurona.

# Cada neurona es independiente, pero no es la mejor forma. El algoritmo de entrenamiento aplica para todas las neuronas simultáneamente. Cada uno de los pasos del entrenamiento aplica para cada una de las neuronas de la red.




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

paused = False


