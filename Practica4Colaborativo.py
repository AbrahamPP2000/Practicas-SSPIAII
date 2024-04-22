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

from sklearn.metrics import confusion_matrix
import seaborn as sns

paused = False


class Adaline:
    def __init__(self, input_size, hidden_size=4, learning_rate=0.01, epochs=100):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, 1)  # Corrección en la forma de los pesos
        self.bias_output = np.random.rand(1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2

    def predict(self, inputs):
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_outputs = self.tanh(hidden_input)
        output = np.dot(self.hidden_outputs, self.weights_hidden_output) + self.bias_output
        return output.item()  # Convertir el array a un escalar antes de devolverlo

    def train(self, training_inputs, labels):
        errors = []
        for epoch in range(1, self.epochs + 1):
            while paused:  # Mientras esté pausado, esperar
                root.update()
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                output = self.predict(inputs)
                output_error = label - output
                total_error += output_error ** 2
                output_delta = output_error * self.tanh_derivative(output)  # Derivada de la función de activación tanh
                output_delta_reshaped = output_delta.reshape(-1, 1)  # Convertir a matriz de (4,1)
                hidden_error = np.dot(output_delta_reshaped.T,
                                      self.weights_hidden_output.T)  # Ajuste en la multiplicación
                hidden_delta = hidden_error * self.tanh_derivative(
                    self.hidden_outputs)  # Derivada de la función de activación tanh
                self.weights_hidden_output += self.learning_rate * np.outer(self.hidden_outputs, output_delta)
                self.weights_input_hidden += self.learning_rate * np.outer(inputs, hidden_delta)
                self.bias_hidden += self.learning_rate * hidden_delta.squeeze()  # Ajuste para eliminar la dimensión adicional
                self.bias_output += self.learning_rate * output_delta.squeeze()  # Ajuste para eliminar la dimensión adicional
            errors.append(total_error)
            epoch_label.config(text=f"Época actual: {epoch}")

            plot_decision_boundary()
            root.update()
            if total_error == 0:
                break
        return errors


def load_data():
    global training_data, labels, title, adaline
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    with open(filename, 'r') as file:
        title = file.readline().strip()
    raw_data = np.loadtxt(filename, skiprows=1, dtype=float)
    training_data = raw_data[:, :2]
    labels = raw_data[:, 2]
    plot_data(title, training_data, labels)
    adaline = Adaline(2)
    train_button.config(state=tk.NORMAL)


def train_adaline():
    global adaline
    learning_rate = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())

    adaline = Adaline(2, learning_rate=learning_rate, epochs=epochs)

    errors = adaline.train(training_data, labels)

    # Obtener predicciones finales
    predictions = [1 if adaline.predict(inputs) > 0 else 0 for inputs in training_data]
    # Calcular matriz de confusión
    cm = confusion_matrix(labels, predictions)
    print("Matriz de Confusión:")
    print(cm)
    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.show()


def plot_data(title, training_inputs, labels):
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    for i, inputs in enumerate(training_inputs):
        color = 'orange' if labels[i] == 1 else 'blue'
        ax.scatter(inputs[0], inputs[1], c=color)

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title(title)

    canvas.draw()


def plot_decision_boundary():
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    for i, inputs in enumerate(training_data):
        color = 'orange' if labels[i] == 1 else 'blue'
        ax.scatter(inputs[0], inputs[1], c=color)

    x_values = np.linspace(-1.5, 1.5, 100)
    y_values = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros_like(X)

    for i in range(len(X)):
        for j in range(len(X[0])):
            point = np.array([X[i][j], Y[i][j]])
            Z[i][j] = adaline.predict(point)

    ax.contourf(X, Y, Z, levels=[-np.inf, 0, np.inf], colors=('blue', 'orange'), alpha=0.3)

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title(title)

    canvas.draw()


def clear_data():
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    canvas.draw()


def fill_random():
    pass  # No se implementa en este ejemplo


def pause_training():
    global paused
    if paused == False:
        pause_button.config(text="Reanudar")
        paused = True
    else:
        pause_button.config(text="Pausar")
        paused = False  # Establecer la señal de pausa


# Interfaz
root = tk.Tk()
root.title("Red neuronal Unicapa")
root.geometry("1000x550")

# Frame del grid
grid_frame = tk.Frame(root)
grid_frame.grid(row=0, column=0, padx=10, pady=10)
grid_frame.grid_rowconfigure(0, weight=1)
grid_frame.grid_columnconfigure(0, weight=1)

fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
canvas = FigureCanvasTkAgg(fig, master=grid_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, sticky="nsew")

# Frame de los datos
data_frame = tk.Frame(root)
data_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Frame de los botones
button_frame = tk.Frame(root)
button_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Botones
load_button = tk.Button(button_frame, text="Cargar Datos", command=load_data)
load_button.grid(row=0, column=0, padx=5)

train_button = tk.Button(button_frame, text="Entrenar", command=train_adaline, state=tk.DISABLED)
train_button.grid(row=0, column=1, padx=5)

clear_button = tk.Button(button_frame, text="Limpiar", command=clear_data)
clear_button.grid(row=0, column=2, padx=5)

fill_button = tk.Button(button_frame, text="Llenar Aleatorio", command=fill_random)
fill_button.grid(row=0, column=3, padx=5)

pause_button = tk.Button(button_frame, text="Pausar", command=pause_training)
pause_button.grid(row=0, column=4, padx=5)

# Entradas para la tasa de aprendizaje y las épocas
learning_rate_label = tk.Label(data_frame, text="Tasa de Aprendizaje:")
learning_rate_label.grid(row=0, column=0, padx=5, pady=5)

learning_rate_entry = tk.Entry(data_frame)
learning_rate_entry.grid(row=0, column=1, padx=5, pady=5)

learning_rate_entry.insert(0, str(0.05))

epochs_label = tk.Label(data_frame, text="Épocas:")
epochs_label.grid(row=1, column=0, padx=5, pady=5)

epochs_entry = tk.Entry(data_frame)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)

epochs_entry.insert(0, str(200))

epoch_label = tk.Label(data_frame, text="Época actual: 0")
epoch_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()

