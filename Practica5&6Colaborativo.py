# Práctica 5%6 Red neuronal multicapa

# Implementar una red neuronal multicapa. Requisitos:
# En la capa de salida sólo va a tener una sola neurona.
# Mínimo debe haber una neurona en capa oculta. Recomendable al menos 4 neuronas en esta capa.
# Tiene que hacerse el barrido.
# Imprimir matriz de confusión.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

from sklearn.metrics import confusion_matrix
import seaborn as sns

paused = False


class Adaline:
    def __init__(self, input_size, hidden_size=10, learning_rate=0.01, epochs=100, regularization=None,
                 initial_weights=None):
        if initial_weights is not None:
            self.weights_input_hidden = initial_weights['weights_input_hidden']
            self.bias_hidden = initial_weights['bias_hidden']
            self.weights_hidden_output = initial_weights['weights_hidden_output']
            self.bias_output = initial_weights['bias_output']
        else:
            self.weights_input_hidden = np.random.randn(input_size, hidden_size)
            self.bias_hidden = np.random.randn(hidden_size)
            self.weights_hidden_output = np.random.randn(hidden_size, 1)
            self.bias_output = np.random.randn(1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.errors = []

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
        for epoch in range(1, self.epochs + 1):
            while paused:
                root.update()
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                output = self.predict(inputs)
                output_error = label - output
                total_error += output_error ** 2
                output_delta = output_error * self.tanh_derivative(output)
                output_delta_reshaped = output_delta.reshape(-1, 1)
                hidden_error = np.dot(output_delta_reshaped.T, self.weights_hidden_output.T)
                hidden_delta = hidden_error * self.tanh_derivative(self.hidden_outputs)
                self.weights_hidden_output += self.learning_rate * np.outer(self.hidden_outputs, output_delta)
                self.weights_input_hidden += self.learning_rate * np.outer(inputs, hidden_delta)
                self.bias_hidden += self.learning_rate * hidden_delta.squeeze()
                self.bias_output += self.learning_rate * output_delta.squeeze()

            if self.regularization == 'L2':
                self.weights_hidden_output -= self.learning_rate * 0.01 * self.weights_hidden_output
                self.weights_input_hidden -= self.learning_rate * 0.01 * self.weights_input_hidden

            self.errors.append(total_error)
            epoch_label.config(text=f"Época actual: {epoch}")
            plot_decision_boundary()
            root.update()

            # Criterio de parada: todos los puntos clasificados correctamente
            if total_error == 0:
                break


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
    global adaline, errors
    learning_rate = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())

    adaline = Adaline(2, learning_rate=learning_rate, epochs=epochs, regularization='L2')

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

    # Definir los límites del área de cálculo
    xlim_min, xlim_max = -1.5, 1.5
    ylim_min, ylim_max = -1.5, 1.5

    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    for i, inputs in enumerate(training_data):
        color = 'orange' if labels[i] == 1 else 'blue'
        ax.scatter(inputs[0], inputs[1], c=color)

    # Crear la malla de puntos para la frontera de decisión dentro de los límites definidos
    x_values = np.linspace(xlim_min, xlim_max, 100)
    y_values = np.linspace(ylim_min, ylim_max, 100)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros_like(X)

    for i in range(len(X)):
        for j in range(len(X[0])):
            point = np.array([X[i][j], Y[i][j]])
            Z[i][j] = adaline.predict(point)

    # Ajustar el color de la frontera de decisión en función del valor de Z
    ax.contourf(X, Y, Z, cmap='coolwarm', alpha=0.5)

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
root.title("Red neuronal multicapa")
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
