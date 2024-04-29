import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

from sklearn.metrics import confusion_matrix
import seaborn as sns

paused = False


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=100, regularization=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.random.randn(layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.errors = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        outputs = [inputs]
        for i in range(self.num_layers - 1):
            inputs = self.sigmoid(np.dot(inputs, self.weights[i]) + self.biases[i])
            outputs.append(inputs)
        return outputs

    def backward(self, outputs, labels):
        errors = [labels - outputs[-1]]
        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(errors[0], self.weights[i].T)
            errors.insert(0, error * self.sigmoid_derivative(outputs[i]))
        return errors

    def update_weights(self, outputs, errors):
        for i in range(self.num_layers - 1):
            self.weights[i] += self.learning_rate * np.dot(outputs[i].T, errors[i])
            self.biases[i] += self.learning_rate * np.sum(errors[i], axis=0)

    def train(self, training_inputs, labels):
        for epoch in range(1, self.epochs + 1):
            while paused:
                root.update()
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                inputs = inputs.reshape(1, -1)  # Reshape inputs to make them compatible with matrix operations
                outputs = self.forward(inputs)
                errors = self.backward(outputs, label)
                self.update_weights(outputs, errors)
                total_error += np.sum((errors[-1]) ** 2)
            
            self.errors.append(total_error)
            epoch_label.config(text=f"Época actual: {epoch}")
            plot_decision_boundary(training_inputs, labels)
            root.update()
            
            # Criterio de parada: todos los puntos clasificados correctamente
            if total_error == 0:
                break
        
        # Calcular predicciones finales
        predictions = [1 if self.forward(inputs.reshape(1, -1))[-1] > 0.5 else 0 for inputs in training_inputs]
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


def load_data():
    global training_data, labels, title, neural_network
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    with open(filename, 'r') as file:
        title = file.readline().strip()
    raw_data = np.loadtxt(filename, skiprows=1, dtype=float)
    training_data = raw_data[:, :2]
    labels = raw_data[:, 2]
    plot_data(title, training_data, labels)
    neural_network = NeuralNetwork(layer_sizes=[2, 5, 1])  # Example: 2 input neurons, 5 neurons in hidden layer, 1 output neuron
    train_button.config(state=tk.NORMAL)


def train_neural_network():
    global neural_network, errors
    learning_rate = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())

    layer_sizes = [2] + [int(neurons_entry.get()) for neurons_entry in neurons_entries] + [1]  # Obtener el número de neuronas por capa
    neural_network = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=learning_rate, epochs=epochs)

    errors = neural_network.train(training_data, labels)


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


def plot_decision_boundary(training_inputs, labels):
    ax.clear()
    ax.grid(True)
    
    # Definir los límites del área de cálculo
    xlim_min, xlim_max = -1.5, 1.5
    ylim_min, ylim_max = -1.5, 1.5
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    for i, inputs in enumerate(training_inputs):
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
            Z[i][j] = neural_network.forward(point.reshape(1, -1))[-1][0]

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
root.title("Red neuronal Multicapa")
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

train_button = tk.Button(button_frame, text="Entrenar", command=train_neural_network, state=tk.DISABLED)
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

learning_rate_entry.insert(0, str(0.01))

epochs_label = tk.Label(data_frame, text="Épocas:")
epochs_label.grid(row=1, column=0, padx=5, pady=5)

epochs_entry = tk.Entry(data_frame)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)

epochs_entry.insert(0, str(100))

# Entradas para el número de neuronas por capa
neurons_entries = []
for i in range(3):  # Aquí puedes definir el número máximo de capas
    label = tk.Label(data_frame, text=f"Neuronas Capa {i+1}:")
    label.grid(row=2+i, column=0, padx=5, pady=5)
    entry = tk.Entry(data_frame)
    entry.grid(row=2+i, column=1, padx=5, pady=5)
    entry.insert(0, str(5))  # Número de neuronas por defecto
    neurons_entries.append(entry)

epoch_label = tk.Label(data_frame, text="Época actual: 0")
epoch_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
