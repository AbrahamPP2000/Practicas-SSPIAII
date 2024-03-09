import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

paused = False

# Cambio del nombre de la clase: Adaline
class Adaline:
    def __init__(self, input_size, learning_rate=0.1, epochs=100, tgt_error=0.01):
        self.weights = np.random.rand(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.target_error = tgt_error # Nuevo atributo: error objetivo
        

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        
        errors = []
        for epoch in range(1, self.epochs + 1):
            while paused:  # Mientras esté pausado, esperar
                root.update()
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += error ** 2
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
            errors.append(total_error)
            plot_decision_boundary()
            root.update()
            predictions = [self.predict(inputs) for inputs in training_inputs]  # Predicciones
            accuracy = calculate_accuracy(predictions, labels)  # Calcular precisión
            confusion_matrix = calculate_confusion_matrix(predictions, labels)
            f1_score = calculate_f1_score(confusion_matrix)
            update_results(epoch, accuracy, confusion_matrix, f1_score)
            if accuracy == 1 and f1_score == 1:
                break
        return errors


def load_data():
    global training_data, labels, title, adaline, predictions
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    with open(filename, 'r') as file:
        title = file.readline().strip()
    raw_data = np.loadtxt(filename, skiprows=1, dtype=str)
    training_data = raw_data[:, :2].astype(float)
    labels = raw_data[:, 2] == 'R'
    labels = labels.astype(int)
    plot_data(title, training_data, labels)
    adaline = Adaline(2)
    predictions = []
    train_button.config(state=tk.NORMAL)

def train_perceptron():
    global adaline, predictions
    bias_value = float(bias_entry.get())
    weight1_value = float(weight1_entry.get())
    weight2_value = float(weight2_entry.get())
    learning_rate = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())
    tgt_error = float(target_error_entry.get())

    adaline = Adaline(2, learning_rate, epochs, tgt_error)
    adaline.weights[0] = bias_value
    adaline.weights[1] = weight1_value
    adaline.weights[2] = weight2_value

    errors = adaline.train(training_data, labels)

    #accuracy, confusion_matrix, f1_score = evaluate_perceptron(perceptron, training_data, labels)
    #update_results(epochs, accuracy, confusion_matrix, f1_score)

def evaluate_perceptron(perceptron, training_data, labels):
    predictions = [perceptron.predict(inputs) for inputs in training_data]
    accuracy = calculate_accuracy(predictions, labels)
    confusion_matrix = calculate_confusion_matrix(predictions, labels)
    f1_score = calculate_f1_score(confusion_matrix)
    return accuracy, confusion_matrix, f1_score

def calculate_accuracy(predictions, labels):
    correct_predictions = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct_predictions / len(labels)

def calculate_confusion_matrix(predictions, labels):
    true_positives = sum(1 for pred, label in zip(predictions, labels) if pred == label and pred == 1)
    false_positives = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    true_negatives = sum(1 for pred, label in zip(predictions, labels) if pred == label and pred == 0)
    false_negatives = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    return [[true_positives, false_positives], [false_negatives, true_negatives]]

def calculate_f1_score(confusion_matrix):
    true_positives, false_positives, false_negatives = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0]
    
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    
    recall = true_positives / (true_positives + false_negatives)
    
    if precision == 0 or recall == 0:
        return 0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def update_results(epoch, accuracy, confusion_matrix, f1_score):
    epoch_label.config(text=f"Epoch Actual: {epoch}")
    accuracy_label.config(text=f"Precisión: {accuracy}")
    confusion_matrix_label.config(text=f"Matriz de Confusión: {confusion_matrix}")
    rounded_f1_score = round(f1_score, 4)
    f1_score_label.config(text=f"Puntuación F1: {rounded_f1_score}")

def clear_data():
    global adaline, predictions
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    canvas.draw()
    adaline = None
    predictions = []
    train_button.config(state=tk.DISABLED)
    epoch_label.config(text="Epoch Actual: ")

def fill_random():
    bias_entry.delete(0, tk.END)
    weight1_entry.delete(0, tk.END)
    weight2_entry.delete(0, tk.END)
    learning_rate_entry.delete(0, tk.END)
    epochs_entry.delete(0, tk.END)

    bias_value = np.random.uniform(-1, 1)
    weight1_value = np.random.uniform(-1, 1)
    weight2_value = np.random.uniform(-1, 1)

    bias_entry.insert(0, str(round(bias_value, 2)))
    weight1_entry.insert(0, str(round(weight1_value, 2)))
    weight2_entry.insert(0, str(round(weight2_value, 2)))
    learning_rate_entry.insert(0, str(0.01))
    epochs_entry.insert(0, str(50))

def pause_training():
    global paused
    if paused == False :
        pause_button.config(text="Reanudar")
        paused = True
    else:
        pause_button.config(text="Pausar")
        paused = False  # Establecer la señal de pausa
        
        
    

def plot_data(title, training_inputs, labels):
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    black_points = False
    red_points = False

    for i, inputs in enumerate(training_inputs):
        color = 'r' if labels[i] else 'k'
        if labels[i]:
            if not red_points:
                ax.scatter(inputs[0], inputs[1], c=color, label='True')
                red_points = True
            else:
                ax.scatter(inputs[0], inputs[1], c=color)
        else:
            if not black_points:
                ax.scatter(inputs[0], inputs[1], c=color, label='False')
                black_points = True
            else:
                ax.scatter(inputs[0], inputs[1], c=color)

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title(title)
    ax.legend()

    canvas.draw()

def plot_decision_boundary():
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    black_points = False
    red_points = False

    for i, inputs in enumerate(training_data):
        color = 'r' if labels[i] else 'k'
        if labels[i]:
            if not red_points:
                ax.scatter(inputs[0], inputs[1], c=color, label='True')
                red_points = True
            else:
                ax.scatter(inputs[0], inputs[1], c=color)
        else:
            if not black_points:
                ax.scatter(inputs[0], inputs[1], c=color, label='False')
                black_points = True
            else:
                ax.scatter(inputs[0], inputs[1], c=color)

    if adaline is not None and adaline.weights[1] != 0:
        x_values = np.linspace(-2, 2, 100)
        y_values = -(adaline.weights[0] + adaline.weights[1] * x_values) / adaline.weights[2]
        ax.plot(x_values, y_values, label='Línea de decisión')
    elif adaline is not None:
        ax.axvline(x=-adaline.weights[0] / adaline.weights[1], color='g', linestyle='--', label='Línea de decisión')

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title(title)
    ax.legend()

    canvas.draw()

# Interfaz
root = tk.Tk()
root.title("Perceptrón")
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

train_button = tk.Button(button_frame, text="Entrenar", command=train_perceptron, state=tk.DISABLED)
train_button.grid(row=0, column=1, padx=5)

clear_button = tk.Button(button_frame, text="Limpiar", command=clear_data)
clear_button.grid(row=0, column=2, padx=5)

fill_button = tk.Button(button_frame, text="Llenar Aleatorio", command=fill_random)
fill_button.grid(row=0, column=3, padx=5)

pause_button = tk.Button(button_frame, text="Pausar", command=pause_training)
pause_button.grid(row=0, column=4, padx=5)

# Entradas para bias y pesos
bias_label = tk.Label(data_frame, text="Bias:")
bias_label.grid(row=0, column=0, padx=5, pady=5)

bias_entry = tk.Entry(data_frame)
bias_entry.grid(row=0, column=1, padx=5, pady=5)

weight1_label = tk.Label(data_frame, text="Peso 1:")
weight1_label.grid(row=1, column=0, padx=5, pady=5)

weight1_entry = tk.Entry(data_frame)
weight1_entry.grid(row=1, column=1, padx=5, pady=5)

weight2_label = tk.Label(data_frame, text="Peso 2:")
weight2_label.grid(row=2, column=0, padx=5, pady=5)

weight2_entry = tk.Entry(data_frame)
weight2_entry.grid(row=2, column=1, padx=5, pady=5)

learning_rate_label = tk.Label(data_frame, text="Tasa de Aprendizaje:")
learning_rate_label.grid(row=3, column=0, padx=5, pady=5)

learning_rate_entry = tk.Entry(data_frame)
learning_rate_entry.grid(row=3, column=1, padx=5, pady=5)

epochs_label = tk.Label(data_frame, text="Épocas:")
epochs_label.grid(row=4, column=0, padx=5, pady=5)

epochs_entry = tk.Entry(data_frame)
epochs_entry.grid(row=4, column=1, padx=5, pady=5)

# Nueva etiqueta: Error objetivo

target_error_label = tk.Label(data_frame, text="Error objetivo:")
target_error_label.grid(row=5, column=0, padx=5, pady=5)

target_error_entry = tk.Entry(data_frame)
target_error_entry.grid(row=5, column=1, padx=5, pady=5)

# Resultados
epoch_label = tk.Label(data_frame, text="Epoca Actual: ")
epoch_label.grid(row=6, column=0, padx=5, pady=5)

accuracy_label = tk.Label(data_frame, text="Accuracy: ")
accuracy_label.grid(row=7, column=0, padx=5, pady=5)

confusion_matrix_label = tk.Label(data_frame, text="Confusion Matrix: ")
confusion_matrix_label.grid(row=8, column=0, padx=5, pady=5)

f1_score_label = tk.Label(data_frame, text="F1 Score: ")
f1_score_label.grid(row=9, column=0, padx=5, pady=5)

root.mainloop()



