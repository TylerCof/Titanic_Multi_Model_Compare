import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Function to load data from CSV file
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return X.values, y.values

# Function to build the model
# Using activation: relu, output_activatation: softmax, loss: sparse_categorical_crosssentropy, metric: accuracy
# Hidden layers are 256 and 128
def build_model(input_shape, n_classes=160, hidden_layers=[12], activation='relu', output_activation='softmax'):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation=activation, input_shape=(input_shape,)))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(n_classes, activation=output_activation))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {acc*100:.4f}%")
    print(f"F1 Score: {f1:.4f}")

# Function to plot history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Load data from CSV file
file_path = "final_data.csv"
X, y = load_data_from_csv(file_path)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = build_model(X_train.shape[1])

# Train model
history = train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Plot history
plot_history(history)

# Save the entire model
model.save("final_model.h5")

