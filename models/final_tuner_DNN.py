import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel, RandomSearch
import tensorflow as tf

# Function to load data from CSV file
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return X.values, y.values

class HyperModelBuilder(HyperModel):
    def __init__(self, input_shape, n_classes=2):
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_first', min_value=16, max_value=256, step=32),
                        activation=hp.Choice('activation_first', ['relu', 'tanh']),
                        input_shape=(self.input_shape,)))
        
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(Dense(units=hp.Int('units_' + str(i), min_value=16, max_value=256, step=32),
                            activation=hp.Choice('activation_hidden', ['relu', 'tanh'])))
        
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {acc*100:.4f}%")
    print(f"F1 Score: {f1:.4f}")

# Load data from CSV file
file_path = "final_data.csv"
X, y = load_data_from_csv(file_path)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure the tuner
tuner = RandomSearch(
    HyperModelBuilder(input_shape=X_train.shape[1], n_classes=2),
    objective='val_accuracy',
    max_trials=30,
    executions_per_trial=10,
    #overwrite=True,
    directory='keras_tuner_dir',
    project_name='titanic_keras_tuner'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
evaluate_model(best_model, X_test, y_test)

# Save the best model
best_model.save("final_model_optimized.h5")

tuner.results_summary()








