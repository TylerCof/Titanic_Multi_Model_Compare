import os
import numpy as np
import pandas as pd

def load_model(model_path):
    """
    Dynamically loads a model based on the file extension.
    Args:
    model_path (str): The path to the model file.

    Returns:
    A loaded model object.
    """
    _, file_extension = os.path.splitext(model_path)
    if file_extension == '.pkl':
        import joblib
        return joblib.load(model_path)
    elif file_extension == '.h5':
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    else:
        raise ValueError("Unsupported file extension: {}".format(file_extension))

def preprocess_data(data):
    """
    Applies preprocessing steps to the data.
    This function should be modified according to your data preprocessing steps.
    Args:
    data (DataFrame): Input data.

    Returns:
    Processed data.
    """
    # Example preprocessing: Assuming data needs scaling and/or encoding
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # processed_data = scaler.fit_transform(data)
    # return processed_data
    


    if 'Survived' in data.columns:
        data = data.drop('Survived', axis=1).values



    return data  # No processing in this placeholder function

def predict(model, data):
    """
    Makes predictions using the loaded model.
    Args:
    model: The loaded model.
    data (DataFrame): The input data.

    Returns:
    Predictions.
    """
    processed_data = preprocess_data(data)
    return model.predict(processed_data)

def main():
    
    data_path = "ex_data.csv"

    '''
    DNN = "final_model_optimized.h5"
    VoterSoft = "soft_voting_classifier.pkl"
    VoterHard = "hard_voting_classifier.pkl"
    Forest = "forest_model.pkl" 
    SVM = "final_svc.pkl"

    print("Please select model to use: ")
    print("1- DNN")
    print("2- VoterSoft")
    print("3- VoterHard")
    print("4- Random Forest")
    print("5- SVM")
    '''

    models = {
        "1": "final_model_optimized.h5",
        "2": "soft_voting_classifier.pkl",
        "3": "hard_voting_classifier.pkl",
        "4": "forest_model.pkl",
        "5": "final_svm.pkl"
    }

    print("Please select the model to use:")
    for key, value in models.items():
        print(f"{key}: {os.path.splitext(value)[0]}")

    choice = input("Enter your choice (1-5): ")
    model_path = models.get(choice)
    if not model_path:
        print("Invalid choice, exiting.")
        return



    #model_path = input("Enter the path to your model file: ")
    model = load_model(model_path)

    # Assume data comes from a CSV or similar; modify as needed
    #data_path = input("Enter the path to your data file: ")
    data = pd.read_csv(data_path)

    predictions = predict(model, data)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
