from panda import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Global variables for the model, scaler, and test data
svm_classifier = None
scaler = None
X_test = None  # Declare X_test globally
y_test = None  # Declare y_test globally

def load_data(file_path):
    """Loads a CSV file into a DataFrame."""
    df = read_csv(file_path)
    return df

def load_model():
    """Loads and trains the water quality model."""
    global svm_classifier, scaler, X_test, y_test  # Use global variables
    df = read_csv('api\\dataset\\water_quality_large_classified.csv')[['ph', 'turbidity', 'tds', 'temperature', 'probability']]
    X = df[['ph', 'turbidity', 'tds', 'temperature']].to_numpy()
    y = df['probability'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_classifier = SVC(kernel='rbf', random_state=42, probability=True)
    svm_classifier.fit(X_train_scaled, y_train)
    
    return X_test_scaled  # No need to return y_test or X_test

def predict_water_quality(ph, turbidity, tds, temperature):
    """Predicts whether water is acceptable based on given parameters.

    Parameters:
        ph (float): The pH of the water
        turbidity (float): The turbidity of the water
        tds (float): The total dissolved solids of the water
        temperature (float): The temperature of the water

    Returns:
        bool: True if the water is not acceptable (probability of 0), False if it is acceptable (probability of 1)

    """
    
    if svm_classifier is None or scaler is None:
        load_model()  # Ensure the model is loaded
    input_data_scaled = scaler.transform([[ph, turbidity, tds, temperature]])
    return svm_classifier.predict(input_data_scaled)[0] == 0

def get_model_accuracy():
    """Returns the accuracy of the SVM model in percent."""
    if svm_classifier is None:
        load_model()  # Ensure the model is loaded
    return accuracy_score(y_test, svm_classifier.predict(scaler.transform(X_test))) * 100

if __name__ == '__main__':
    print("Predicted water quality acceptable:", predict_water_quality(6.32, 1.82, 258.39, 23.67))
    print("Model accuracy:", get_model_accuracy(), "%")
