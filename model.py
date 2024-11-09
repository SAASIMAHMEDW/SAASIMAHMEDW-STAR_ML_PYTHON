from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class WaterQualityModel:
    def __init__(self, file_path="dataset/water_quality_large_classified.csv"):
        self.file_path = file_path
        self.svm_classifier = None
        self.scaler = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        """Loads a CSV file into a DataFrame."""
        return read_csv(self.file_path)

    def prepare_data(self):
        """Prepares the dataset and splits it into training and testing sets."""
        df = self.load_data()[['ph', 'turbidity', 'tds', 'temperature', 'probability']]
        X = df[['ph', 'turbidity', 'tds', 'temperature']].to_numpy()
        y = df['probability'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_test, self.y_test = X_test_scaled, y_test
        return X_train_scaled, y_train

    def load_model(self):
        """Loads and trains the water quality model if not already trained."""
        if self.svm_classifier is None:
            X_train, y_train = self.prepare_data()
            self.svm_classifier = SVC(kernel='rbf', random_state=42, probability=True)
            self.svm_classifier.fit(X_train, y_train)

    def predict_water_quality(self, ph, turbidity, tds, temperature):
        """Predicts water quality based on input parameters."""
        self.load_model()  # Ensure model is loaded
        input_data_scaled = self.scaler.transform([[ph, turbidity, tds, temperature]])
        return self.svm_classifier.predict(input_data_scaled)[0] == 0

    def get_model_accuracy(self):
        """Returns the accuracy of the SVM model in percent."""
        self.load_model()  # Ensure model is loaded
        predictions = self.svm_classifier.predict(self.X_test)
        return accuracy_score(self.y_test, predictions) * 100


if __name__ == '__main__':
    model = WaterQualityModel()
    print("Predicted water quality acceptable:", model.predict_water_quality(8, 2, 198, 30))
    print("Model accuracy:", model.get_model_accuracy(), "%")
