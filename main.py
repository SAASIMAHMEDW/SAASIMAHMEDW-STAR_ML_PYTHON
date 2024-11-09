from firebase import FireStore
from model import WaterQualityModel
import time


class STAR_ML:
    def __init__(self, document_path):
        self.app = FireStore()
        self.db = self.app.get_firestore_db_instance()
        self.model = WaterQualityModel()
        self.document_path = document_path  # Store the document path

    def on_snapshot(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            # Read the required fields from the document
            data = doc.to_dict()
            temperature = data.get("TEMPERATURE")
            turbidity = data.get("TURBIDITY")
            ph = data.get("PH")
            tds = data.get("TDS")

            # Ensure the fields are in the correct format
            if isinstance(temperature, (int, float)) and isinstance(turbidity, (int, float)) and isinstance(ph, (
            int, float)) and isinstance(tds, list) and tds:
                # Use the last element of the TDS array
                last_tds = tds[-1]

                # Print the data for verification
                print(
                    f"Data received - Temperature: {temperature}, Turbidity: {turbidity}, PH: {ph}, Last TDS: {last_tds}")

                # Run prediction using the WaterQualityModel
                prediction = self.model.predict_water_quality(ph, turbidity, last_tds, temperature)
                prediction = bool(prediction)

                # Update the PREDICTION field in Firestore
                doc_ref = self.db.document(self.document_path)
                doc_ref.update({"PREDICTION": prediction})
                print(f"Updated document {doc.id} with prediction: {prediction}")
            else:
                print("Invalid data format in document")

    def start_listener(self):
        # Setup listener for document changes
        doc_ref = self.db.document(self.document_path)
        doc_ref.on_snapshot(self.on_snapshot)
        print(f"Listening to changes on document {self.document_path}...")


if __name__ == "__main__":
    # Define the document path
    document_path = "INFORMATIONS/STAR_ML"

    # Create an instance of STAR_ML with the document path
    star_ml_instance = STAR_ML(document_path)

    # Start the listener
    star_ml_instance.start_listener()

    # Keep the program running
    while True:
        time.sleep(1)
