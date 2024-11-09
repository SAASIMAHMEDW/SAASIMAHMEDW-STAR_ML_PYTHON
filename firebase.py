from firebase_admin import credentials, firestore, initialize_app, _apps


class FireStore:
    def __init__(self):
        self.db = self._initialize()

    @staticmethod
    def _initialize():
        if not _apps:
            print("INITIALIZING APP...")
            cred = credentials.Certificate("config/eco-gaurdian-bot-firebase-adminsdk.json")
            initialize_app(cred)
            print("INITIALIZED APP")
        return firestore.client()

    def get_firestore_db_instance(self):
        return self.db
