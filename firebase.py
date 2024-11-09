import firebase_admin
from firebase_admin import credentials
cred = credentials.Certificate("path later add")
firebase_admin.initialize_app(cred)
