import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def insert_log(db, data: dict):
    db.collection("logs").document().set(data)
    pass


def init_firebase(projectId):
    path_to_key = os.path.join(os.path.dirname(__file__), "key.json")
    cred = credentials.Certificate(path_to_key)
    firebase_admin.initialize_app(
        cred,
        {
            "projectId": f"{projectId}",
        },
    )
    db = firestore.client()
    return db
