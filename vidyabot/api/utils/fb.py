import os
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

path_to_key = os.path.join(os.path.dirname(__file__), "mkey.json")


def insert_log(db, uid, query, sentiment):
    if db:
        timestamp = datetime.datetime.now()

        db.collection("users").document(uid).collection("vidya").document(
            timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ).set(
            {
                "query": query,
                "timestamp": timestamp,
                "sentiment": sentiment,
            }
        )
        try:
            if sentiment.lower() == "negative":
                db.collection("users").document(uid).collection(
                    "negative_sentiments"
                ).document(timestamp.strftime("%Y-%m-%d %H:%M:%S")).set(
                    {
                        "query": query,
                        "timestamp": timestamp,
                        "sentiment": sentiment,
                    }
                )
                print(uid)

                db.collection("users").document(uid).update(
                    {
                        "negative_sentiments": True,
                    }
                )
        except:
            pass
    else:
        print("No db found")


def init_firebase():
    try:
        cred = credentials.Certificate(path_to_key)
    except FileNotFoundError:
        print("Firebase key not found")
        return None
    firebase_admin.initialize_app(
        cred,
        {
            "projectId": "nakshekadam2022",
        },
    )
    db = firestore.client()
    return db


if __name__ == "__main__":
    db = init_firebase()
    insert_log(db, "jjNm8BnaS1edpx9X2DWrJKEM8EU2", "bye", "positive")
    print("done")
