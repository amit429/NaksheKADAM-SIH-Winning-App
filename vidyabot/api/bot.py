from haystack.nodes import FARMReader
import os
import pandas as pd
import json
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
from flask import Flask, make_response, request
from googletrans import Translator
from flask_cors import CORS
from utils.fb import *

app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


def get_config():
    global languages
    file_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(file_path)
    config_path = os.path.join(parent_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    languages = handle_multilingual(pd.read_csv(config.get("QNA_CSV_PATH")))
    return config


def fetch_trans(question):
    global df, languages
    translations = dict()
    print(df)
    print(languages)
    for lang in languages:
        print(f'{list(df[df["content"] == question][f"{lang}"])=}')
        translations[lang] = list(df[df["content"] == question][f"{lang}"])[0]
    return translations


def gen_result(text):
    global config
    translator = Translator()
    detect_lang = translator.detect(text).lang
    dt1 = translator.translate(text, dest="en")
    translated_query = dt1.text

    global pipe
    prediction = pipe.run(
        query=translated_query,
        params={
            "Retriever": {"top_k": 3},
            "Reader": {"top_k": 1},
        },  # optimization parameters
    )
    print(type(prediction))
    # print(prediction)
    documents = prediction["documents"]
    print(documents[0])
    final_response = []
    for doc in documents:
        question = doc.content
        answer = doc.meta["answer_en"]
        print(question)
        print(answer)
        answers = fetch_trans(question)
        response = {
            "question": question,
            "answer": answer,
        }
        response.update(answers)
        final_response.append(response)

    logging_data = {
        "text": text,
        "question": final_response[0]["question"],
        "language": detect_lang,
    }
    if config.get("needs_logging"):
        insert_log(db, logging_data)
    return {"documents": final_response}


debug = True  # set to True to see the request and response


def debug_requests():
    global debug
    if debug:
        print(request)
        print(request.method)
        print(request.get_json())


def handle_multilingual(df):
    global languages
    # get all columns, which begin with "answer_"
    languages = [col for col in df.columns if col.startswith("answer_")]
    # create a new column "answers" that contains all answers
    return languages


def init_pipeline(config: dict):
    global df, languages
    QNA_CSV_PATH = config.get("QNA_CSV_PATH")
    model_path = config.get("model_path")
    model_name = config.get("model_name")
    document_store = InMemoryDocumentStore()
    df = pd.read_csv(QNA_CSV_PATH)
    df["question"] = (
        df["question"].apply(lambda x: x.strip()).apply(lambda x: x.replace("\n", " "))
    )
    for answer in languages:
        df[answer] = (
            df[answer].apply(lambda x: x.strip()).apply(lambda x: x.replace("\n", " "))
        )
    df = df.rename(columns={"question": "content"})
    docs_to_index = df.to_dict(orient="records")
    document_store.write_documents(docs_to_index)
    retriever = TfidfRetriever(document_store=document_store)
    # if model_path is not None and model exists
    if model_path is not None and os.path.exists(model_path):
        reader = FARMReader(model_name_or_path=model_path, use_gpu=False)
    else:
        reader = FARMReader(model_name_or_path=model_name, use_gpu=False)
        reader.save(model_path)
    pipe = ExtractiveQAPipeline(reader, retriever)
    return pipe


@app.route("/query", methods=["GET", "POST"])
def query():
    debug_requests()
    request_data = request.get_json()  # ["msg":"query"]
    text = request_data["msg"]
    result = gen_result(text)
    return make_response(result)


if __name__ == "__main__":
    global config
    config = get_config()
    tranlation_mode = config.get("translation_mode")
    needs_logging = config.get("needs_logging")
    pipe = init_pipeline(config)
    app.config["JSON_AS_ASCII"] = False
    if needs_logging:
        projectId = config.get("projectId")
        db = init_firebase(projectId)
    app.run(debug=True, host="0.0.0.0")
