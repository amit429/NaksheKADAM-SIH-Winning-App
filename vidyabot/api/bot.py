from haystack.nodes import FARMReader
import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
from flask import Flask, make_response, request
from googletrans import Translator
from flask_cors import CORS
import os
from utils.fb import *

import openai
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
# app.config["JSON_AS_ASCII"] = False
app.json.ensure_ascii = False


def get_sentiment(text):
    global openai
    if openai is None:
        return None
    prompt = f"""Positive or negative sentiment,
    neutral : general queries
    positive : Confidence/Optimism
    negative : Fear/Pessimism/Anxiety/Pressure  
    sentiment for : {text=}"""
    response = openai.Completion.create(
        model="text-davinci-001",
        prompt=prompt,
        temperature=0.3,
        max_tokens=10,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
    )
    sentiment = response.choices[0].text.strip()
    print(sentiment)
    try:
        if "negative" in sentiment.lower():
            return "negative"
        elif "positive" in sentiment.lower():
            return "positive"
    except:
        return "neutral"


def init_haystack():
    document_store = InMemoryDocumentStore()
    if os.path.exists("models/roberta-base-squad2"):
        reader = FARMReader(
            model_name_or_path="models/roberta-base-squad2",
            use_gpu=False,
            context_window_size=500,
        )
    else:
        reader = FARMReader(
            model_name_or_path="deepset/roberta-base-squad2",
            use_gpu=False,
            context_window_size=500,
        )
        reader.save("models/roberta-base-squad2")
    retriever = TfidfRetriever(document_store)
    return document_store, reader, retriever


def get_docs_to_index():
    global df
    df = pd.read_csv("../csv/QnA.csv")
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip().replace("\t", " "))
    df = df.rename(columns={"question": "content"})
    df = df.drop_duplicates(subset=["content"])
    docs_to_index = df.to_dict(orient="records")
    return docs_to_index


def init_pipeline(reader, retriever):
    # Initialize the pipeline
    pipeline = ExtractiveQAPipeline(reader, retriever)
    return pipeline


def get_translation(text):
    global translator
    detect_lang = translator.detect(text).lang
    if detect_lang == "en":
        return text, detect_lang
    dt1 = translator.translate(text, dest="en")
    translated_query = dt1.text
    return translated_query, detect_lang


def translate_to_lang(text, dest):
    global translator
    dt1 = translator.translate(text, dest=dest)
    translated_query = dt1.text
    return translated_query


def get_query_result(query_txt):
    global pipeline, df
    query_txt, lang = get_translation(query_txt)
    predictions = pipeline.run(
        query=query_txt,
        params={
            "Retriever": {"top_k": 3},
            "Reader": {"top_k": 3},
        },  # optimization parameters
    )
    print(predictions["answers"][0].meta["answer_en"])
    final_response = []
    for k in predictions["answers"]:
        final_response.append(
            {f"answer_{lang}": translate_to_lang(k.meta["answer_en"], lang)}
        )
    return {"documents": final_response}


@app.route("/query", methods=["GET", "POST"])
def query():
    global db
    print(request)
    if request.method == "POST":
        query = request.json["query"]
        try:
            uid = request.json["uid"]
        except KeyError:
            uid = "anon"
        result = get_query_result(query)
        sentiment = get_sentiment(get_translation(query)[0])
        print(sentiment)
        print(result)
        insert_log(db, uid, query, sentiment)
        # return result, sentiment  # ask amit to parse result as [0] and sentiment [1]
        return {
            "result": result,
            "sentiment": sentiment,
        }
    else:
        return "Hello World"


@app.route("/hello", methods=["GET", "POST"])
def hello():
    return make_response("Hello World")


if __name__ == "__main__":
    load_dotenv()
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    if OPEN_AI_KEY == "YOUR_KEY_HERE":
        print("OpenAI key not found")
        openai = None
    else:
        openai.api_key = OPEN_AI_KEY

    translator = Translator()
    db = init_firebase()
    document_store, reader, retriever = init_haystack()
    docs_to_index = get_docs_to_index()
    document_store.write_documents(docs_to_index)
    pipeline = init_pipeline(reader, retriever)
    app.run(debug=True)
