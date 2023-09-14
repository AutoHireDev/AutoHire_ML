from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)
model = SentenceTransformer("bert-base-nli-mean-tokens")


@app.route("/api/sbert-encode", methods=["POST"])
def sbert_encode():
    data = request.get_json()
    sentences = data["sentences"]

    sentence_embeddings = model.encode(sentences)
    encoded_sentences = sentence_embeddings.tolist()
    result = util.cos_sim(encoded_sentences[0], encoded_sentences[1])

    return jsonify(result.tolist())


if __name__ == "__main__":
    app.run()
