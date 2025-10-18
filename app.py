from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__, static_folder="static")

# Loaded summarization model
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"summary": "Please enter some text to summarize."})
    
    summary = summarizer(
        text,
        max_length=50,
        min_length=20,
        do_sample=False
    )
    return jsonify({"summary": summary[0]["summary_text"]})

@app.route("/")
def home():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(debug=True)
