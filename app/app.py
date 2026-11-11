"""Flask website for Information-Extractor (Member 2 owns this file)."""
from flask import Flask, jsonify, render_template, request

from extractor import extract

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    text = request.form.get("text", "")
    result = extract(text) if request.method == "POST" else None
    return render_template("index.html", text=text, result=result)


@app.post("/api/extract")
def api_extract():
    data = request.get_json(silent=True) or {}
    return jsonify(extract(data.get("text", "")))


if __name__ == "__main__":
    app.run(debug=True)
