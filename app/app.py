"""Flask website for Information-Extractor (Member 2 owns this file)."""
import time

from flask import Flask, jsonify, render_template, request

from extractor import extract

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    text = request.form.get("text", "")
    result, elapsed_ms = None, None
    if request.method == "POST":
        start = time.perf_counter()
        result = extract(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
    return render_template("index.html", text=text, result=result, elapsed_ms=elapsed_ms)


@app.post("/api/extract")
def api_extract():
    data = request.get_json(silent=True) or {}
    return jsonify(extract(data.get("text", "")))


if __name__ == "__main__":
    app.run(debug=True)
