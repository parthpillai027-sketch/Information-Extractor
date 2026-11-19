"""Flask website + JSON API for Information-Extractor.

Routes
  GET  /               the website (app/templates/index.html)
  POST /               form post from the website (fields: text, smart)
  GET  /api/health     {"status": "ok", ...}: is the server up, is smart mode available
  POST /api/extract    {"text": "...", "smart": false} -> JSON in docs/json_contract.md
Errors on /api/* are JSON: {"error": "...", "status": 400}.
"""
import logging
import os
import time

import nltk
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

import llm_refiner
from extractor import MAX_CHARS, extract

MAX_INPUT_CHARS = 100_000          # longer requests are refused; NLP itself reads MAX_CHARS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024     # 2 MB request body
app.json.sort_keys = False                             # keep the key order of the contract
log = logging.getLogger(__name__)


def api_error(status, message):
    return jsonify({"error": message, "status": status}), status


def run(text, smart):
    start = time.perf_counter()
    result = extract(text, smart=smart)
    result["stats"]["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


@app.route("/", methods=["GET", "POST"])
def home():
    text = request.form.get("text", "")[:MAX_INPUT_CHARS]
    smart = request.form.get("smart", "").lower() in ("1", "true", "on", "yes")
    result = run(text, smart) if request.method == "POST" else None
    elapsed_ms = result["stats"]["elapsed_ms"] if result else None
    return render_template("index.html", text=text, result=result, elapsed_ms=elapsed_ms)


@app.get("/api/health")
def health():
    cfg = llm_refiner.config()
    return jsonify({"status": "ok", "nltk_version": nltk.__version__,
                    "smart_available": bool(cfg["key"]), "ai_provider": cfg["provider"],
                    "max_input_chars": MAX_INPUT_CHARS, "nlp_chars": MAX_CHARS})


@app.post("/api/extract")
def api_extract():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return api_error(400, 'Send JSON like {"text": "...", "smart": false} '
                              "with the header Content-Type: application/json.")
    text = data.get("text")
    if not isinstance(text, str):
        return api_error(400, '"text" is required and must be a string.')
    if len(text) > MAX_INPUT_CHARS:
        return api_error(400, f"Text is too long: {len(text):,} characters "
                              f"(the limit is {MAX_INPUT_CHARS:,}).")
    smart = data.get("smart", False)
    if not isinstance(smart, bool):
        return api_error(400, '"smart" must be true or false.')
    return jsonify(run(text, smart))


@app.errorhandler(HTTPException)
def http_error(err):
    if request.path.startswith("/api/"):
        return api_error(err.code, err.description)
    return err


@app.errorhandler(Exception)
def server_error(err):
    log.exception("Unhandled error")
    if request.path.startswith("/api/"):
        return api_error(500, "Internal error while extracting. Please try again.")
    return "Internal Server Error", 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
