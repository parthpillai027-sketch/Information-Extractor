# Notes for Claude Code

- Never add Co-Authored-By or Claude-Session lines to commits or pull requests.
- The NLP engine (app/extractor.py, app/gazetteer.py, setup_nltk.py, tests/, samples/, scripts/) belongs to S Krithik Yadav.
  The website (app/app.py, app/templates/, app/static/) belongs to the teammate: change it only when asked.
- Run `pytest -q` and `python scripts/evaluate.py` before committing. Keep the JSON keys in docs/json_contract.md.
