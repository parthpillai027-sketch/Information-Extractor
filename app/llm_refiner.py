"""Optional AI layer ("smart mode"): an LLM checks and corrects the NLTK result.

How it works
  1. NLTK always runs first (extractor.extract). This module never replaces it.
  2. Only when the request asks for smart mode AND the server has LLM_API_KEY set, the text
     plus NLTK's typed items and facts are sent to the LLM, which returns corrected JSON.
  3. The answer is validated strictly: allowed types only, and every piece of text must be
     copied exactly from the input text (so the LLM cannot invent facts).
  4. If the provider is busy (HTTP 5xx, common on free tiers) or does not answer in time, the
     call is tried once more within the same time limit (Gemini: with a backup model).
     Any other problem (no key, timeout after LLM_TIMEOUT seconds, HTTP error, bad JSON) -> the
     NLTK result is kept and "ai.error" says why. The website never breaks because of the LLM.

Settings (environment variables, or a .env file in the project folder; see .env.example):
  LLM_API_KEY    required to enable smart mode (never commit it)
  LLM_PROVIDER   "xai" (default, Grok), "gemini", or "openai" (any OpenAI-compatible API)
  LLM_MODEL      model name. Empty = ask the provider's /models list and pick a Grok text
                 model (xai), or "gemini-3.1-flash-lite" (Gemini's fast lite model)
  LLM_BASE_URL   API address for xai/openai; default https://api.x.ai/v1 for xai
  LLM_TIMEOUT    seconds for the whole AI step (both tries), default 25
"""
import functools
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

try:                                    # read LLM_* settings from ./.env when present
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:                     # python-dotenv not installed: use real env vars only
    pass

ALLOWED_TYPES = {"PERSON", "PLACE", "ORGANIZATION", "DATE", "TIME", "MONEY", "PERCENT",
                 "CONTACT", "THING"}
LIST_SLOTS = ["who", "what", "where", "when", "how_much", "contact", "affiliation"]
DEFAULT_BASE_URL = {"xai": "https://api.x.ai/v1", "openai": "https://api.openai.com/v1"}
MAX_TEXT_FOR_LLM = 8000
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# The "flash-latest" alias points to a slower thinking model (often 10-30 s); the lite model
# answered in about 5-10 s with the same corrections in our tests.
GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_BACKUP_MODEL = "gemini-flash-lite-latest"   # used for the second try when the first is busy
DEFAULT_TIMEOUT = 25.0                  # total for both tries; one answer usually takes 5-12 s
BUSY_CODES = {500, 502, 503, 504}       # provider overloaded: worth one more try
FIRST_TRY_SHARE = 0.6                   # 25 s -> first try 15 s, second try up to ~10 s
MIN_RETRY_SECONDS = 5

SYSTEM_PROMPT = """You check the output of an NLTK information-extraction system for English text
(often Indian names, places, deities, organisations, notices and messages).

You receive the TEXT, the ITEMS NLTK found (text + type) and the FACTS it built.
Return ONLY a JSON object:
{"items": [{"text": "...", "type": "..."}],
 "facts": [{"who": [], "action": "", "what": [], "where": [], "when": [], "how_much": [], "contact": []}]}

Rules:
- type is one of PERSON, PLACE, ORGANIZATION, DATE, TIME, MONEY, PERCENT, CONTACT, THING.
- Gods, goddesses and epic characters are PERSON (keep titles like "Lord", "Goddess", "Sri").
  Mountains, rivers, lakes, temples, cities, states and countries are PLACE.
- Fix wrong types, remove items that are not real names/values, add clearly missed ones.
- Every "text" and every list value MUST be copied exactly, character for character, from TEXT.
  Do not translate, expand, guess or add information that is not written in TEXT.
- One fact per verb/action. who = the doer (person or organisation), action = the verb words
  as written, what = object, where = place, when = date/time, how_much = money/percent/amount,
  contact = email/phone/link.
- Output JSON only, no explanation."""


def config():
    provider = (os.environ.get("LLM_PROVIDER") or "xai").strip().lower()
    if provider == "grok":
        provider = "xai"
    default_model = GEMINI_DEFAULT_MODEL if provider == "gemini" else ""
    try:
        timeout = float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    return {
        "provider": provider,
        "key": (os.environ.get("LLM_API_KEY") or "").strip(),
        "model": (os.environ.get("LLM_MODEL") or default_model).strip(),
        "base_url": (os.environ.get("LLM_BASE_URL")
                     or DEFAULT_BASE_URL.get(provider, DEFAULT_BASE_URL["openai"])).rstrip("/"),
        "timeout": timeout,
    }


def available():
    """True when smart mode can run (an API key is configured)."""
    return bool(config()["key"])


def _post_json(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers=dict(headers, **{"Content-Type": "application/json"}))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@functools.lru_cache(maxsize=4)
def _pick_model(base_url, key, timeout):
    """No LLM_MODEL set: ask the provider which models this key can use and pick a text one."""
    req = urllib.request.Request(base_url + "/models", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ids = [m.get("id", "") for m in json.loads(resp.read().decode("utf-8")).get("data", [])]
    skip = ("image", "imagine", "vision", "video", "embed", "audio", "tts", "voice")
    text_models = [i for i in ids if not any(w in i.lower() for w in skip)]
    grok = [i for i in text_models if "grok" in i.lower()]
    for pool in (grok, text_models):
        fast = [i for i in pool if "fast" in i.lower() or "mini" in i.lower()]
        if fast or pool:
            return (fast or pool)[0]
    raise ValueError("the provider returned no usable model; set LLM_MODEL")


def _call_llm(system, user, cfg):
    """Send one request to the configured provider and return the model's text answer."""
    if cfg["provider"] == "gemini":
        data = _post_json(
            GEMINI_URL.format(model=cfg["model"]),
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"role": "user", "parts": [{"text": user}]}],
             "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
            {"x-goog-api-key": cfg["key"]}, cfg["timeout"])
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    if cfg["provider"] in ("xai", "openai"):            # OpenAI-compatible chat completions
        if not cfg["model"]:
            cfg["model"] = _pick_model(cfg["base_url"], cfg["key"], cfg["timeout"])
        data = _post_json(
            cfg["base_url"] + "/chat/completions",
            {"model": cfg["model"], "temperature": 0, "response_format": {"type": "json_object"},
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            {"Authorization": "Bearer " + cfg["key"]}, cfg["timeout"])
        return data["choices"][0]["message"]["content"]
    raise ValueError(f"unknown LLM_PROVIDER '{cfg['provider']}'")


def _parse_json(answer):
    answer = answer.strip()
    if answer.startswith("```"):                       # ```json ... ``` fences
        answer = answer.strip("`")
        answer = answer[answer.find("{"):]
    start, end = answer.find("{"), answer.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the answer")
    return json.loads(answer[start:end + 1])


def validate(data, text):
    """Keep only well-formed items/facts whose text really appears in the input."""
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("answer has no 'items' list")
    items, seen = [], set()
    for it in data["items"]:
        if not isinstance(it, dict):
            continue
        value, typ = it.get("text"), str(it.get("type", "")).upper()
        if isinstance(value, str) and value.strip() and value in text and typ in ALLOWED_TYPES:
            key = (value.lower(), typ)
            if key not in seen:
                seen.add(key)
                items.append({"text": value, "type": typ})
    facts = []
    for f in data.get("facts") or []:
        if not isinstance(f, dict):
            continue
        clean = {"who": [], "action": "", "what": [], "where": [], "when": [], "how_much": [],
                 "contact": [], "affiliation": []}
        action = f.get("action") or ""
        if isinstance(action, str) and action.lower() in text.lower():
            clean["action"] = action
        for slot in LIST_SLOTS:
            values = f.get(slot) or []
            if isinstance(values, str):
                values = [values]
            clean[slot] = [v for v in values if isinstance(v, str) and v.strip() and v in text]
        if clean["action"] or any(clean[s] for s in LIST_SLOTS):
            facts.append(clean)
    return items, facts


def _apply(result, items, facts):
    """Replace items/entities/facts with the checked AI answer and list what changed."""
    old = [it for it in result["items"] if it["type"] != "THING"]
    old_type = {it["text"].lower(): it for it in old}
    new_type = {it["text"].lower(): it["type"] for it in items}
    corrections = []
    for it in items:
        before = old_type.get(it["text"].lower())
        if it["type"] != "THING" and (before is None or before["type"] != it["type"]):
            corrections.append({"text": it["text"], "nltk_type": before["type"] if before else None,
                                "final_type": it["type"]})
    for it in old:
        if it["text"].lower() not in new_type:
            corrections.append({"text": it["text"], "nltk_type": it["type"], "final_type": None})

    merged = []
    for it in items:
        before = old_type.get(it["text"].lower())
        if before and before["type"] == it["type"]:
            merged.append(before)
        else:
            merged.append({"text": it["text"], "type": it["type"], "label": it["type"].lower(),
                           "source": "ai"})
    merged += [it for it in result["items"] if it["type"] == "THING"
               and it["text"].lower() not in new_type]
    result["items"] = merged

    entities = {}
    for it in merged:
        if it["type"] in ("PERSON", "ORGANIZATION"):
            entities.setdefault(it["type"], []).append(it["text"])
        elif it["type"] == "PLACE" and it.get("label") != "pin_code":
            label = it["label"] if it.get("label") in ("GPE", "LOCATION", "FACILITY", "GSP") else "GPE"
            entities.setdefault(label, []).append(it["text"])
    result["entities"] = {k: list(dict.fromkeys(v)) for k, v in entities.items()}
    if facts:
        result["facts"] = facts
    result["corrections"] = corrections


def _is_timeout(err):
    return isinstance(err, (socket.timeout, TimeoutError)) or (
        isinstance(err, urllib.error.URLError) and isinstance(err.reason, socket.timeout))


def _ask(call, system, user, cfg):
    """Call the LLM, with at most one more try inside the same time limit (LLM_TIMEOUT).

    The first try gets 60% of the time. If the provider is busy (HTTP 5xx) or does not answer
    in that time, the second try gets what is left; for Gemini it uses a backup model, which
    has its own capacity. Other errors (wrong key, bad model, rate limit) are not retried."""
    total, start = cfg["timeout"], time.perf_counter()
    cfg["timeout"] = total * FIRST_TRY_SHARE
    try:
        return call(system, user, cfg)
    except Exception as err:
        busy = isinstance(err, urllib.error.HTTPError) and err.code in BUSY_CODES
        left = total - (time.perf_counter() - start) - (1 if busy else 0)
        if not (busy or _is_timeout(err)) or left < MIN_RETRY_SECONDS:
            raise
    if busy:
        time.sleep(1)
    if cfg["provider"] == "gemini" and cfg["model"] != GEMINI_BACKUP_MODEL:
        cfg["model"] = GEMINI_BACKUP_MODEL
    cfg["timeout"] = left
    return call(system, user, cfg)


def refine(text, result, call=None):
    """Improve `result` in place with the LLM. Never raises; always leaves a usable result."""
    cfg = config()
    ai = {"requested": True, "used": False, "provider": cfg["provider"], "model": cfg["model"],
          "error": None, "seconds": None}
    result["ai"] = ai
    result["mode"] = "nltk"
    if not cfg["key"]:
        ai["error"] = "smart mode is off on this server (no LLM_API_KEY)"
        return result
    payload = {
        "text": text[:MAX_TEXT_FOR_LLM],
        "items": [{"text": it["text"], "type": it["type"]} for it in result["items"]][:150],
        "facts": [{k: f.get(k, []) for k in ["who", "action"] + LIST_SLOTS[1:]}
                  for f in result["facts"]][:40],
    }
    start, limit = time.perf_counter(), cfg["timeout"]
    try:
        answer = _ask(call or _call_llm, SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False), cfg)
        items, facts = validate(_parse_json(answer), text)
        if not items:
            raise ValueError("AI answer had no valid items")
        _apply(result, items, facts)
        ai["used"] = True
        ai["model"] = cfg["model"]
        result["mode"] = "hybrid"
    except urllib.error.HTTPError as err:
        if err.code in (400, 401, 403, 404):
            hint = " (check LLM_MODEL / LLM_API_KEY)"
        elif err.code == 429:
            hint = " (rate limit reached; wait a minute)"
        elif err.code in BUSY_CODES:
            hint = " (the AI service is busy; try again)"
        else:
            hint = ""
        ai["error"] = f"AI request failed: HTTP {err.code}{hint}"
    except (socket.timeout, TimeoutError):
        ai["error"] = f"AI request timed out after {limit:g}s"
    except urllib.error.URLError as err:
        reason = err.reason if not isinstance(err.reason, socket.timeout) else "timeout"
        ai["error"] = f"AI request failed: {reason}"
    except Exception as err:                      # bad JSON, wrong shape, anything else
        ai["error"] = f"AI answer rejected: {err}"
    ai["seconds"] = round(time.perf_counter() - start, 2)
    return result
