// Information Extractor dashboard: input tabs, file upload, samples, history, theme,
// highlighting, summary, JSON/CSV export.
(function () {
  "use strict";

  var SAMPLES = [
    "Dayananda Sagar University is hosting its annual tech fest on 20th November, 2026 at 9:30 am. " +
      "The chief guest is Dr. Kavitha Menon from ISRO. Registration costs Rs. 300 per team. Details are " +
      "on www.dsu.edu.in, or write to fest@dsu.edu.in. Venue: Kumaraswamy Layout, Bengaluru 560078.",
    "Mumbai, 3 March 2026: Tata Motors reported a 14.5% rise in quarterly profit to ₹5,400 crore. " +
      "Chief executive Shailesh Chandra said exports to the United Kingdom grew 22 percent.",
    "Order #40213 from Flipkart has shipped. Expected delivery: 13th November 2026 before 8 p.m. " +
      "Deliver to Neha Gupta, 14 Park Street, Kolkata 700016. Amount paid: ₹2,349. " +
      "Call 98300 12345 for help.",
  ];
  var MAX_FILE = 1024 * 1024;
  var HISTORY_KEY = "ie-history";

  var $ = function (id) { return document.getElementById(id); };
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage blocked */ } },
    del: function (k) { try { localStorage.removeItem(k); } catch (e) { /* storage blocked */ } },
  };
  var esc = function (s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  };

  var form = $("extract-form");
  var textarea = $("text");
  var count = $("count");
  var submitBtn = $("submit-btn");

  // ------------------------------------------------------------- theme
  $("theme-btn").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    store.set("ie-theme", root.dataset.theme);
  });

  // ------------------------------------------------------------- input tabs
  var tabs = { paste: $("tab-paste"), upload: $("tab-upload") };
  function showTab(name) {
    Object.keys(tabs).forEach(function (k) {
      var on = k === name;
      tabs[k].classList.toggle("active", on);
      tabs[k].setAttribute("aria-selected", on ? "true" : "false");
      $("pane-" + k).hidden = !on;
    });
  }
  tabs.paste.addEventListener("click", function () { showTab("paste"); textarea.focus(); });
  tabs.upload.addEventListener("click", function () { showTab("upload"); });

  function updateCount() {
    var n = textarea.value.length;
    count.textContent = n.toLocaleString() + (n === 1 ? " character" : " characters");
  }
  textarea.addEventListener("input", updateCount);
  updateCount();

  document.querySelectorAll("[data-sample]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      textarea.value = SAMPLES[Number(btn.dataset.sample)];
      updateCount();
      textarea.focus();
    });
  });

  $("clear-btn").addEventListener("click", function () {
    textarea.value = "";
    updateCount();
    $("file-card").hidden = true;
    showTab("paste");
    textarea.focus();
  });

  textarea.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) form.requestSubmit();
  });

  // ------------------------------------------------------------- file upload
  var fileInput = $("file");
  var dropzone = $("dropzone");
  var uploadError = $("upload-error");

  function fmtSize(b) {
    return b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(0) + " KB" : (b / 1048576).toFixed(1) + " MB";
  }
  function fail(msg) { uploadError.textContent = msg; uploadError.hidden = false; }

  function loadFile(file) {
    uploadError.hidden = true;
    if (!file) return;
    var name = file.name.toLowerCase();
    if (/\.(pdf|docx?|rtf|odt)$/.test(name)) {
      return fail("PDF and Word files aren't supported yet. Open the file, copy the text and paste it in, or save it as .txt.");
    }
    if (!/\.(txt|md|csv|text)$/.test(name) && file.type.indexOf("text/") !== 0) {
      return fail("Please choose a plain-text file (.txt, .md or .csv).");
    }
    if (file.size > MAX_FILE) return fail("That file is " + fmtSize(file.size) + ". The limit is 1 MB.");
    var reader = new FileReader();
    reader.onload = function () {
      textarea.value = String(reader.result).slice(0, Number(textarea.maxLength) || 100000);
      updateCount();
      $("file-name").textContent = file.name;
      $("file-size").textContent = fmtSize(file.size) + " · loaded into the text box";
      $("file-card").hidden = false;
    };
    reader.onerror = function () { fail("Could not read that file."); };
    reader.readAsText(file);
  }
  fileInput.addEventListener("change", function () { loadFile(fileInput.files[0]); fileInput.value = ""; });
  ["dragenter", "dragover"].forEach(function (t) {
    dropzone.addEventListener(t, function (ev) { ev.preventDefault(); dropzone.classList.add("drag"); });
  });
  ["dragleave", "drop"].forEach(function (t) {
    dropzone.addEventListener(t, function () { dropzone.classList.remove("drag"); });
  });
  dropzone.addEventListener("drop", function (ev) {
    ev.preventDefault();
    loadFile(ev.dataTransfer.files[0]);
  });

  // ------------------------------------------------------------- history
  function readHistory() {
    try { return JSON.parse(store.get(HISTORY_KEY)) || []; } catch (e) { return []; }
  }
  function renderHistory() {
    var list = $("history-list");
    var items = readHistory();
    $("history").querySelector(".history-empty").hidden = items.length > 0;
    list.innerHTML = items.map(function (h, i) {
      var when = new Date(h.at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
      return '<li><button type="button" class="history-item" data-i="' + i + '"><span>' +
        esc(h.text.slice(0, 70)) + "</span><small>" + esc(when) + "</small></button></li>";
    }).join("") + (items.length ? '<li><button type="button" class="history-clear" id="history-clear">Clear history</button></li>' : "");
  }
  $("history-btn").addEventListener("click", function () {
    var panel = $("history");
    panel.hidden = !panel.hidden;
    this.setAttribute("aria-expanded", panel.hidden ? "false" : "true");
    if (!panel.hidden) renderHistory();
  });
  $("history-list").addEventListener("click", function (ev) {
    if (ev.target.closest("#history-clear")) { store.del(HISTORY_KEY); renderHistory(); return; }
    var btn = ev.target.closest(".history-item");
    if (!btn) return;
    textarea.value = readHistory()[Number(btn.dataset.i)].text;
    updateCount();
    form.requestSubmit();
  });

  form.addEventListener("submit", function () {
    var t = textarea.value.trim();
    if (t) {
      var items = readHistory().filter(function (h) { return h.text !== t; });
      items.unshift({ text: t.slice(0, 20000), at: Date.now() });
      store.set(HISTORY_KEY, JSON.stringify(items.slice(0, 8)));
    }
    submitBtn.disabled = true;
    submitBtn.querySelector("span").textContent = "Analyzing…";
  });

  // ------------------------------------------------------------- scroll spy
  var links = Array.prototype.slice.call(document.querySelectorAll("[data-spy]"));
  if ("IntersectionObserver" in window) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var id = en.target.id;
        links.forEach(function (l) { l.classList.toggle("active", l.dataset.spy === id); });
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    links.forEach(function (l) { var t = $(l.dataset.spy); if (t && t !== document.body) spy.observe(t); });
  }
  // Opening "NLP Steps" from the sidebar also expands it.
  var stepsLink = document.querySelector('[data-spy="steps"]');
  if (stepsLink && $("steps")) stepsLink.addEventListener("click", function () { $("steps").open = true; });

  // ================================================================ results
  var dataEl = $("result-data");
  if (!dataEl) return;
  var result = JSON.parse(dataEl.textContent);
  var e = result.entities || {};

  // Every found item with its type, in one list (used for highlights and CSV).
  var TYPES = [
    ["Person", "person", e.PERSON], ["Organization", "org", e.ORGANIZATION],
    ["Location", "place", [].concat(e.GPE || [], e.LOCATION || [], e.GSP || [])],
    ["Facility", "place", e.FACILITY], ["Date", "date", result.dates], ["Time", "date", result.times],
    ["Email", "contact", result.emails], ["Phone", "contact", result.phones], ["Link", "contact", result.urls],
    ["Amount", "money", result.money], ["Percentage", "money", result.percentages],
    ["PIN code", "money", result.pin_codes],
  ];
  var found = [];
  TYPES.forEach(function (t) {
    (t[2] || []).forEach(function (v) { if (v) found.push({ type: t[0], cls: t[1], v: v }); });
  });

  // ---- highlight every found item inside the original text (longest first, no overlaps)
  var target = $("highlighted");
  if (target) {
    var text = target.textContent;
    var owner = new Array(text.length).fill(null);
    found.slice().sort(function (a, b) { return b.v.length - a.v.length; }).forEach(function (it) {
      var from = 0, at;
      while ((at = text.indexOf(it.v, from)) !== -1) {
        var free = true;
        for (var i = at; i < at + it.v.length; i++) if (owner[i]) { free = false; break; }
        if (free) for (var j = at; j < at + it.v.length; j++) owner[j] = it;
        from = at + it.v.length;
      }
    });
    var html = "", k = 0;
    while (k < text.length) {
      var o = owner[k], end = k;
      while (end < text.length && owner[end] === o) end++;
      var part = esc(text.slice(k, end));
      html += o ? '<span class="hl ' + o.cls + '" title="' + o.type + '">' + part + "</span>" : part;
      k = end;
    }
    target.innerHTML = html;
  }

  // ---- extractive summary: score sentences by keywords and entities they contain
  var summary = $("summary");
  if (summary && result.nlp_steps && result.nlp_steps.length) {
    var steps = result.nlp_steps;
    var kw = {};
    (result.keywords || []).forEach(function (w, i) { kw[w.toLowerCase()] = (result.keywords.length - i); });
    var scored = steps.map(function (s, i) {
      var score = 0;
      (s.tokens || []).forEach(function (tok) { score += kw[tok.toLowerCase()] || 0; });
      found.forEach(function (it) { if (s.sentence.indexOf(it.v) !== -1) score += 3; });
      score = score / Math.sqrt(Math.max(4, (s.tokens || []).length)); // don't just favour long sentences
      return { i: i, s: s.sentence, score: score };
    });
    var keep = steps.length <= 2 ? 2 : steps.length <= 6 ? 2 : 3;
    var top = scored.slice().sort(function (a, b) { return b.score - a.score; }).slice(0, keep)
      .sort(function (a, b) { return a.i - b.i; });
    summary.textContent = top.map(function (t) { return t.s; }).join(" ");
    if (steps.length <= keep) $("summary-note").textContent = "The text is short, so all of it is shown.";
    else $("summary-note").textContent = "Extractive: " + top.length + " of " + steps.length +
      " sentences, picked by how many keywords and entities they contain.";
  }

  // ---- JSON colouring
  var pre = $("json");
  if (pre) {
    pre.innerHTML = esc(pre.textContent).replace(
      /(&quot;(?:[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(-?\d+(?:\.\d+)?|true|false|null)\b/g,
      function (m, str, colon, lit) {
        if (str) return '<span class="' + (colon ? "j-key" : "j-str") + '">' + str + "</span>" + (colon || "");
        return '<span class="j-num">' + lit + "</span>";
      });
  }

  // ---- POS tag meanings as tooltips
  var TAGS = {
    NN: "noun", NNS: "plural noun", NNP: "proper noun", NNPS: "plural proper noun",
    VB: "verb", VBD: "verb, past tense", VBG: "verb, -ing", VBN: "verb, past participle",
    VBP: "verb, present", VBZ: "verb, 3rd person present", JJ: "adjective", JJR: "comparative adjective",
    JJS: "superlative adjective", RB: "adverb", IN: "preposition", DT: "determiner", CC: "conjunction",
    CD: "number", PRP: "pronoun", "PRP$": "possessive pronoun", MD: "modal verb", TO: "to",
    WP: "wh-pronoun", WRB: "wh-adverb", POS: "possessive ending", ".": "sentence end", ",": "comma",
    ":": "colon", "$": "currency symbol",
  };
  document.querySelectorAll(".tok").forEach(function (t) {
    var tag = t.dataset.tag;
    if (TAGS[tag]) t.title = tag + " = " + TAGS[tag];
  });

  // ---- copy + export
  function download(name, type, content) {
    var url = URL.createObjectURL(new Blob([content], { type: type }));
    var a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }
  var copyBtn = $("copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText(pre.textContent).then(function () {
        copyBtn.innerHTML = '<svg><use href="#i-check"/></svg>';
        setTimeout(function () { copyBtn.innerHTML = '<svg><use href="#i-copy"/></svg>'; }, 1500);
      }, function () {});
    });
  }
  var expJson = $("export-json");
  if (expJson) expJson.addEventListener("click", function () {
    download("extraction.json", "application/json", JSON.stringify(result, null, 2));
  });
  var expCsv = $("export-csv");
  if (expCsv) expCsv.addEventListener("click", function () {
    var cell = function (v) {
      v = String(v);
      if (/^[=+\-@]/.test(v)) v = "'" + v; // stop spreadsheets running it as a formula
      return '"' + v.replace(/"/g, '""') + '"';
    };
    var rows = [["type", "value"]].concat(found.map(function (it) { return [it.type, it.v]; }));
    (result.keywords || []).forEach(function (k) { rows.push(["Keyword", k]); });
    download("extraction.csv", "text/csv",
      "﻿" + rows.map(function (r) { return r.map(cell).join(","); }).join("\r\n"));
  });
})();
