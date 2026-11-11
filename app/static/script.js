// Information Extractor page: sample texts, character count, highlighting, copy JSON.
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

  var form = document.getElementById("extract-form");
  var textarea = document.getElementById("text");
  var count = document.getElementById("count");
  var submitBtn = document.getElementById("submit-btn");

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

  document.getElementById("clear-btn").addEventListener("click", function () {
    textarea.value = "";
    updateCount();
    textarea.focus();
  });

  // Ctrl+Enter submits.
  textarea.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) form.requestSubmit();
  });

  form.addEventListener("submit", function () {
    submitBtn.disabled = true;
    submitBtn.textContent = "Extracting…";
  });

  // ---------------------------------------------------------------- results
  var dataEl = document.getElementById("result-data");
  if (!dataEl) return;
  var result = JSON.parse(dataEl.textContent);

  var results = document.getElementById("results");
  if (results) results.scrollIntoView({ behavior: "smooth", block: "start" });

  // Highlight every found item inside the original text.
  var target = document.getElementById("highlighted");
  if (target) {
    var text = target.textContent;
    var items = [];
    var add = function (list, cls) {
      (list || []).forEach(function (v) { if (v) items.push({ v: v, cls: cls }); });
    };
    var e = result.entities || {};
    add(e.PERSON, "person");
    add(e.ORGANIZATION, "org");
    ["GPE", "LOCATION", "GSP", "FACILITY"].forEach(function (k) { add(e[k], "place"); });
    add(result.dates, "date");
    add(result.times, "date");
    add(result.emails, "contact");
    add(result.phones, "contact");
    add(result.urls, "contact");
    add(result.money, "money");
    add(result.percentages, "money");
    add(result.pin_codes, "money");
    items.sort(function (a, b) { return b.v.length - a.v.length; }); // longest first

    // Mark which characters are taken, so highlights never overlap.
    var owner = new Array(text.length).fill(null);
    items.forEach(function (it) {
      var from = 0, at;
      while ((at = text.indexOf(it.v, from)) !== -1) {
        var free = true;
        for (var i = at; i < at + it.v.length; i++) if (owner[i]) { free = false; break; }
        if (free) for (var j = at; j < at + it.v.length; j++) owner[j] = it;
        from = at + it.v.length;
      }
    });

    var esc = function (s) {
      return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    };
    var html = "", k = 0;
    while (k < text.length) {
      var o = owner[k], end = k;
      while (end < text.length && owner[end] === o) end++;
      var part = esc(text.slice(k, end));
      html += o ? '<span class="hl ' + o.cls + '">' + part + "</span>" : part;
      k = end;
    }
    target.innerHTML = html;
  }

  // POS tag meanings as tooltips.
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

  // Copy JSON.
  var copyBtn = document.getElementById("copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var json = document.getElementById("json").textContent;
      var done = function () {
        copyBtn.textContent = "Copied!";
        setTimeout(function () { copyBtn.textContent = "Copy JSON"; }, 1500);
      };
      if (navigator.clipboard) navigator.clipboard.writeText(json).then(done, function () {});
    });
  }
})();
