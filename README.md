# Fraud Detection & Investigation Assistant

An ML classifier + RAG pipeline for fraud analysts. Given a transaction, it
outputs a fraud flag, a probability, and a plain-English explanation that cites
specific past cases and compliance policies — with every citation verified
against what was actually retrieved.

For the full write-up (architecture, measured results, limitations) see
[`REPORT.md`](REPORT.md).

---

## What it does

```
Transaction ─► XGBoost classifier ─► flag + probability + SHAP drivers
                     │ (if flagged)
                     ▼
        feature → natural-language query
                     │
                     ▼
        retrieval over cases + policies  (neural or TF-IDF)
                     │
                     ▼
        LLM explanation using ONLY retrieved context  (Gemini or offline stub)
                     │
                     ▼
        citation verification → Streamlit UI
```

Every stage after the classifier has a **swappable backend**. The whole system
runs offline out of the box (TF-IDF retrieval + a deterministic stub) and
upgrades automatically to neural retrieval and real Gemini generation once the
optional pieces are in place — no code changes.

---

## Project layout

```
fraud-rag-project/
├── data/
│   ├── transactions/        # put the real creditcard.csv here
│   └── knowledge_base/       # 35 generated case + policy .txt files
├── ml/
│   ├── data_loader.py        # loads real CSV; synthesises a surrogate if absent
│   ├── train.py              # trains + compares 3 imbalance strategies
│   ├── predict.py            # scoring + SHAP attribution (reusable)
│   └── artifacts/            # saved model + metrics.json
├── rag/
│   ├── build_knowledge_base.py  # (re)generates the knowledge base
│   ├── retriever.py          # query builder + swappable retrieval backend
│   ├── generator.py          # prompt + Gemini client + offline stub + citations
│   └── pipeline.py           # end-to-end orchestrator (used by the UI)
├── evaluation/
│   ├── evaluate_retrieval.py    # retrieval precision/recall/MRR (offline)
│   ├── evaluate_generation.py   # manual rubric + RAGAS harness
│   ├── requirements-ragas.txt   # pinned, known-good RAGAS dependency set
│   └── retrieval_metrics.json   # saved retrieval numbers
├── app/
│   └── streamlit_app.py      # the UI
├── requirements.txt          # main environment
├── REPORT.md                 # project report
└── README.md
```

---

## Setup

### 1. Main environment

Python 3.10+ recommended. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs the classifier, retrieval, generation, and UI dependencies. The
system is fully runnable at this point using the offline backends.

### 2. The dataset

Download the Kaggle "Credit Card Fraud Detection" dataset and place the file at:

```
data/transactions/creditcard.csv
```

The loader uses the real file whenever it is present. If it is absent, it
generates a clearly-labelled synthetic surrogate with the identical schema so
the pipeline still runs — but any metrics then describe the surrogate, not real
fraud detection, and the code prints a loud banner saying so. Drop in the real
file and everything downstream uses it automatically.

### 3. (Optional) Neural retrieval

For semantic retrieval instead of the TF-IDF fallback:

```bash
pip install sentence-transformers faiss-cpu
```

The retriever detects this automatically. On first use it downloads the
`all-MiniLM-L6-v2` weights (needs internet once), then swaps from TF-IDF to
neural embeddings with no code change. Relevance scores will be on a different
scale than TF-IDF — that is expected; compare rankings, not raw numbers.

### 4. (Optional) Real Gemini generation

For real LLM explanations instead of the offline stub:

```bash
pip install google-genai
export GEMINI_API_KEY="your-key-here"     # or GOOGLE_API_KEY
```

Do **not** hard-code the key or commit it. Use the environment variable or a
git-ignored `.env`. The generator reads `GEMINI_API_KEY` first, then
`GOOGLE_API_KEY`. Once set, the pipeline swaps from stub to Gemini
automatically. The default model is `gemini-1.5-flash` (one line to change at
the top of `rag/generator.py`).

---

## Running everything

From the project root, with the main environment active:

```bash
# 1. Train the classifier — writes ml/artifacts/{fraud_model.joblib, metrics.json}
python ml/train.py

# 2. (Re)generate the knowledge base — writes data/knowledge_base/*.txt
python rag/build_knowledge_base.py

# 3. Phase-by-phase demos
python ml/predict.py                      # score sample transactions + SHAP
python rag/retriever.py                   # retrieval on 3 sample scenarios
python rag/generator.py                   # generation on 2 sample scenarios
python rag/pipeline.py                    # full end-to-end on one real transaction

# 4. Evaluation
python evaluation/evaluate_retrieval.py   # retrieval P@k / R@k / MRR (offline)
python evaluation/evaluate_generation.py  # manual rubric (+ RAGAS if key is set)

# 5. The UI
streamlit run app/streamlit_app.py
```

The Streamlit app shows a live backend banner (data source, retriever, and
generator), so you always know whether you are looking at stub or real-LLM
output, TF-IDF or neural retrieval.

---

## The RAGAS evaluation environment (separate venv)

RAGAS computes LLM-judged faithfulness and relevance. Its current releases have
a broken dependency chain against recent LangChain, so this project pins a
**known-good combination** in `evaluation/requirements-ragas.txt`. Those pins
downgrade LangChain, so install them in a **separate virtual environment** to
avoid clashing with the main one.

```bash
python -m venv .venv-ragas
source .venv-ragas/bin/activate
pip install -r evaluation/requirements-ragas.txt
export GEMINI_API_KEY="your-key-here"

# run the generation eval; the RAGAS section now executes instead of skipping
python evaluation/evaluate_generation.py
```

Notes:

- The harness uses Gemini as both the judge model and the embeddings, and passes
  your key explicitly, so `GEMINI_API_KEY` works throughout (the RAGAS judge
  otherwise only reads `GOOGLE_API_KEY`).
- Expect the RAGAS section to take a minute or two and to consume a little Gemini
  quota (one generation per scenario plus several judge/embedding calls per
  metric). If it appears to hang, it is almost always a network or quota issue on
  the judge calls, not a logic error.
- Without the key, `evaluate_generation.py` still runs — it prints the manual
  rubric and cleanly skips the RAGAS section with a message.

---

## Notes on honesty of results

- Classifier metrics in `REPORT.md` are from the **real** dataset.
- Retrieval metrics are from the **TF-IDF** backend; rerun locally with
  `sentence-transformers` to get the neural comparison.
- Generation output and the manual rubric currently reflect the **offline stub**
  unless the Gemini key is set. RAGAS numbers require the key and are absent
  (not estimated) until then.
- The knowledge base is **synthetic** but grounded in real measured statistics.
  See the limitations section of `REPORT.md`.
