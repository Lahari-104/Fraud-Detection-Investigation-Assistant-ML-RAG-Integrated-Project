# Fraud Detection & Investigation Assistant — Project Report

A system for fraud analysts that combines a machine-learning classifier over
tabular transaction data with a retrieval-augmented generation (RAG) pipeline.
Given a transaction, it produces a fraud flag, a probability, and a plain-English,
evidence-backed explanation that cites specific past cases and compliance
policies — not just a raw score.

All numbers in this report are measured, taken directly from the saved run
artifacts (`ml/artifacts/metrics.json`, `evaluation/retrieval_metrics.json`) on
a full local run with the real dataset, neural retrieval, and real Gemini
generation all active. Where a number is produced by an LLM judge that was not
run (RAGAS), that is stated plainly rather than estimated.

---

## 1. Problem statement

Fraud review teams face two connected problems. First, fraud is rare and
buried: in the reference dataset it is roughly 1 transaction in 578, so a model
that never flags anything is 99.83% "accurate" and catches zero fraud — accuracy
is actively misleading here. Second, even a good flag is only a number. An
analyst asked to action a flag needs to know *why*, grounded in prior cases and
written policy, and needs that reasoning recorded for audit.

This project addresses both: a classifier tuned and evaluated for the imbalance,
and a retrieval-grounded explanation layer that turns a flag into a cited,
inspectable rationale.

---

## 2. Architecture

```
Transaction (structured: Time, V1..V28, Amount)
        │
        ▼
   ML Classifier (XGBoost)  ──►  fraud flag + probability + SHAP attribution
        │
        ▼ (if flagged)
   Feature → natural-language query
     (amount band, derived hour, probability band, dominant anomaly components)
        │
        ▼
   Retrieval  (swappable backend: sentence-transformers+FAISS  |  TF-IDF+cosine)
        │
        ▼
   Top-k documents (past cases + policies)
        │
        ▼
   Generation  (Gemini via google-genai  |  deterministic offline stub)
     prompt: use ONLY retrieved context; cite every claim by document id
        │
        ▼
   Citation verification  (every cited id must be a retrieved id)
        │
        ▼
   Streamlit UI: flag, probability, drivers, explanation, cited sources
```

The two halves connect at exactly one point: the classifier's flag triggers the
explanation step, and the classifier's SHAP attribution supplies the anomaly
components that the query builder uses. Every stage after the classifier has a
swappable backend so the whole system is demonstrable offline and upgrades to
neural retrieval + real LLM generation locally with no code change.

---

## 3. Data

**Transactions.** The Kaggle "Credit Card Fraud Detection" dataset: 284,807
transactions with 492 frauds — a fraud rate of 0.1727%, i.e. about 1 in 578.
Features are `Time`, 28 anonymised PCA components `V1..V28`, and `Amount`. The
PCA anonymisation is central to several design decisions below, because the
components carry no human-readable meaning.

**Knowledge base.** 35 short documents generated for this project: 12 policy
statements and 23 case writeups. Every document is grounded strictly in features
derivable from the real dataset — amount, hour-of-day (from `Time`), anomaly
profile (which components deviate), and model probability. Fields the dataset
cannot support (country, merchant category, device, cardholder tenure) are
deliberately excluded, because a policy about them could never be matched by a
query built from the available features. The case set deliberately includes
confirmed-legitimate lookalikes (e.g. large legitimate purchases flagged as
false positives) so retrieval must distinguish fraud from fraud-lookalikes
rather than always returning fraud narratives.

Key statistics measured from the real data, used to ground the documents:

- Fraud median amount $9.25 vs legitimate $22.00.
- 36.8% of fraud is at or below $1.00, vs 10.7% of legitimate activity
  (the card-testing signature).
- Hours 00:00–05:59 carry roughly 3× the base fraud rate and about a quarter
  of all fraud.
- The strongest separating components are V17, V14, V12, V10 (all several
  standard deviations below their legitimate mean in fraud).

---

## 4. ML classifier — results

XGBoost, evaluated on a stratified 30% held-out test split (85,443 rows, 148
fraud). The decision threshold was tuned on a separate validation split, never
on the test split. Three imbalance strategies were compared head-to-head.

| Strategy | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| **SMOTE (train split only)** | **0.9569** | **0.7500** | **0.8409** | 0.8319 | 5 | 37 |
| Threshold tuning only | 0.8806 | 0.7973 | 0.8369 | 0.8423 | 16 | 30 |
| Class weighting | 0.9565 | 0.7432 | 0.8365 | 0.8417 | 5 | 38 |

Selected model: **SMOTE** (highest F1). Confusion matrix on the test split, at
the tuned threshold 0.9242:

```
        pred_legit  pred_fraud
  legit     85,290           5
  fraud         37         111
```

Accuracy is 0.99951 — but "always legitimate" scores 0.99827 on the same split,
so accuracy is reported only to show it is uninformative. The meaningful numbers
are precision, recall, and F1.

**Reading the result — the three strategies are tied within noise.** The
headline finding is not that SMOTE "won" but that all three strategies land
within 0.004 F1 of each other (0.8409 / 0.8369 / 0.8365). SMOTE took the top
slot on this run; a repeat run selected threshold-tuning instead, because
XGBoost training carries some nondeterminism and the margin is four thousandths
of an F1 point. The honest conclusion is that on this dataset, for a
gradient-boosted model that already produces usable probability rankings, the
imbalance-handling strategy barely matters once the decision threshold is tuned
— what matters is that the threshold is tuned at all (never left at 0.5) and
that accuracy is never used as the metric.

The strategies do differ in *where* they sit on the precision/recall trade-off,
and that is the useful thing to report. SMOTE and class weighting both push
precision to ~0.957 with only 5 false positives, at the cost of catching fewer
frauds (111 and 110 of 148). Threshold-tuning alone is more aggressive: it
catches the most frauds (118 of 148, the best recall at 0.797) but triples the
false positives to 16. So the choice is not "which is best" but "which error is
costlier for the deployment": if a missed fraud costs more than customer
friction from a false decline, the higher-recall threshold-only setting is
preferable; if false positives are expensive, SMOTE or class weighting. The
threshold is a single dial that moves along this curve.

Feature importance is dominated by V14 (0.61), with V4, V17, V10, V12 following
— consistent with the components that most separate fraud in the data.

**A note on the original build spec.** The spec said not to train on imbalanced
data "without addressing it," implying resampling or reweighting is essential.
The evidence here qualifies that: threshold tuning alone is competitive with
both SMOTE and class weighting on F1, so threshold selection is itself a valid
form of imbalance handling — arguably the one doing most of the work here. All
three arms are retained above as evidence rather than collapsing to a single
reported number.

---

## 5. Retrieval — results

Evaluated on a 14-query test set, each query hand-labelled with the documents
genuinely relevant to it *by meaning* (not keyword overlap — otherwise a lexical
method would win by construction). Each query has at least two relevant
documents so recall is measurable.

Both retrieval backends were evaluated on the identical test set:

| Metric | TF-IDF + cosine | Neural (MiniLM + FAISS) |
|---|---|---|
| Mean Precision@5 | **0.514** | 0.457 |
| Mean Recall@5 | **0.780** | 0.706 |
| MRR | 1.000 | 1.000 |
| Hit@5 | 1.000 | 1.000 |

**Reading the result — the lexical baseline is competitive, and that is the
finding.** Both backends put a relevant document first on every query (MRR
1.000) and find at least one relevant document for every query (Hit@5 1.000). On
Precision@5 and Recall@5, the TF-IDF baseline slightly *outscored* the neural
embeddings on this test set. That result is worth stating honestly rather than
assuming embeddings must win.

The reason is the query builder. Queries are composed from the documents' own
vocabulary — "micro-charge," "overnight," "zero-amount," "V14," "manual-hold" —
so the query and its target document share rare, distinctive words. TF-IDF is
built precisely for that literal overlap and does well. Neural embeddings match
on meaning, which is more robust to paraphrase but slightly fuzzier when the
exact keyword is already present: MiniLM sometimes ranks a semantically adjacent
document above the exact-keyword one (e.g. pulling a "daytime anomaly" case for
a "dominant anomaly components" query). Neither ranking is wrong; they encode
different notions of "similar."

(The saved `evaluation/retrieval_metrics.json` holds the TF-IDF baseline, which
reproduces offline with no model download; rerun the eval script with
`sentence-transformers` installed to regenerate the neural column.)

The defensible conclusion: on a small knowledge base with a keyword-rich query
builder, lexical retrieval is a genuinely strong baseline, and the neural
backend's advantage — robustness to paraphrase and natural-language queries —
would show up on a larger corpus and less templated queries, neither of which
this test set has. Two caveats bound both columns: 14 queries is a small set, so
the ~0.05 gap is within noise and neither backend is meaningfully better here;
and the relevance labels are hand-judged, with a few genuinely arguable calls
that would shift individual-query precision.

---

## 6. Generation and citation verification

The generation prompt instructs the model to use only the retrieved documents
and supplied transaction facts, to cite every claim by document id, and to state
explicitly when the context does not support a conclusion. It frames the output
as advice for analyst review, never an automated decline.

Crucially, the system does not trust the model's self-reported citations. After
generation, every cited id (`[POL-xx]`, `[CASE-xx]`) is extracted and verified
against the ids actually retrieved. A citation to a document that was not in
context is flagged as a hallucinated citation. This check caught a real bug
during development: an early version of the offline stub cited policies by habit
that had not been retrieved for that query; the verifier flagged them, and the
stub was corrected to cite only retrieved ids and adapt when an expected policy
is absent.

**Confirmed real Gemini generation.** The pipeline was run end-to-end with
`gemini-2.5-flash` on a real flagged fraud transaction (amount $99.99, 07:00,
p(fraud) 0.9997, V14-dominated). The model produced a complete, grounded
explanation that cited `[CASE-18]` and `[CASE-08]` — both in the retrieved set,
so the citation check passed — and used each correctly (CASE-18 for the
V14-driven confirmation, CASE-08 for the "flagged despite daytime timing" point).
Every factual claim (probability, threshold, drivers, timing) matched the
transaction; nothing outside the retrieved context was asserted. This is the
key validation: the citation-verification design works against a real language
model, not only the deterministic stub.

One practical finding from that run, worth recording. `gemini-2.5-flash` is a
reasoning model: at the original 700-token output cap it spent nearly the whole
budget on internal reasoning and emitted a truncated, citation-free answer. The
fix was to raise `max_output_tokens` and set the thinking budget to zero for
this task — restating retrieved evidence does not need chain-of-thought — after
which the full cited explanation appeared. Model choice matters here: pinned
stable versions (`gemini-2.5-flash`) are used rather than `-latest` aliases or
preview endpoints, both of which change or retire without notice (an earlier
`gemini-1.5-flash` reference 404'd for exactly that reason).

**Manual generation rubric.** Three deterministic checks that need no second
LLM: citation validity (zero hallucinated citations), factual grounding (stated
amount, time window, and probability band match the transaction), and citation
coverage (at least one retrieved document cited). These run against whichever
backend generated the text. Against the offline stub they pass by construction
(the stub reads its facts from the same values the rubric checks), so stub
passes are *not* evidence of quality; the rubric earns its keep against the
Gemini output, where a model can drift and these checks would catch it.

**RAGAS (LLM-judged) metrics.** Faithfulness, answer relevancy, and context
precision require an LLM judge. The harness is written, its dependency set is
pinned to a known-good combination (see `evaluation/requirements-ragas.txt`),
and it is wired to use Gemini as both judge and embeddings. It runs locally in a
separate environment with the key set. If those scores are not present in this
report, they were not run; they are intentionally absent rather than estimated.

---

## 7. Limitations

Stated plainly; this system is a working prototype, not production-ready.

1. **Velocity is a weak, honest proxy.** The dataset has no card identifier, so
   true per-card velocity is impossible. Burst/velocity policies and cases are
   written around processing-time proximity in the global ordering and are
   labelled "provisional pending account-level correlation." This is the honest
   version of the signal, not a real velocity feature.

2. **Synthetic knowledge base.** The 35 case and policy documents are generated
   for this project. They are grounded in real measured statistics, but they are
   not a real institution's case history or compliance manual. A production
   system would retrieve over vetted, maintained documents.

3. **Small knowledge base.** 35 documents is enough to demonstrate retrieval and
   generation but is far below what production retrieval quality needs. Precision
   figures should be read in that light.

4. **RAGAS faithfulness/relevance not yet scored.** Real Gemini generation has
   been validated end-to-end (grounded, correctly cited output), but the
   LLM-judged RAGAS metrics have not been run. Generation quality is so far
   evidenced by the manual rubric and manual inspection, not by RAGAS scores.

5. **Retrieval evaluated on a small, keyword-rich test set.** Both backends were
   measured (TF-IDF and neural), but on only 14 templated queries, where the
   ~0.05 gap between them is within noise and the lexical baseline is
   competitive. These numbers should not be read as a general verdict on
   lexical-vs-neural retrieval; a larger corpus and natural-language queries
   would likely favour the neural backend and are the right next test.

6. **PCA-anonymised features limit explanation richness.** V1..V28 carry no
   human meaning, so explanations reason about "dominant anomaly components"
   rather than concrete behaviours (a specific merchant, an impossible-travel
   event). The system is honest about this rather than inventing meaning.

7. **No temporal or concept-drift handling.** The model is trained and tested on
   one static dataset. Fraud patterns shift; a deployed model needs monitoring
   and retraining, which is out of scope here.

8. **Not a decision system.** Every output is framed as a prompt for human
   review. The system deliberately never auto-declines a transaction.

---

## 8. Suggested future improvements

- Run the RAGAS harness to add LLM-judged faithfulness/relevance scores to the
  generation evaluation (the one metric set not yet produced).
- Re-test retrieval on a larger corpus with paraphrased, natural-language
  queries, where the neural backend's paraphrase-robustness should show an
  advantage the current keyword-rich test set does not surface.
- Replace the synthetic knowledge base with a real, maintained corpus of cases
  and policies, and grow it substantially.
- Add a real velocity feature by joining on a card/account identifier if the
  source data supports it, replacing the processing-time proxy.
- Calibrate probabilities (e.g. isotonic) so the probability bands in the
  policies map to true likelihoods.
- Add drift monitoring and a scheduled retraining/evaluation loop.
- Human-in-the-loop feedback: let analysts confirm or correct flags and
  explanations, and feed that back into both the threshold and the knowledge
  base.

---

## 9. Reproducing the results

See `README.md` for full setup. In brief, from the project root with the main
environment installed and the real `creditcard.csv` in `data/transactions/`:

```
python ml/train.py                       # Section 4 numbers -> ml/artifacts/
python rag/build_knowledge_base.py       # the 35-document knowledge base
python rag/pipeline.py                   # end-to-end on one real transaction
python evaluation/evaluate_retrieval.py  # Section 5 numbers
python evaluation/evaluate_generation.py # manual rubric (+ RAGAS if key set)
streamlit run app/streamlit_app.py       # the UI
```

The retrieval and generation backends upgrade automatically when
`sentence-transformers` is installed and `GEMINI_API_KEY` is set.
