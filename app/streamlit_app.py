"""
Phase 6 -- Streamlit UI.

Run:  streamlit run app/streamlit_app.py

Drives the SAME pipeline as everything else (ml.predict -> rag.retriever ->
rag.generator), so whatever backends are active on your machine -- TF-IDF or
neural retrieval, stub or Gemini generation -- are what the UI shows. The header
states the live backend configuration so a viewer always knows whether they are
looking at stub or real-LLM output.

Three ways to choose a transaction:
  1. Sample a known-fraud row from the dataset
  2. Sample a known-legitimate row
  3. Enter Amount + hour manually (V-components default to a legit-typical
     profile; this path is for feeling out the amount/time policies, not for
     precise anomaly scoring)
"""

from __future__ import annotations

import os
import sys

# make the sibling packages importable when run via `streamlit run app/...`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("ml", "rag"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection & Investigation Assistant",
                   page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------- #
# Styling (presentation only — no logic depends on any of this)
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  /* ---- type & spacing -------------------------------------------------- */
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
  h1, h2, h3 { letter-spacing: -0.01em; }

  /* ---- app header ------------------------------------------------------ */
  .app-header {
    background: linear-gradient(135deg, #1a2744 0%, #243b6b 100%);
    border-radius: 16px; padding: 1.6rem 1.9rem; margin-bottom: 0.4rem;
    border: 1px solid rgba(255,255,255,0.06);
  }
  .app-header h1 { margin: 0; font-size: 1.6rem; color: #f4f7ff; }
  .app-header p  { margin: 0.35rem 0 0; color: #aab8d6; font-size: 0.95rem; }

  /* ---- status pills (plain-language config banner) --------------------- */
  .pill-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.9rem 0 0.2rem; }
  .pill {
    background: rgba(120,150,220,0.10); border: 1px solid rgba(120,150,220,0.25);
    border-radius: 999px; padding: 0.32rem 0.85rem; font-size: 0.82rem;
    color: #c7d4f0; white-space: nowrap;
  }
  .pill b { color: #eaf0ff; font-weight: 600; }

  /* ---- verdict banners (the focal point) ------------------------------- */
  .verdict {
    border-radius: 14px; padding: 1.3rem 1.6rem; margin: 0.2rem 0 0.4rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .verdict-flag { background: linear-gradient(135deg,#3a1220,#5c1a2e);
                  border: 1px solid #a83b57; }
  .verdict-clear{ background: linear-gradient(135deg,#0f2c22,#143d30);
                  border: 1px solid #2f9e78; }
  .verdict .icon { font-size: 2.1rem; line-height: 1; }
  .verdict .label{ font-size: 1.35rem; font-weight: 700; color: #fff; margin: 0; }
  .verdict .sub  { font-size: 0.9rem; color: rgba(255,255,255,0.75); margin: 0.15rem 0 0; }

  /* ---- explanation card ------------------------------------------------ */
  .explain-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid #4f7cff; border-radius: 12px;
    padding: 1.25rem 1.5rem; font-size: 1.02rem; line-height: 1.65; color: #e8ecf5;
  }

  /* ---- verified badge -------------------------------------------------- */
  .verified { color: #46c48a; font-weight: 600; }
  .unverified { color: #e5883b; font-weight: 600; }

  /* soften Streamlit's default metric labels */
  [data-testid="stMetricLabel"] { opacity: 0.75; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Plain-language labels for the backend config (presentation only)
# --------------------------------------------------------------------------- #
def _retrieval_label(backend: str) -> str:
    if "sentence-transformers" in backend or "faiss" in backend:
        return "Semantic search"
    if "tfidf" in backend:
        return "Keyword search"
    return backend


def _model_label(backend: str) -> str:
    if backend == "offline-stub":
        return "Built-in explainer (no AI key)"
    if "gemini" in backend:
        # "gemini:gemini-2.5-flash" -> "Gemini 2.5"
        tail = backend.split(":")[-1]
        if "2.5" in tail:
            return "Gemini 2.5"
        if "2.0" in tail:
            return "Gemini 2.0"
        return "Gemini"
    if "anthropic" in backend:
        return "Claude"
    return backend


def _data_label(source: str) -> str:
    return "Real transactions" if source == "real" else "Sample data"


# --------------------------------------------------------------------------- #
# Cached resources -- built once per session
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model, retriever, and knowledge base...")
def get_pipeline():
    from pipeline import FraudRAGPipeline
    return FraudRAGPipeline()


@st.cache_data(show_spinner="Loading transactions...")
def get_sample_rows(n_each: int = 200):
    """A modest sample of fraud/legit rows for the pickers (not the full file)."""
    from data_loader import load_transactions
    df, info = load_transactions(verbose=False)
    fraud = df[df.Class == 1]
    legit = df[df.Class == 0].sample(min(n_each, (df.Class == 0).sum()),
                                     random_state=1)
    return (pd.concat([fraud, legit]).reset_index(drop=False)
              .rename(columns={"index": "row_id"}),
            info.source, len(df), int(df.Class.sum()))


FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _legit_typical_row(amount: float, hour: int) -> pd.Series:
    """Build a plausible legitimate-profile row for the manual-entry path.
    V-components are set near zero (the legit mean) so the manual path exercises
    amount/time logic without pretending to a real anomaly profile."""
    row = {c: 0.0 for c in FEATURE_COLS}
    row["Amount"] = float(amount)
    row["Time"] = float(hour * 3600 + 1800)  # mid-hour
    return pd.Series(row)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_verdict(pred):
    if pred.is_flagged:
        st.markdown(f"""
        <div class="verdict verdict-flag">
          <div class="icon">🚩</div>
          <div>
            <p class="label">Flagged for review</p>
            <p class="sub">Fraud likelihood {pred.probability*100:.1f}% &nbsp;·&nbsp;
               ${pred.amount:,.2f} at {pred.hour_of_day:02d}:00</p>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict verdict-clear">
          <div class="icon">✓</div>
          <div>
            <p class="label">Not flagged</p>
            <p class="sub">Fraud likelihood {pred.probability*100:.1f}% &nbsp;·&nbsp;
               ${pred.amount:,.2f} at {pred.hour_of_day:02d}:00</p>
          </div>
        </div>""", unsafe_allow_html=True)


def render_drivers(pred):
    st.caption("These are anonymized transaction-pattern features (the dataset's "
               "privacy-protected signals) that most influenced the score — not "
               "raw account numbers or personal data. A higher value means the "
               "feature pushed the transaction toward being flagged.")
    drv = pd.DataFrame(pred.top_contributions, columns=["Pattern feature", "Influence on score"])
    st.dataframe(drv, hide_index=True, width='stretch')


def render_sources(retrieved):
    st.caption("The past cases and policies the assistant used as evidence. "
               "The explanation can only draw on these.")
    for r in retrieved:
        d = r.document
        badge = "📘 Policy" if d.kind == "policy" else "📕 Past case"
        with st.expander(f"{badge}  ·  {d.title}"):
            st.write(d.body)


def render_explanation(exp):
    st.markdown(f'<div class="explain-card">{exp.text}</div>',
                unsafe_allow_html=True)
    st.write("")

    # plain-language trust signal (the focal takeaway)
    if exp.all_citations_valid:
        n = len(exp.cited_ids)
        if n:
            st.markdown(f'<span class="verified">✓ Sources verified</span> '
                        f'&nbsp;—&nbsp; every claim is backed by one of the '
                        f'retrieved documents ({n} cited).',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="verified">✓ Grounded</span> '
                        '&nbsp;—&nbsp; the explanation stays within the retrieved '
                        'evidence.', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="unverified">⚠ Unverified claims detected</span> '
                    f'&nbsp;—&nbsp; the explanation referenced material that was '
                    f'not in the retrieved evidence.', unsafe_allow_html=True)

    # engineering detail tucked away for those who want it
    with st.expander("Technical details"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Citations", len(exp.cited_ids))
        c2.metric("Citation check", "PASS" if exp.all_citations_valid else "FAIL",
                  help="Every cited id must be one of the retrieved documents.")
        c3.metric("Generator", exp.backend)
        if exp.hallucinated_ids:
            st.warning("Citations not found in retrieved context: "
                       + ", ".join(exp.hallucinated_ids))
        else:
            st.caption("Cited document ids: "
                       + (", ".join(exp.cited_ids) or "(none cited)"))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    pipe = get_pipeline()
    info = pipe.info
    sample, data_source, n_rows, n_fraud = get_sample_rows()

    # styled header
    st.markdown("""
    <div class="app-header">
      <h1>🔎 Fraud Detection &amp; Investigation Assistant</h1>
      <p>Flags suspicious transactions and explains why — in plain language,
         backed by past cases and policy.</p>
    </div>""", unsafe_allow_html=True)

    # plain-language configuration pills (no raw backend strings)
    st.markdown(f"""
    <div class="pill-row">
      <span class="pill">Data: <b>{_data_label(data_source)}</b></span>
      <span class="pill">Retrieval: <b>{_retrieval_label(info['retriever_backend'])}</b></span>
      <span class="pill">AI model: <b>{_model_label(info['generator_backend'])}</b></span>
      <span class="pill">Knowledge base: <b>{info['kb_size']} documents</b></span>
    </div>""", unsafe_allow_html=True)

    if data_source == "synthetic":
        st.warning("Running on sample data — add the real transaction file for "
                   "real results (see the project README).")
    if info["generator_backend"] == "offline-stub":
        st.info("Explanations are written by a built-in explainer. Add a Gemini "
                "API key for AI-generated explanations — the app switches over "
                "automatically.")

    st.divider()

    # ---- transaction selection ------------------------------------------- #
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Choose a transaction")
        mode = st.radio("Source", ["Known fraud", "Known legitimate",
                                    "Enter manually"], label_visibility="collapsed")

        transaction = None
        ground_truth = None
        if mode in ("Known fraud", "Known legitimate"):
            cls = 1 if mode == "Known fraud" else 0
            pool = sample[sample.Class == cls]
            row_ids = pool["row_id"].tolist()
            if "sel_row" not in st.session_state or \
                    st.session_state.get("sel_mode") != mode:
                st.session_state.sel_row = int(row_ids[0])
                st.session_state.sel_mode = mode
            if st.button("🎲 Random row", width='stretch'):
                st.session_state.sel_row = int(np.random.choice(row_ids))
            chosen = st.selectbox("Row id", row_ids,
                                  index=row_ids.index(st.session_state.sel_row)
                                  if st.session_state.sel_row in row_ids else 0)
            st.session_state.sel_row = int(chosen)
            transaction = pool[pool.row_id == chosen].iloc[0][FEATURE_COLS]
            ground_truth = "FRAUD" if cls == 1 else "LEGITIMATE"
            st.caption(f"Known outcome for this transaction: **{ground_truth}**")
        else:
            amount = st.number_input("Amount ($)", min_value=0.0,
                                     value=0.79, step=1.0)
            hour = st.slider("Hour of day", 0, 23, 2)
            transaction = _legit_typical_row(amount, hour)
            st.caption("Manual mode tests the amount and time rules using an "
                       "ordinary transaction profile. It won't reproduce a real "
                       "fraud pattern — use a known-fraud example for that.")

        run = st.button("Analyze transaction", type="primary",
                        width='stretch')

    with right:
        st.subheader("Result")
        if not run:
            st.info("Pick a transaction on the left and click "
                    "**Analyze transaction** to see the assessment.")
            return

        out = pipe.run(transaction, explain_even_if_clean=True)
        render_verdict(out.prediction)

        if ground_truth:
            agree = ((out.prediction.is_flagged and ground_truth == "FRAUD") or
                     (not out.prediction.is_flagged and ground_truth == "LEGITIMATE"))
            if agree:
                st.success(f"✓ The assessment matches the known outcome "
                           f"({ground_truth.lower()}).")
            else:
                st.warning(f"The assessment differs from the known outcome "
                           f"(actually {ground_truth.lower()}).")
        elif not out.prediction.is_flagged:
            # manual-entry path: explain why amount alone didn't trigger a flag
            st.info("Not flagged — in manual mode the transaction pattern is set "
                    "to an ordinary profile, so only amount and time affect the "
                    "score. A real flag needs the suspicious transaction-pattern "
                    "signature found in genuine fraud. Try a **Known fraud** "
                    "example to see a flag and its explanation.")

        with st.expander("What influenced this score?", expanded=False):
            render_drivers(out.prediction)

        st.divider()
        if out.was_explained:
            st.subheader("Why this was flagged")
            tab_exp, tab_src = st.tabs(["Explanation", "Evidence used"])
            with tab_exp:
                render_explanation(out.explanation)
                with st.expander("Search query (technical)"):
                    st.caption("The plain-language query the assistant built from "
                               "the transaction to search its knowledge base:")
                    st.code(out.query, language=None)
            with tab_src:
                render_sources(out.retrieved)
        else:
            st.info("This transaction wasn't flagged, so no explanation was "
                    "needed. Try a flagged transaction to see the assistant's "
                    "reasoning and evidence.")


if __name__ == "__main__":
    main()
