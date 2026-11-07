import json
from pathlib import Path

import joblib
import streamlit as st

from preprocess import clean_text

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(BASE / "models" / "spam_model.joblib")


@st.cache_data
def load_metrics():
    with open(BASE / "models" / "metrics.json") as f:
        return json.load(f)


SPAM_EXAMPLE = ("Subject: You WON a free prize!!!\n\nCongratulations! Click http://claim-now.biz "
                "to get your $1000 gift card. Limited time offer, act now!")
HAM_EXAMPLE = ("Subject: Meeting notes\n\nHi team, attaching the notes from yesterday's meeting. "
               "Let's review the project schedule on Monday at 10am.")

st.title("📧 Spam or Not Spam")
st.caption("Paste an email below and the machine learning model will classify it.")

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

c1, c2, _ = st.columns([1, 1, 2])
if c1.button("Try spam example"):
    st.session_state.email_text = SPAM_EXAMPLE
if c2.button("Try normal example"):
    st.session_state.email_text = HAM_EXAMPLE

text = st.text_area("Email text (subject + body)", key="email_text", height=220,
                    placeholder="Paste the email here...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some email text first.")
    else:
        model = load_model()
        prob_spam = float(model.predict_proba([clean_text(text)])[0][1])
        if prob_spam >= 0.5:
            st.error(f"🚨 **SPAM** — confidence {prob_spam:.1%}")
        else:
            st.success(f"✅ **NOT SPAM** — confidence {1 - prob_spam:.1%}")
        st.progress(prob_spam, text=f"Spam probability: {prob_spam:.1%}")

with st.expander("Model performance"):
    m = load_metrics()
    best = m["best_model"]
    st.write(f"**Model:** {best} with TF-IDF (tested on {m['test_size']} unseen emails)")
    b = m[best]
    a, p, r, f = st.columns(4)
    a.metric("Accuracy", f"{b['accuracy']:.2%}")
    p.metric("Precision", f"{b['precision']:.2%}")
    r.metric("Recall", f"{b['recall']:.2%}")
    f.metric("F1-score", f"{b['f1']:.2%}")
