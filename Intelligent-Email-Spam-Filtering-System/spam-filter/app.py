import streamlit as st
import joblib
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Shield — Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #2a3550;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; color: #e2e8f0; margin: 0; }
    .hero p  { color: #94a3b8; font-size: 1.05rem; margin-top: 0.4rem; }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        flex: 1;
        background: #1e2435;
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-card .val { font-size: 1.9rem; font-weight: 700; }
    .metric-card .lbl { font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .green { color: #4ade80; }
    .blue  { color: #60a5fa; }
    .purple{ color: #c084fc; }
    .amber { color: #fbbf24; }

    /* Result box */
    .result-spam {
        background: linear-gradient(135deg, #3f1515, #2d1010);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        animation: pulse-red 2s infinite;
    }
    .result-ham {
        background: linear-gradient(135deg, #0f3020, #0a2218);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
        50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
    }
    .result-label { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-sub   { color: #94a3b8; font-size: 0.9rem; }

    /* Confidence bar */
    .conf-bar-bg {
        background: #2a3550;
        border-radius: 99px;
        height: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s ease;
    }

    /* History table */
    .hist-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        background: #1e2435;
        border: 1px solid #2a3550;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .badge-spam { background:#ef4444; color:#fff; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
    .badge-ham  { background:#22c55e; color:#fff; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
    .hist-msg   { color:#cbd5e1; font-size:0.88rem; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .hist-time  { color:#64748b; font-size:0.78rem; }

    /* Sidebar */
    .sidebar-card {
        background: #1e2435;
        border: 1px solid #2a3550;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-card h4 { color: #e2e8f0; margin: 0 0 0.6rem; font-size: 0.9rem; }

    /* Model comparison table */
    .model-row { display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #2a3550; font-size:0.85rem; }
    .model-row:last-child { border:none; }
    .model-name { color:#94a3b8; }
    .model-val  { color:#60a5fa; font-weight:600; }

    /* Spinner override */
    .stSpinner > div { border-top-color: #6366f1 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Load model & metrics ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, 'models', 'spam_model.pkl'))

@st.cache_data
def load_metrics():
    with open(os.path.join(BASE, 'models', 'metrics.json')) as f:
        return json.load(f)

model   = load_model()
metrics = load_metrics()

# ── Session state ─────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_checked' not in st.session_state:
    st.session_state.total_checked = 0
if 'spam_caught' not in st.session_state:
    st.session_state.spam_caught = 0

# ── Text cleaner (must match training) ───────────────────────────────
def clean(text):
    text = text.lower()
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text):
    cleaned = clean(text)
    label   = model.predict([cleaned])[0]
    # Confidence via decision function (SVM) or predict_proba
    clf = model.named_steps['clf']
    tfidf = model.named_steps['tfidf']
    vec = tfidf.transform([cleaned])
    if hasattr(clf, 'decision_function'):
        score = clf.decision_function(vec)[0]
        # sigmoid squash to 0-1
        confidence = 1 / (1 + np.exp(-abs(score)))
    else:
        proba = clf.predict_proba(vec)[0]
        confidence = max(proba)
    return int(label), float(confidence)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Spam Shield")
    st.markdown("---")

    # Session stats
    spam_rate = (st.session_state.spam_caught / st.session_state.total_checked * 100
                 if st.session_state.total_checked > 0 else 0)
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📊 Session Stats</h4>
        <div class="model-row"><span class="model-name">Emails Checked</span><span class="model-val">{st.session_state.total_checked}</span></div>
        <div class="model-row"><span class="model-name">Spam Caught</span><span class="model-val" style="color:#ef4444">{st.session_state.spam_caught}</span></div>
        <div class="model-row"><span class="model-name">Spam Rate</span><span class="model-val" style="color:#fbbf24">{spam_rate:.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Model info
    best = metrics['best_model']
    r    = metrics['results'][best]
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>🤖 Active Model</h4>
        <div class="model-row"><span class="model-name">Algorithm</span><span class="model-val">Linear SVM</span></div>
        <div class="model-row"><span class="model-name">Accuracy</span><span class="model-val">{r['accuracy']*100:.2f}%</span></div>
        <div class="model-row"><span class="model-name">Precision</span><span class="model-val">{r['precision']*100:.2f}%</span></div>
        <div class="model-row"><span class="model-name">Recall</span><span class="model-val">{r['recall']*100:.2f}%</span></div>
        <div class="model-row"><span class="model-name">F1 Score</span><span class="model-val">{r['f1']*100:.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Training data
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📁 Training Dataset</h4>
        <div class="model-row"><span class="model-name">Total Samples</span><span class="model-val">{metrics['total_samples']:,}</span></div>
        <div class="model-row"><span class="model-name">Ham (Legit)</span><span class="model-val" style="color:#22c55e">{metrics['ham_count']:,}</span></div>
        <div class="model-row"><span class="model-name">Spam</span><span class="model-val" style="color:#ef4444">{metrics['spam_count']:,}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.total_checked = 0
        st.session_state.spam_caught = 0
        st.rerun()

# ── Main content ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ Spam Shield</h1>
    <p>Intelligent Email Spam Detection · Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Top metrics
r = metrics['results'][metrics['best_model']]
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><div class="val green">{r['accuracy']*100:.1f}%</div><div class="lbl">Accuracy</div></div>
    <div class="metric-card"><div class="val blue">{r['precision']*100:.1f}%</div><div class="lbl">Precision</div></div>
    <div class="metric-card"><div class="val purple">{r['recall']*100:.1f}%</div><div class="lbl">Recall</div></div>
    <div class="metric-card"><div class="val amber">{r['f1']*100:.1f}%</div><div class="lbl">F1 Score</div></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Classifier", "📊 Analytics", "ℹ️ About"])

# ── Tab 1: Classifier ─────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("#### Paste your email or message below")
        user_input = st.text_area(
            label="",
            height=220,
            placeholder="Example: Congratulations! You've won a FREE iPhone. Click here to claim now...",
            label_visibility="collapsed"
        )

        # Quick test examples
        st.markdown("**⚡ Quick test examples:**")
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            if st.button("🚨 Spam Example 1"):
                st.session_state['example'] = "WINNER!! You have been selected as a lucky winner of $1,000,000. Call 09061743810 to claim now!"
        with ecol2:
            if st.button("🚨 Spam Example 2"):
                st.session_state['example'] = "FREE entry in 2 a wkly comp to win FA Cup Final tkts 21st May 2005. Text FA to 87121"
        with ecol3:
            if st.button("✅ Ham Example"):
                st.session_state['example'] = "Hey, are we still on for lunch tomorrow? Let me know what time works for you."

        if 'example' in st.session_state:
            user_input = st.session_state.pop('example')
            st.rerun()

        analyze = st.button("🔍 Analyze Message")

    with col2:
        if analyze and user_input.strip():
            with st.spinner("Analyzing..."):
                label, confidence = predict(user_input)

                # Update session
                st.session_state.total_checked += 1
                if label == 1:
                    st.session_state.spam_caught += 1

                # Save to history
                st.session_state.history.insert(0, {
                    'label': label,
                    'confidence': confidence,
                    'message': user_input[:120],
                    'time': datetime.now().strftime("%H:%M:%S")
                })

            is_spam = label == 1
            emoji   = "🚨" if is_spam else "✅"
            verdict = "SPAM DETECTED" if is_spam else "LEGITIMATE EMAIL"
            color   = "#ef4444" if is_spam else "#22c55e"
            cls     = "result-spam" if is_spam else "result-ham"
            desc    = "This message has been flagged as spam." if is_spam else "This message appears to be legitimate."
            bar_color = "#ef4444" if is_spam else "#22c55e"

            st.markdown(f"""
            <div class="{cls}">
                <div class="result-label" style="color:{color}">{emoji} {verdict}</div>
                <div class="result-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Confidence: {confidence*100:.1f}%**")
            st.markdown(f"""
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence*100:.1f}%;background:{bar_color};"></div>
            </div>
            """, unsafe_allow_html=True)

            # Word count / length stats
            words = len(user_input.split())
            chars = len(user_input)
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-top:1rem;">
                <div style="flex:1;background:#1e2435;border-radius:8px;padding:0.6rem;text-align:center;">
                    <div style="font-size:1.3rem;font-weight:700;color:#60a5fa">{words}</div>
                    <div style="font-size:0.75rem;color:#94a3b8">WORDS</div>
                </div>
                <div style="flex:1;background:#1e2435;border-radius:8px;padding:0.6rem;text-align:center;">
                    <div style="font-size:1.3rem;font-weight:700;color:#c084fc">{chars}</div>
                    <div style="font-size:0.75rem;color:#94a3b8">CHARS</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif analyze and not user_input.strip():
            st.warning("Please enter a message to analyze.")
        else:
            st.markdown("""
            <div style="background:#1e2435;border:1px dashed #2a3550;border-radius:12px;padding:3rem 1rem;text-align:center;color:#64748b;">
                <div style="font-size:2.5rem">🛡️</div>
                <div style="margin-top:0.5rem">Result will appear here</div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Analytics ──────────────────────────────────────────────────
with tab2:
    st.markdown("#### Model Comparison")

    results_data = []
    for name, res in metrics['results'].items():
        results_data.append({
            'Model': name,
            'Accuracy': f"{res['accuracy']*100:.2f}%",
            'Precision': f"{res['precision']*100:.2f}%",
            'Recall': f"{res['recall']*100:.2f}%",
            'F1 Score': f"{res['f1']*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Dataset Distribution")
        dist_df = pd.DataFrame({
            'Category': ['Ham (Legitimate)', 'Spam'],
            'Count': [metrics['ham_count'], metrics['spam_count']],
            'Percentage': [
                f"{metrics['ham_count']/metrics['total_samples']*100:.1f}%",
                f"{metrics['spam_count']/metrics['total_samples']*100:.1f}%"
            ]
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Confusion Matrix (Best Model)")
        best_cm = metrics['results'][metrics['best_model']]['confusion_matrix']
        cm_df = pd.DataFrame(
            best_cm,
            index=['Actual Ham', 'Actual Spam'],
            columns=['Predicted Ham', 'Predicted Spam']
        )
        st.dataframe(cm_df, use_container_width=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 📋 Session History")
        for item in st.session_state.history[:20]:
            badge = '<span class="badge-spam">SPAM</span>' if item['label'] else '<span class="badge-ham">HAM</span>'
            st.markdown(f"""
            <div class="hist-row">
                {badge}
                <span class="hist-msg">{item['message']}</span>
                <span class="hist-time">{item['confidence']*100:.0f}% · {item['time']}</span>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 3: About ──────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 How It Works

        This system uses **TF-IDF vectorization** combined with a **Linear SVM classifier** trained on 5,572 real SMS/email messages.

        **Pipeline:**
        1. Text is lowercased and cleaned
        2. Numbers replaced with `NUM` token
        3. URLs replaced with `URL` token  
        4. Converted to TF-IDF feature vectors (bigrams, 10k features)
        5. Linear SVM classifies as spam or ham

        **Why Linear SVM?**  
        Outperformed Naive Bayes and Logistic Regression on this dataset with 99.1% accuracy and 96.6% F1 score.
        """)
    with c2:
        st.markdown("""
        #### 📚 Dataset

        **UCI SMS Spam Collection** — a public benchmark dataset containing 5,572 real SMS messages labeled as spam or ham.

        - 4,825 legitimate messages (86.6%)
        - 747 spam messages (13.4%)

        #### 🔧 Tech Stack

        - **Python** — core language
        - **Scikit-learn** — ML pipeline
        - **Streamlit** — web interface
        - **Pandas / NumPy** — data processing
        - **TF-IDF** — feature extraction

        #### 👤 Built By
        **Pradeep Kumar R** — AIML Undergrad @ SKIT Bengaluru  
        [github.com/Pradeep-0987](https://github.com/Pradeep-0987)  
        [kumarrpradeep970@gmail.com](mailto:kumarrpradeep970@gmail.com)
        """)
