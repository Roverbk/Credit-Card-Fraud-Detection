import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS — minimalist dark theme ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e8;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 4rem 2rem 4rem; max-width: 1200px; }

/* Hero header */
.hero { border-bottom: 1px solid #1f1f1f; padding-bottom: 2rem; margin-bottom: 2.5rem; }
.hero-title {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 300;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
}
.hero-sub {
    font-size: 0.85rem;
    color: #555;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
}

/* Metric cards */
.metric-row { display: flex; gap: 1px; margin: 2rem 0; background: #1a1a1a; }
.metric-card {
    flex: 1;
    background: #0a0a0a;
    padding: 1.5rem 1.75rem;
    border-top: 2px solid #1f1f1f;
}
.metric-card.alert { border-top-color: #ff3b3b; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 300;
    color: #ffffff;
    line-height: 1;
}
.metric-value.danger { color: #ff3b3b; }
.metric-value.success { color: #00c37a; }

/* Section headers */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #444;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a1a;
}

/* Upload zone */
.upload-zone {
    border: 1px dashed #2a2a2a;
    border-radius: 2px;
    padding: 3rem;
    text-align: center;
    margin: 1.5rem 0;
    background: #0d0d0d;
}
.upload-hint {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #444;
    margin-top: 0.5rem;
}

/* Result badge */
.result-fraud {
    background: #1a0000;
    border: 1px solid #ff3b3b;
    border-radius: 2px;
    padding: 1.25rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #ff3b3b;
    margin: 1.5rem 0;
}
.result-legit {
    background: #001a0e;
    border: 1px solid #00c37a;
    border-radius: 2px;
    padding: 1.25rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #00c37a;
    margin: 1.5rem 0;
}
.prob-value {
    font-size: 1.5rem;
    font-weight: 500;
    float: right;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #1a1a1a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #444;
    padding: 0.75rem 1.5rem;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
    background: transparent !important;
}

/* Inputs */
.stNumberInput input, .stTextInput input {
    background: #0d0d0d !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 2px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stNumberInput label, .stTextInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}

/* Button */
.stButton button {
    background: #ffffff !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    font-weight: 500 !important;
    transition: opacity 0.15s ease !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Dataframe */
.stDataFrame {
    border: 1px solid #1a1a1a !important;
    border-radius: 2px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d0d0d !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    color: #555 !important;
    border: 1px solid #1f1f1f !important;
    font-size: 0.68rem !important;
}
[data-testid="stDownloadButton"] button:hover {
    color: #e8e8e8 !important;
    border-color: #444 !important;
    opacity: 1 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('notebooks/model/fraud_model.pkl')
    scaler = joblib.load('notebooks/model/scaler.pkl')
    with open('notebooks/model/feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_columns = load_artifacts()
THRESHOLD = 0.948

# ── Preprocess helper ────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df['Amount_Scaled'] = scaler.transform(df[['Amount']].values)
    df['Time_Scaled']   = scaler.transform(df[['Time']].values)
    df.drop(columns=['Amount', 'Time'], inplace=True)
    return df[feature_columns]

# ── Matplotlib dark style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0a0a0a',
    'axes.facecolor':    '#0d0d0d',
    'axes.edgecolor':    '#1f1f1f',
    'axes.labelcolor':   '#555',
    'xtick.color':       '#555',
    'ytick.color':       '#555',
    'text.color':        '#e8e8e8',
    'grid.color':        '#1a1a1a',
    'font.family':       'monospace',
    'font.size':         9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">⬡ Fraud Detection</p>
    <p class="hero-sub">XGBoost · threshold 0.948 · AUC-ROC 0.976</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Batch Upload", "Single Transaction"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch Upload
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Upload CSV</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#444;">'
        'Required columns — Time · V1–V28 · Amount · Class (optional)</p>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("", type="csv", label_visibility="collapsed")

    if uploaded:
        raw_df      = pd.read_csv(uploaded)
        has_labels  = 'Class' in raw_df.columns
        true_labels = raw_df['Class'].values if has_labels else None

        try:
            input_df = preprocess(raw_df.drop(columns=['Class'], errors='ignore'))
        except KeyError as e:
            st.error(f"Missing column: {e}")
            st.stop()

        probs  = model.predict_proba(input_df)[:, 1]
        preds  = (probs >= THRESHOLD).astype(int)

        results_df = raw_df.copy()
        results_df['fraud_probability'] = probs.round(4)
        results_df['flagged']           = preds
        results_df['decision']          = np.where(preds == 1, 'FRAUD', 'LEGIT')

        # ── Metrics ──
        flagged_count = int(preds.sum())
        avg_prob      = float(probs[preds == 1].mean()) if flagged_count > 0 else 0

        if has_labels:
            from sklearn.metrics import recall_score, precision_score
            rec  = recall_score(true_labels, preds)
            prec = precision_score(true_labels, preds, zero_division=0)
            rp   = f"{rec:.1%} / {prec:.1%}"
        else:
            rp = "—"

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Transactions</div>
                <div class="metric-value">{len(results_df):,}</div>
            </div>
            <div class="metric-card alert">
                <div class="metric-label">Flagged</div>
                <div class="metric-value danger">{flagged_count:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Fraud Prob</div>
                <div class="metric-value">{avg_prob:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall / Precision</div>
                <div class="metric-value">{rp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Flagged table ──
        st.markdown('<p class="section-label">Flagged Transactions</p>',
                    unsafe_allow_html=True)
        flagged_df = (
            results_df[results_df['flagged'] == 1]
            [['fraud_probability', 'decision', 'Amount', 'V14', 'V4', 'V12']]
            .sort_values('fraud_probability', ascending=False)
        )
        st.dataframe(
            flagged_df.style.background_gradient(subset=['fraud_probability'],
                                                  cmap='Reds'),
            use_container_width=True, height=280
        )

        # ── Distribution chart ──
        st.markdown('<p class="section-label">Probability Distribution</p>',
                    unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        fig.patch.set_facecolor('#0a0a0a')

        # Left — log scale
        axes[0].hist(probs[preds == 0], bins=50, alpha=0.5,
                     color='#3a7bd5', label='Legit')
        axes[0].hist(probs[preds == 1], bins=20, alpha=0.9,
                     color='#ff3b3b', label='Fraud')
        axes[0].axvline(THRESHOLD, color='#ffffff', linestyle='--',
                        linewidth=0.8, label=f't = {THRESHOLD}')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Fraud Probability')
        axes[0].set_ylabel('Count (log)')
        axes[0].set_title('All Transactions', color='#888', fontsize=9)
        axes[0].legend(fontsize=8)

        # Right — zoom
        high = probs[probs > 0.5]
        if len(high) > 0:
            axes[1].hist(high, bins=20, color='#ff3b3b', alpha=0.8)
            axes[1].axvline(THRESHOLD, color='#ffffff', linestyle='--',
                            linewidth=0.8, label=f't = {THRESHOLD}')
            axes[1].set_xlabel('Fraud Probability')
            axes[1].set_ylabel('Count')
            axes[1].set_title('High-Risk Zone (prob > 0.5)',
                              color='#888', fontsize=9)
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, 'No high-risk transactions',
                         ha='center', va='center', color='#444',
                         transform=axes[1].transAxes)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)

        # ── Download ──
        st.markdown('<p class="section-label">Export</p>',
                    unsafe_allow_html=True)
        # Only flagged fraud transactions
        fraud_only_df = results_df[results_df['flagged'] == 1].copy()
        csv = fraud_only_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download flagged transactions ({len(fraud_only_df):,})",
            data=csv,
            file_name="fraud_flagged.csv",
            mime="text/csv"
        )

    else:
        # Empty state
        st.markdown("""
        <div style="padding: 4rem 0; text-align: center; border: 1px dashed #1a1a1a;
                    margin-top: 1.5rem;">
            <p style="font-family:'DM Mono',monospace; font-size:0.75rem;
                      color:#333; margin:0;">
                drop a csv file above to begin
            </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Single Transaction
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Transaction Details</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        amount = st.number_input("Amount ($)", min_value=0.0,
                                  max_value=30000.0, value=100.0, step=10.0)
    with c2:
        time = st.number_input("Time (seconds)", min_value=0,
                                max_value=200000, value=50000)

    st.markdown('<p class="section-label">PCA Features — V1 to V28</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
        'color:#333;margin-bottom:1rem;">Leave at 0 for an average transaction</p>',
        unsafe_allow_html=True
    )

    cols  = st.columns(7)
    v_vals = {}
    for i in range(28):
        feat = f'V{i+1}'
        v_vals[feat] = cols[i % 7].number_input(
            feat, value=0.0, format="%.3f", key=feat
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", use_container_width=False)

    if predict_btn:
        row      = {**v_vals, 'Amount': amount, 'Time': time}
        row_df   = pd.DataFrame([row])
        processed = preprocess(row_df)

        prob = float(model.predict_proba(processed)[0][1])
        pred = int(prob >= THRESHOLD)

        if pred == 1:
            st.markdown(f"""
            <div class="result-fraud">
                FRAUD DETECTED
                <span class="prob-value">{prob:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                LEGITIMATE
                <span class="prob-value">{prob:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        # SHAP waterfall
        st.markdown('<p class="section-label">Explanation</p>',
                    unsafe_allow_html=True)

        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(processed)

        fig3 = plt.figure(figsize=(9, 4))
        fig3.patch.set_facecolor('#0a0a0a')
        shap.waterfall_plot(
            shap.Explanation(
                values        = shap_vals[0],
                base_values   = explainer.expected_value,
                data          = processed.values[0],
                feature_names = feature_columns
            ),
            show=False
        )
        ax = plt.gca()
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='#555', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
