import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import shap

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ── Load model artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load('notebooks/model/fraud_model.pkl')
    scaler  = joblib.load('notebooks/model/scaler.pkl')
    with open('notebooks/model/feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_columns = load_artifacts()
THRESHOLD = 0.948

# ── Helper: preprocess uploaded CSV ────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df['Amount_Scaled'] = scaler.transform(df[['Amount']].values)
    df['Time_Scaled']   = scaler.transform(df[['Time']])
    df.drop(columns=['Amount', 'Time'], inplace=True)

    # Ensure column order matches training
    df = df[feature_columns]
    return df

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Upload a transaction CSV to score it, or enter a single transaction manually.")

tab1, tab2 = st.tabs(["📁 Batch Upload", "🔢 Single Transaction"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch Upload
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload Transaction CSV")
    st.markdown("File must have columns: `Time`, `V1–V28`, `Amount`  \n`Class` column is optional.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.write(f"**{len(raw_df):,} transactions loaded**")

        # Preprocess
        has_labels = 'Class' in raw_df.columns
        true_labels = raw_df['Class'].values if has_labels else None

        try:
            input_df = preprocess(raw_df.drop(columns=['Class'], errors='ignore'))
        except KeyError as e:
            st.error(f"Missing column in uploaded file: {e}")
            st.stop()

        # Score
        probs  = model.predict_proba(input_df)[:, 1]
        preds  = (probs >= THRESHOLD).astype(int)

        results_df = raw_df.copy()
        results_df['fraud_probability'] = probs.round(4)
        results_df['flagged']           = preds
        results_df['decision']          = np.where(preds == 1, '🚨 FRAUD', '✅ LEGIT')

        # ── Summary metrics ──
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(results_df):,}")
        col2.metric("Flagged as Fraud",   f"{preds.sum():,}")
        col3.metric("Avg Fraud Prob (flagged)",
                    f"{probs[preds==1].mean():.3f}" if preds.sum() > 0 else "N/A")

        if has_labels:
            from sklearn.metrics import recall_score, precision_score
            rec  = recall_score(true_labels, preds)
            prec = precision_score(true_labels, preds, zero_division=0)
            col4.metric("Recall / Precision", f"{rec:.1%} / {prec:.1%}")

        # ── Flagged transactions table ──
        st.markdown("### 🚨 Flagged Transactions")
        flagged_df = results_df[results_df['flagged'] == 1][
            ['fraud_probability', 'decision', 'Amount', 'V14', 'V4', 'V12']
        ].sort_values('fraud_probability', ascending=False)

        st.dataframe(flagged_df.style.background_gradient(
            subset=['fraud_probability'], cmap='Reds'), use_container_width=True)

        # ── Fraud probability distribution ──
        st.markdown("### 📊 Fraud Probability Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        # Left plot — log scale so both classes are visible
        axes[0].hist(probs[preds == 0], bins=50, alpha=0.6,
                    color='steelblue', label='Predicted Legit')
        axes[0].hist(probs[preds == 1], bins=20, alpha=0.8,
                    color='crimson', label='Predicted Fraud')
        axes[0].axvline(THRESHOLD, color='black', linestyle='--',
                        label=f'Threshold = {THRESHOLD}')
        axes[0].set_yscale('log')          # ← this fixes the visibility
        axes[0].set_xlabel('Fraud Probability')
        axes[0].set_ylabel('Count (log scale)')
        axes[0].set_title('Full Distribution (Log Scale)')
        axes[0].legend()

        # Right plot — zoom into fraud region only (prob > 0.5)
        high_prob = probs[probs > 0.5]
        colors    = ['crimson' if p >= THRESHOLD else 'orange' for p in high_prob]
        axes[1].hist(high_prob, bins=20, color='crimson', alpha=0.8)
        axes[1].axvline(THRESHOLD, color='black', linestyle='--',
                        label=f'Threshold = {THRESHOLD}')
        axes[1].set_xlabel('Fraud Probability')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Zoomed: High-Risk Transactions (prob > 0.5)')
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # ── Download results ──
        st.markdown("### ⬇️ Download Scored Results")
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV with fraud scores",
            data=csv,
            file_name="fraud_scores.csv",
            mime="text/csv"
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Single Transaction
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Score a Single Transaction")
    st.markdown("Adjust the values below and click **Predict**.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0,
                                  max_value=30000.0, value=100.0, step=10.0)
        time   = st.number_input("Time (seconds from first txn)",
                                  min_value=0, max_value=200000, value=50000)

    st.markdown("**PCA Features (V1–V28)** — leave at 0 for an average transaction")
    v_cols = st.columns(7)
    v_vals = {}
    for i, col in enumerate(v_cols * 4):
        feat = f'V{i+1}'
        if i >= 28:
            break
        v_vals[feat] = col.number_input(feat, value=0.0,
                                         format="%.3f", key=feat)

    if st.button("🔍 Predict", use_container_width=True):
        # Build input row
        row = {**v_vals, 'Amount': amount, 'Time': time}
        row_df = pd.DataFrame([row])
        processed = preprocess(row_df)

        prob = model.predict_proba(processed)[0][1]
        pred = int(prob >= THRESHOLD)

        st.markdown("---")
        if pred == 1:
            st.error(f"🚨 **FRAUD DETECTED**  —  Probability: `{prob:.4f}`")
        else:
            st.success(f"✅ **LEGITIMATE**  —  Probability: `{prob:.4f}`")

        # SHAP explanation for this transaction
        st.markdown("### Why this decision?")
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(processed)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values       = shap_vals[0],
                base_values  = explainer.expected_value,
                data         = processed.values[0],
                feature_names= feature_columns
            ),
            show=False
        )
        st.pyplot(plt.gcf())