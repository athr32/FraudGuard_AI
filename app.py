"""
app.py – Financial Fraud Detection Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Main background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid #1e2d4a;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1224 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,100,255,0.08);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-title { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
.metric-value { font-size: 36px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; margin: 8px 0 4px; }
.metric-delta { font-size: 12px; }

.red   { color: #ff4d6d; }
.green { color: #22d3ee; }
.amber { color: #f59e0b; }
.blue  { color: #60a5fa; }

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #64748b;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* Alert boxes */
.alert-critical {
    background: rgba(255,77,109,0.1);
    border-left: 4px solid #ff4d6d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}
.alert-high {
    background: rgba(245,158,11,0.1);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 12px;
}
.alert-medium {
    background: rgba(96,165,250,0.07);
    border-left: 4px solid #60a5fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 12px;
}

/* Streamlit overrides */
.stSelectbox label, .stSlider label { color: #94a3b8 !important; font-size: 12px !important; }
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
.stButton > button {
    background: linear-gradient(90deg, #1e40af, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2d4a !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
}

/* DataFrame */
.stDataFrame { border: 1px solid #1e2d4a; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Load / Train Model ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from models import FraudDetectionPipeline
    from data_generator import generate_transactions

    model_path = "models/fraud_model.joblib"
    if os.path.exists(model_path):
        return FraudDetectionPipeline.load(model_path)

    os.makedirs("models", exist_ok=True)
    df = generate_transactions(8000)
    pipe = FraudDetectionPipeline()
    pipe.train(df)
    pipe.save(model_path)
    return pipe


@st.cache_data(show_spinner=False)
def load_data():
    from data_generator import generate_transactions
    return generate_transactions(10000)


# ─── Helpers ────────────────────────────────────────────────────────────────────
PLOT_BG   = "rgba(0,0,0,0)"
GRID_CLR  = "#1e2d4a"
TEXT_CLR  = "#94a3b8"
ACCENT    = "#60a5fa"
RED       = "#ff4d6d"
AMBER     = "#f59e0b"
GREEN     = "#22d3ee"

def plotly_theme(fig, height=350):
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_CLR, family="IBM Plex Sans"),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_CLR),
    )
    fig.update_xaxes(gridcolor=GRID_CLR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False)
    return fig


def risk_badge(level: str) -> str:
    colors = {"Critical": RED, "High": AMBER, "Medium": ACCENT, "Low": "#22d3ee"}
    c = colors.get(str(level), "#64748b")
    return f'<span style="background:{c}22;color:{c};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;border:1px solid {c}55">{level}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛡️ **FraudGuard AI**")
    st.markdown("<div style='color:#64748b;font-size:11px;margin-bottom:20px'>Financial Fraud Detection System v2.0</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Overview Dashboard", "🔍 Transaction Analysis", "🤖 Model Performance",
         "⚡ Live Detection", "🔬 Single Transaction"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div class='section-header'>Settings</div>", unsafe_allow_html=True)
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability threshold above which a transaction is flagged as fraud")
    n_live_transactions = st.slider("Live Transactions / Batch", 5, 50, 15)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#334155'>
    <b>Models Deployed:</b><br>
    • Isolation Forest<br>
    • One-Class SVM<br>
    • Autoencoder (MLP)<br>
    • Random Forest ✓<br>
    • Gradient Boosting ✓<br>
    • <b>Ensemble Voting</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Initializing FraudGuard AI... (first run trains models — ~60s)"):
    pipeline = load_pipeline()
    raw_df   = load_data()

with st.spinner("Running inference on dataset..."):
    results_df = pipeline.predict(raw_df, threshold=threshold)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown("## 📊 Overview Dashboard")
    st.markdown("<div style='color:#64748b;font-size:13px;margin-bottom:24px'>Real-time monitoring of transaction anomalies and fraud patterns</div>", unsafe_allow_html=True)

    total   = len(results_df)
    flagged = results_df["is_fraud_predicted"].sum()
    actual  = results_df["is_fraud"].sum() if "is_fraud" in results_df.columns else 0
    total_amount  = results_df["amount"].sum()
    fraud_amount  = results_df[results_df["is_fraud_predicted"] == 1]["amount"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, title, val, color, delta in [
        (c1, "Total Transactions", f"{total:,}", "blue", "Last 365 days"),
        (c2, "Flagged Fraud", f"{flagged:,}", "red", f"{flagged/total*100:.2f}% of total"),
        (c3, "Actual Fraud", f"{actual:,}", "amber", "Ground truth labels"),
        (c4, "Total Amount (₹)", f"₹{total_amount/1e6:.1f}M", "green", "Processed volume"),
        (c5, "Fraud Amount (₹)", f"₹{fraud_amount/1e6:.2f}M", "red", f"{fraud_amount/total_amount*100:.1f}% of volume"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value {color}">{val}</div>
            <div class="metric-delta" style="color:#64748b">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Time series + Distribution ────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-header'>Fraud Flagging Over Time</div>", unsafe_allow_html=True)
        ts = results_df.copy()
        ts["date"] = pd.to_datetime(ts["timestamp"]).dt.to_period("W").dt.start_time
        agg = ts.groupby("date").agg(
            total=("transaction_id", "count"),
            fraud=("is_fraud_predicted", "sum")
        ).reset_index()
        agg["fraud_rate"] = agg["fraud"] / agg["total"] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["total"], name="Total Tx",
                                  line=dict(color=ACCENT, width=1.5), fill="tozeroy",
                                  fillcolor="rgba(96,165,250,0.06)"))
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["fraud"], name="Flagged Fraud",
                                  line=dict(color=RED, width=2), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, color=TEXT_CLR))
        plotly_theme(fig, 300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Risk Level Distribution</div>", unsafe_allow_html=True)
        risk_counts = results_df["risk_level"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=risk_counts.index.astype(str),
            values=risk_counts.values,
            hole=0.55,
            marker=dict(colors=["#22d3ee", ACCENT, AMBER, RED]),
        ))
        fig2.update_traces(textfont=dict(color="white"), textinfo="percent+label")
        plotly_theme(fig2, 300)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Category + Hour heatmap ────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-header'>Fraud by Merchant Category</div>", unsafe_allow_html=True)
        cat = results_df.groupby("merchant_category")["is_fraud_predicted"].agg(["sum", "count"])
        cat["rate"] = cat["sum"] / cat["count"] * 100
        cat = cat.sort_values("rate", ascending=True).reset_index()
        fig3 = go.Figure(go.Bar(
            x=cat["rate"], y=cat["merchant_category"],
            orientation="h",
            marker=dict(color=cat["rate"], colorscale=[[0, "#1e3a5f"], [1, RED]]),
        ))
        fig3.update_layout(xaxis_title="Fraud Rate (%)")
        plotly_theme(fig3, 300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Transaction Amount vs Risk</div>", unsafe_allow_html=True)
        sample = results_df.sample(min(2000, len(results_df)), random_state=42)
        fig4 = px.scatter(
            sample, x="amount", y="ensemble_score",
            color="risk_level",
            color_discrete_map={"Low": GREEN, "Medium": ACCENT, "High": AMBER, "Critical": RED},
            opacity=0.6, log_x=True,
            labels={"ensemble_score": "Fraud Score", "amount": "Amount (₹, log scale)"}
        )
        plotly_theme(fig4, 300)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Analysis":
    st.markdown("## 🔍 Transaction Analysis")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Risk Level", ["Low", "Medium", "High", "Critical"],
                                      default=["High", "Critical"])
    with col2:
        cat_filter = st.multiselect("Merchant Category",
                                     results_df["merchant_category"].unique().tolist(),
                                     default=[])
    with col3:
        min_score = st.slider("Min Fraud Score", 0.0, 1.0, 0.5)

    filtered = results_df[results_df["ensemble_score"] >= min_score]
    if risk_filter:
        filtered = filtered[filtered["risk_level"].astype(str).isin(risk_filter)]
    if cat_filter:
        filtered = filtered[filtered["merchant_category"].isin(cat_filter)]

    st.markdown(f"<div style='color:#64748b;font-size:12px;margin-bottom:12px'>Showing {len(filtered):,} transactions</div>", unsafe_allow_html=True)

    # Display table
    display_cols = ["transaction_id", "timestamp", "amount", "merchant_category",
                    "transaction_country", "is_international", "distance_from_home_km",
                    "num_transactions_last_24h", "ensemble_score", "risk_level", "is_fraud_predicted"]
    show_df = filtered[display_cols].sort_values("ensemble_score", ascending=False).head(500)
    st.dataframe(
        show_df.style.background_gradient(subset=["ensemble_score"], cmap="RdYlGn_r"),
        use_container_width=True, height=400
    )

    # Feature distributions
    st.markdown("<div class='section-header'>Feature Distributions (Fraud vs Normal)</div>", unsafe_allow_html=True)
    feat_col = st.selectbox("Select Feature", ["amount", "distance_from_home_km",
                                                 "num_transactions_last_24h", "account_age_days",
                                                 "utilization_ratio", "hour_of_day"])
    fig5 = go.Figure()
    for label, color, name in [(0, ACCENT, "Normal"), (1, RED, "Fraud")]:
        subset = results_df[results_df["is_fraud"] == label][feat_col]
        fig5.add_trace(go.Histogram(x=subset, name=name, opacity=0.7,
                                     marker_color=color, nbinsx=50))
    fig5.update_layout(barmode="overlay", xaxis_title=feat_col)
    plotly_theme(fig5, 320)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance")

    with st.spinner("Evaluating models..."):
        metrics = pipeline.evaluate(results_df, threshold=threshold)

    # Summary table
    rows = []
    for model, m in metrics.items():
        if isinstance(m, dict) and "auc_roc" in m:
            rows.append({
                "Model": model.replace("_", " ").title(),
                "AUC-ROC": m["auc_roc"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1"],
                "Avg Precision": m["ap"],
            })
    metrics_df = pd.DataFrame(rows).set_index("Model")

    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style
        .background_gradient(subset=["AUC-ROC", "F1 Score"], cmap="Blues")
        .format("{:.4f}"),
        use_container_width=True
    )

    # Bar chart comparison
    col1, col2 = st.columns(2)
    with col1:
        fig6 = go.Figure()
        for metric in ["AUC-ROC", "F1 Score", "Precision", "Recall"]:
            fig6.add_trace(go.Bar(name=metric, x=metrics_df.index, y=metrics_df[metric]))
        fig6.update_layout(barmode="group", title="Metric Comparison Across Models")
        plotly_theme(fig6, 370)
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        # Confusion matrix
        cm = np.array(metrics["confusion_matrix"])
        fig7 = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Normal", "Fraud"],
            y=["Normal", "Fraud"],
            color_continuous_scale=[[0, "#0d1224"], [1, ACCENT]],
            text_auto=True,
            title="Confusion Matrix (Ensemble)"
        )
        fig7.update_traces(textfont=dict(size=18))
        plotly_theme(fig7, 370)
        st.plotly_chart(fig7, use_container_width=True)

    # Feature importance
    if pipeline.feature_importances_ is not None:
        st.markdown("<div class='section-header'>Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
        from models import FEATURE_COLS, CAT_COLS
        feat_names = FEATURE_COLS + CAT_COLS
        fi = pd.DataFrame({
            "Feature": feat_names[:len(pipeline.feature_importances_)],
            "Importance": pipeline.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig8 = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"],
            orientation="h",
            marker=dict(color=fi["Importance"], colorscale=[[0, "#1e3a5f"], [1, ACCENT]])
        ))
        plotly_theme(fig8, 420)
        st.plotly_chart(fig8, use_container_width=True)

    # Score distribution
    st.markdown("<div class='section-header'>Ensemble Score Distribution</div>", unsafe_allow_html=True)
    fig9 = go.Figure()
    for label, color, name in [(0, ACCENT, "Normal"), (1, RED, "Fraud")]:
        subset = results_df[results_df["is_fraud"] == label]["ensemble_score"]
        fig9.add_trace(go.Histogram(x=subset, name=name, opacity=0.75,
                                     marker_color=color, nbinsx=60))
    fig9.add_vline(x=threshold, line_dash="dash", line_color=AMBER,
                   annotation_text=f"Threshold={threshold}", annotation_font_color=AMBER)
    fig9.update_layout(barmode="overlay", xaxis_title="Ensemble Fraud Score")
    plotly_theme(fig9, 300)
    st.plotly_chart(fig9, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Live Detection":
    st.markdown("## ⚡ Live Transaction Monitoring")
    st.markdown("<div style='color:#64748b;font-size:13px;margin-bottom:20px'>Simulating real-time Kafka stream ingestion and instant fraud scoring</div>", unsafe_allow_html=True)

    # Session state for live transactions
    if "live_transactions" not in st.session_state:
        st.session_state.live_transactions = []
    if "live_alerts" not in st.session_state:
        st.session_state.live_alerts = []

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    run_batch = col_btn1.button(f"▶ Process {n_live_transactions} Transactions")
    clear_btn = col_btn2.button("🗑 Clear")

    if clear_btn:
        st.session_state.live_transactions = []
        st.session_state.live_alerts = []

    if run_batch:
        from data_generator import generate_streaming_transaction
        progress = st.progress(0, text="Processing transactions...")
        for i in range(n_live_transactions):
            tx = generate_streaming_transaction(fraud_prob=0.12)
            # derive features
            cl = tx["credit_limit"]
            ab = tx["available_balance"]
            tx["utilization_ratio"] = (cl - ab) / cl if cl else 0
            tx["amount_to_limit_ratio"] = tx["amount"] / cl if cl else 0

            pred = pipeline.predict_single(tx, threshold=threshold)
            tx.update(pred)
            st.session_state.live_transactions.append(tx)
            if pred["risk_level"] in ("Critical", "High"):
                st.session_state.live_alerts.append(tx)
            time.sleep(0.05)
            progress.progress((i + 1) / n_live_transactions, text=f"Processing tx {i+1}/{n_live_transactions}")
        progress.empty()

    # Live metrics
    if st.session_state.live_transactions:
        live_df = pd.DataFrame(st.session_state.live_transactions)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Processed", len(live_df))
        col2.metric("Fraud Flagged", live_df["is_fraud_predicted"].sum(),
                    delta=f"{live_df['is_fraud_predicted'].mean()*100:.1f}%")
        col3.metric("Avg Score", f"{live_df['ensemble_score'].mean():.3f}")
        col4.metric("Alerts (High+Critical)", sum(1 for t in st.session_state.live_transactions
                                                    if t.get("risk_level") in ("Critical", "High")))

    # Alerts panel
    if st.session_state.live_alerts:
        st.markdown("<div class='section-header'>🚨 Active Alerts</div>", unsafe_allow_html=True)
        for alert in reversed(st.session_state.live_alerts[-10:]):
            css_class = "alert-critical" if alert["risk_level"] == "Critical" else "alert-high"
            icon = "🔴" if alert["risk_level"] == "Critical" else "🟠"
            st.markdown(f"""
            <div class="{css_class}">
                {icon} <b>{alert['transaction_id']}</b> &nbsp;|&nbsp;
                ₹{alert['amount']:,.2f} &nbsp;|&nbsp;
                {alert.get('merchant_category','—')} &nbsp;|&nbsp;
                {alert.get('transaction_country','—')} &nbsp;|&nbsp;
                Score: <b>{alert['ensemble_score']:.3f}</b> &nbsp;|&nbsp;
                Risk: <b>{alert['risk_level']}</b>
            </div>""", unsafe_allow_html=True)

    # Live score chart
    if st.session_state.live_transactions:
        live_df = pd.DataFrame(st.session_state.live_transactions)
        live_df["idx"] = range(len(live_df))
        fig10 = go.Figure()
        colors = live_df["risk_level"].map(
            {"Low": GREEN, "Medium": ACCENT, "High": AMBER, "Critical": RED}
        ).fillna("#64748b")
        fig10.add_trace(go.Scatter(
            x=live_df["idx"], y=live_df["ensemble_score"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=1.5),
            marker=dict(color=colors, size=8),
            name="Fraud Score"
        ))
        fig10.add_hline(y=threshold, line_dash="dash", line_color=AMBER,
                        annotation_text="Threshold", annotation_font_color=AMBER)
        fig10.update_layout(xaxis_title="Transaction #", yaxis_title="Fraud Score")
        plotly_theme(fig10, 300)
        st.plotly_chart(fig10, use_container_width=True)

        # Mini table
        st.markdown("<div class='section-header'>Recent Transactions</div>", unsafe_allow_html=True)
        show_cols = ["transaction_id", "amount", "merchant_category",
                     "transaction_country", "ensemble_score", "risk_level", "is_fraud_predicted"]
        st.dataframe(
            live_df[[c for c in show_cols if c in live_df.columns]]
            .sort_values("ensemble_score", ascending=False)
            .head(20)
            .style.background_gradient(subset=["ensemble_score"], cmap="RdYlGn_r"),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SINGLE TRANSACTION CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Single Transaction":
    st.markdown("## 🔬 Single Transaction Checker")
    st.markdown("<div style='color:#64748b;font-size:13px;margin-bottom:20px'>Manually enter transaction details to get an instant fraud score</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=1000000.0, value=15000.0)
        merchant_cat = st.selectbox("Merchant Category",
                                     ["grocery", "retail", "restaurant", "gas_station", "online",
                                      "travel", "healthcare", "entertainment", "atm", "utilities"])
        customer_age = st.number_input("Customer Age", 18, 90, 35)
        account_age  = st.number_input("Account Age (days)", 1, 5000, 500)

    with col2:
        credit_limit = st.number_input("Credit Limit (₹)", 5000, 500000, 50000)
        avail_bal    = st.number_input("Available Balance (₹)", 0, 500000, 20000)
        is_intl      = st.selectbox("International Transaction?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        country      = st.selectbox("Country", ["India", "USA", "UK", "UAE", "Germany", "Singapore", "Australia", "China"])

    with col3:
        hour_of_day  = st.slider("Hour of Day (24h)", 0, 23, 14)
        day_of_week  = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        distance     = st.number_input("Distance from Home (km)", 0.0, 10000.0, 5.0)
        n_tx_24h     = st.number_input("Transactions in last 24h", 0, 100, 2)
        avg_7d       = st.number_input("Avg Amount (7-day)", 100.0, 100000.0, 2000.0)

    if st.button("🔍 Analyze Transaction"):
        tx = {
            "amount": amount,
            "merchant_category": merchant_cat,
            "customer_age": customer_age,
            "account_age_days": account_age,
            "credit_limit": credit_limit,
            "available_balance": avail_bal,
            "is_international": is_intl,
            "transaction_country": country,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "distance_from_home_km": distance,
            "num_transactions_last_24h": n_tx_24h,
            "avg_transaction_amount_7d": avg_7d,
            "utilization_ratio": (credit_limit - avail_bal) / credit_limit if credit_limit else 0,
            "amount_to_limit_ratio": amount / credit_limit if credit_limit else 0
        }
        with st.spinner("Analyzing..."):
            pred = pipeline.predict_single(tx, threshold=threshold)

        score = pred["ensemble_score"]
        risk  = pred["risk_level"]
        color_map = {"Critical": RED, "High": AMBER, "Medium": ACCENT, "Low": GREEN}
        c = color_map.get(risk, ACCENT)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1224,#111827);border:2px solid {c};
                     border-radius:16px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:12px;color:#64748b;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px">Fraud Risk Score</div>
            <div style="font-size:72px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:{c}">{score:.3f}</div>
            <div style="font-size:24px;font-weight:700;color:{c};margin:8px 0">{risk} Risk</div>
            <div style="font-size:14px;color:#94a3b8">{'🚨 FRAUD ALERT — Flag for review' if pred['is_fraud_predicted'] else '✅ Transaction appears legitimate'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Model-Level Scores</div>", unsafe_allow_html=True)
        score_items = {
            "Isolation Forest": pred["score_isolation_forest"],
            "One-Class SVM": pred["score_ocsvm"],
            "Autoencoder": pred["score_autoencoder"],
            "Random Forest": pred["score_random_forest"],
            "Gradient Boosting": pred["score_gradient_boosting"],
        }
        cols = st.columns(len(score_items))
        for col, (name, val) in zip(cols, score_items.items()):
            clr = RED if val > 0.6 else AMBER if val > 0.4 else GREEN
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{name}</div>
                <div class="metric-value" style="font-size:28px;color:{clr}">{val:.3f}</div>
            </div>""", unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"color": c, "family": "IBM Plex Mono", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": TEXT_CLR}},
                "bar": {"color": c},
                "steps": [
                    {"range": [0, 30], "color": "#0d2010"},
                    {"range": [30, 60], "color": "#1e2d4a"},
                    {"range": [60, 80], "color": "#3a2200"},
                    {"range": [80, 100], "color": "#3a0010"},
                ],
                "threshold": {"line": {"color": AMBER, "width": 3}, "thickness": 0.8, "value": threshold * 100}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor=PLOT_BG, font=dict(color=TEXT_CLR, family="IBM Plex Sans"), height=280, margin=dict(l=30,r=30,t=30,b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
