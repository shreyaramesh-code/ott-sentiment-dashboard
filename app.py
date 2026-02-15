import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pytrends.request import TrendReq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide")
st.title("Pre-Release OTT Acquisition Intelligence System")
st.markdown("Decision Analytics Framework for Korean Content – Indian Market")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("youtube_master_clean.csv")

df = load_data()

# =====================================================
# AUTO DETECT REQUIRED COLUMNS
# =====================================================

sentiment_col = next((c for c in df.columns if "sentiment" in c.lower()), None)
compound_col = next((c for c in df.columns if "compound" in c.lower()), None)
text_col = next((c for c in df.columns if "text" in c.lower()), None)
date_col = next((c for c in df.columns if "publish" in c.lower()), None)

if not sentiment_col or not compound_col:
    st.error("Dataset missing required sentiment columns.")
    st.stop()

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date"] = df[date_col].dt.date

# =====================================================
# GOOGLE TRENDS
# =====================================================

st.sidebar.header("Content Selection")

selected_title = st.sidebar.selectbox(
    "Select Title",
    ["Squid Game", "Squid Game 2", "Can This Love Be Translated"]
)

pytrends = TrendReq(hl='en-IN', tz=330)

@st.cache_data(ttl=3600)
def get_trends(keyword):
    pytrends.build_payload([keyword], timeframe='today 3-m', geo='IN')
    return pytrends.interest_over_time()

trend_data = get_trends(selected_title)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Demand",
    "Audience Perception",
    "Risk Analysis",
    "Acquisition Engine",
    "Tools & Model Validation"
])

# =====================================================
# TAB 1 – MARKET DEMAND
# =====================================================

with tab1:

    st.subheader("Search Demand Trend – India")

    if not trend_data.empty:
        fig = px.line(trend_data, y=selected_title)
        st.plotly_chart(fig, use_container_width=True)

        latest_value = trend_data[selected_title].iloc[-1]
        st.metric("Current Demand Index", int(latest_value))

    else:
        st.warning("No trend data available.")


# =====================================================
# TAB 2 – AUDIENCE PERCEPTION
# =====================================================

with tab2:

    positive_ratio = (df[sentiment_col].str.lower() == "positive").mean()
    negative_ratio = (df[sentiment_col].str.lower() == "negative").mean()
    net_sentiment = df[compound_col].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive Sentiment %", f"{round(positive_ratio*100,2)}%")
    col2.metric("Negative Sentiment %", f"{round(negative_ratio*100,2)}%")
    col3.metric("Net Sentiment Score", round(net_sentiment,3))

    emotion_cols = [c for c in ["joy","anticipation","trust","sadness"] if c in df.columns]

    if emotion_cols:
        emotion_mean = df[emotion_cols].mean()
        fig = go.Figure(data=go.Bar(x=emotion_mean.index, y=emotion_mean.values))
        fig.update_layout(title="Dominant Emotional Drivers")
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 3 – RISK ANALYSIS
# =====================================================

with tab3:

    sentiment_std = df[compound_col].std()
    polarization_index = round(sentiment_std, 3)

    col1, col2 = st.columns(2)
    col1.metric("Sentiment Volatility", polarization_index)
    col2.metric("Risk Adjusted Score", round(positive_ratio - negative_ratio,3))

    st.markdown("""
    Higher volatility indicates polarization risk.  
    Risk Adjusted Score reflects net positivity strength.
    """)


# =====================================================
# TAB 4 – ACQUISITION ENGINE
# =====================================================

with tab4:

    st.subheader("Composite Acquisition Evaluation")

    # Normalize components
    demand_score = (trend_data[selected_title].iloc[-1] / 100) if not trend_data.empty else 0
    sentiment_score = positive_ratio
    momentum_score = df.groupby("date").size().pct_change().mean() if "date" in df.columns else 0
    momentum_score = 0 if pd.isna(momentum_score) else min(max(momentum_score,0),1)
    risk_score = 1 - negative_ratio

    acquisition_score = (
        0.3 * demand_score +
        0.3 * sentiment_score +
        0.2 * momentum_score +
        0.2 * risk_score
    )

    st.metric("Acquisition Readiness Score", round(acquisition_score,2))

    if acquisition_score > 0.7:
        st.success("Recommendation: Strong Acquisition Candidate")
    elif acquisition_score > 0.5:
        st.warning("Recommendation: Moderate Potential")
    else:
        st.error("Recommendation: Caution – Evaluate Further")

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[demand_score, sentiment_score, momentum_score, risk_score],
        theta=['Demand','Sentiment','Momentum','Low Risk'],
        fill='toself'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])))
    st.plotly_chart(radar_fig, use_container_width=True)


# =====================================================
# TAB 5 – TOOLS & MODEL VALIDATION
# =====================================================

with tab5:

    st.subheader("Content Perception Simulation Tool")

    if text_col:
        @st.cache_resource
        def train_model():
            vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
            X = vectorizer.fit_transform(df[text_col])
            le = LabelEncoder()
            y = le.fit_transform(df[sentiment_col])
            model = LinearSVC()
            model.fit(X, y)
            return vectorizer, model, le

        vectorizer, model, le = train_model()

        user_input = st.text_area("Simulate Audience Reaction (Enter tagline / dialogue / trailer caption)")

        if st.button("Simulate Sentiment Response"):

            if user_input.strip():
                vec = vectorizer.transform([user_input])
                pred = model.predict(vec)
                label = le.inverse_transform(pred)[0]

                st.success(f"Predicted Public Sentiment: {label.upper()}")

            else:
                st.warning("Enter text to simulate audience perception.")

    st.divider()
    st.subheader("Model Reliability Overview")

    if text_col:
        X = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform(df[text_col])
        le = LabelEncoder()
        y = le.fit_transform(df[sentiment_col])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearSVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = (y_pred == y_test).mean()
        st.metric("Model Accuracy", round(float(accuracy),3))

        cm = confusion_matrix(y_test, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=le.classes_,
            y=le.classes_,
            colorscale="Blues"
        ))
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        The sentiment classifier underpins audience perception metrics used across the system.  
        Model validation ensures signal reliability before acquisition scoring.
        """)
