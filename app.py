import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="OTT Sentiment Intelligence", layout="wide")

st.title("🎬 AI-Driven OTT Sentiment Intelligence Dashboard")
st.markdown("### Indian Audience Perception Analysis of Korean Entertainment")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_sentiment.csv")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.date
    return df

df = load_data()

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("🔎 Filters")

sentiment_filter = st.sidebar.selectbox(
    "Select Sentiment",
    ["All"] + list(df["sentiment_label"].unique())
)

video_filter = st.sidebar.selectbox(
    "Select Video Type",
    ["All"] + list(df["video_type"].unique()) if "video_type" in df.columns else ["All"]
)

filtered_df = df.copy()

if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["sentiment_label"] == sentiment_filter]

if video_filter != "All" and "video_type" in df.columns:
    filtered_df = filtered_df[filtered_df["video_type"] == video_filter]

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "😊 Emotion Analytics",
    "🤖 ML Engine",
    "📈 Market Insights"
])

# ===================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ===================================================
with tab1:

    total_comments = len(filtered_df)
    positive_pct = round((filtered_df["sentiment_label"] == "positive").mean() * 100, 2)
    neutral_pct = round((filtered_df["sentiment_label"] == "neutral").mean() * 100, 2)
    negative_pct = round((filtered_df["sentiment_label"] == "negative").mean() * 100, 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Comments", total_comments)
    col2.metric("Positive %", f"{positive_pct}%")
    col3.metric("Neutral %", f"{neutral_pct}%")
    col4.metric("Negative %", f"{negative_pct}%")

    st.divider()

    sentiment_counts = filtered_df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig1 = px.pie(sentiment_counts,
                  names="Sentiment",
                  values="Count",
                  hole=0.5,
                  title="Sentiment Distribution")

    st.plotly_chart(fig1, use_container_width=True)

    # AI Insight
    if positive_pct > 60:
        insight = "Strong positive resonance among Indian viewers."
    elif negative_pct > 30:
        insight = "Mixed reception – localization or adaptation may be required."
    else:
        insight = "Balanced engagement with moderate audience sentiment."

    st.info(f"📌 AI Insight: {insight}")

# ===================================================
# TAB 2 — EMOTION ANALYTICS
# ===================================================
with tab2:

    if "joy" in filtered_df.columns:
        emotion_cols = ["joy", "trust", "anticipation", "sadness"]
        emotion_means = filtered_df[emotion_cols].mean().reset_index()
        emotion_means.columns = ["Emotion", "Average Score"]

        fig2 = px.bar(emotion_means,
                      x="Emotion",
                      y="Average Score",
                      title="Emotional Engagement Overview")

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧠 Word Cloud")

    text_data = " ".join(filtered_df["analysis_text"].dropna())
    wordcloud = WordCloud(width=800, height=400,
                          background_color="white").generate(text_data)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ===================================================
# TAB 3 — ML ENGINE
# ===================================================
with tab3:

    st.subheader("SVM Sentiment Classification Engine")

    @st.cache_resource
    def train_model(data):
        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        X = vectorizer.fit_transform(data["analysis_text"])

        le = LabelEncoder()
        y = le.fit_transform(data["sentiment_label"])

        model = LinearSVC()
        model.fit(X, y)

        return vectorizer, model, le

    vectorizer, model, le = train_model(df)

    user_input = st.text_area("Enter a comment to predict sentiment:")

    if st.button("Predict Sentiment"):
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        predicted_label = le.inverse_transform(prediction)[0]
        st.success(f"Predicted Sentiment: **{predicted_label.upper()}**")

# ===================================================
# TAB 4 — MARKET INSIGHTS
# ===================================================
with tab4:

    st.subheader("📈 Sentiment Trend Over Time")

    if "date" in filtered_df.columns:
        daily_sentiment = filtered_df.groupby("date")["compound"].mean().reset_index()

        fig3 = px.line(daily_sentiment,
                       x="date",
                       y="compound",
                       title="Average Sentiment Trend")

        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    st.subheader("📊 Feasibility Indicator")

    avg_sentiment = filtered_df["compound"].mean()

    if avg_sentiment > 0.3:
        feasibility = "HIGH"
    elif avg_sentiment > 0.1:
        feasibility = "MEDIUM"
    else:
        feasibility = "LOW"

    st.metric("Content Feasibility Level", feasibility)
