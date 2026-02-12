import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="OTT Sentiment Intelligence Dashboard", layout="wide")

st.title("🎬 AI-Driven Content Feasibility Dashboard")
st.markdown("Indian Audience Perception Analysis of Korean Entertainment")

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_sentiment.csv")
    return df

df = load_data()

# --------------------------
# KPI SECTION
# --------------------------
total_comments = len(df)
positive_pct = round((df["sentiment_label"] == "positive").mean() * 100, 2)
neutral_pct = round((df["sentiment_label"] == "neutral").mean() * 100, 2)
negative_pct = round((df["sentiment_label"] == "negative").mean() * 100, 2)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Comments", total_comments)
col2.metric("Positive %", f"{positive_pct}%")
col3.metric("Neutral %", f"{neutral_pct}%")
col4.metric("Negative %", f"{negative_pct}%")

st.divider()

# --------------------------
# SENTIMENT DISTRIBUTION
# --------------------------
st.subheader("📊 Sentiment Distribution")

sentiment_counts = df["sentiment_label"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig1 = px.pie(sentiment_counts, names="Sentiment", values="Count",
              hole=0.5, title="Overall Sentiment Split")

st.plotly_chart(fig1, use_container_width=True)

# --------------------------
# EMOTION DISTRIBUTION (if available)
# --------------------------
if "joy" in df.columns:
    st.subheader("😊 Emotional Engagement Overview")

    emotion_cols = ["joy", "trust", "anticipation", "sadness"]

    emotion_means = df[emotion_cols].mean().reset_index()
    emotion_means.columns = ["Emotion", "Average Score"]

    fig2 = px.bar(emotion_means,
                  x="Emotion",
                  y="Average Score",
                  title="Average Emotional Intensity")

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --------------------------
# TRAILER vs REACTION
# --------------------------
if "video_type" in df.columns:
    st.subheader("🎥 Trailer vs Reaction Sentiment")

    video_sentiment = df.groupby("video_type")["compound"].mean().reset_index()

    fig3 = px.bar(video_sentiment,
                  x="video_type",
                  y="compound",
                  title="Average Sentiment by Video Type")

    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --------------------------
# LIVE SENTIMENT PREDICTION
# --------------------------
st.subheader("🤖 Live Sentiment Prediction")

user_input = st.text_area("Enter a comment to predict sentiment:")

if st.button("Predict Sentiment"):

    # Train model quickly on load
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vectorizer.fit_transform(df["analysis_text"])

    le = LabelEncoder()
    y = le.fit_transform(df["sentiment_label"])

    model = LinearSVC()
    model.fit(X, y)

    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)

    predicted_label = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Sentiment: **{predicted_label.upper()}**")

st.divider()

# --------------------------
# FEASIBILITY SCORE (Simple Rule-Based)
# --------------------------
st.subheader("📈 Feasibility Indicator")

avg_sentiment = df["compound"].mean()

if avg_sentiment > 0.3:
    feasibility = "HIGH"
elif avg_sentiment > 0.1:
    feasibility = "MEDIUM"
else:
    feasibility = "LOW"

st.metric("Content Feasibility Level", feasibility)
