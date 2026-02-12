import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(page_title="OTT Content Intelligence Dashboard", layout="wide")

st.title("OTT Content Acquisition Intelligence Interface")
st.markdown("Pre-release Audience Signal Analysis for Indian Market")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_sentiment.csv")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.date
    return df

df = load_data()

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("Data Scope")

video_type_filter = st.sidebar.selectbox(
    "Video Type",
    ["All"] + list(df["video_type"].unique()) if "video_type" in df.columns else ["All"]
)

filtered_df = df.copy()

if video_type_filter != "All" and "video_type" in df.columns:
    filtered_df = filtered_df[filtered_df["video_type"] == video_type_filter]

# =====================================================
# COMPUTE CORE METRICS
# =====================================================
total_comments = len(filtered_df)
positive_ratio = (filtered_df["sentiment_label"] == "positive").mean()
negative_ratio = (filtered_df["sentiment_label"] == "negative").mean()
net_sentiment = filtered_df["compound"].mean()
engagement_intensity = filtered_df["likes"].mean()

risk_index = negative_ratio * 100

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Intelligence",
    "Audience Structure",
    "Engagement & Momentum",
    "ML Decision Engine"
])

# =====================================================
# TAB 1 — EXECUTIVE INTELLIGENCE
# =====================================================
with tab1:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Audience Signals Captured", total_comments)
    col2.metric("Positive Sentiment Ratio", f"{round(positive_ratio*100,2)}%")
    col3.metric("Negative Risk Index", f"{round(risk_index,2)}%")
    col4.metric("Net Sentiment Score", round(net_sentiment,3))

    st.divider()

    # Feasibility Recommendation Logic
    if positive_ratio > 0.65 and negative_ratio < 0.15:
        recommendation = "High Acquisition Potential"
        confidence = "Strong"
    elif positive_ratio > 0.50:
        recommendation = "Moderate Acquisition Potential"
        confidence = "Medium"
    else:
        recommendation = "Acquisition Risk Present"
        confidence = "Caution Advised"

    col5, col6 = st.columns(2)

    col5.metric("Acquisition Recommendation", recommendation)
    col6.metric("Confidence Level", confidence)

    st.markdown("""
    This recommendation is derived from aggregated polarity balance, 
    emotional drivers, and engagement strength.
    """)

# =====================================================
# TAB 2 — AUDIENCE STRUCTURE
# =====================================================
with tab2:

    st.subheader("Sentiment Distribution")

    sentiment_counts = filtered_df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig1 = px.bar(sentiment_counts,
                  x="Sentiment",
                  y="Count",
                  color="Sentiment",
                  title="Sentiment Class Distribution")

    st.plotly_chart(fig1, use_container_width=True)

    if "joy" in filtered_df.columns:
        st.subheader("Emotional Contribution Matrix")

        emotion_cols = ["joy", "trust", "anticipation", "sadness"]
        emotion_data = filtered_df[emotion_cols].mean()

        fig2 = go.Figure(data=go.Heatmap(
            z=[emotion_data.values],
            x=emotion_cols,
            y=["Average Emotional Strength"],
            colorscale="Blues"
        ))

        fig2.update_layout(title="Emotion Intensity Overview")
        st.plotly_chart(fig2, use_container_width=True)

    if "video_type" in filtered_df.columns:
        st.subheader("Sentiment by Video Format")

        format_sentiment = filtered_df.groupby("video_type")["compound"].mean().reset_index()

        fig3 = px.bar(format_sentiment,
                      x="video_type",
                      y="compound",
                      title="Average Sentiment by Format")

        st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# TAB 3 — ENGAGEMENT & MOMENTUM
# =====================================================
with tab3:

    st.subheader("Sentiment Momentum Over Time")

    if "date" in filtered_df.columns:
        trend = filtered_df.groupby("date")["compound"].mean().reset_index()

        fig4 = px.line(trend,
                       x="date",
                       y="compound",
                       title="Daily Sentiment Trend")

        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Engagement Intensity Trend")

    if "date" in filtered_df.columns:
        volume_trend = filtered_df.groupby("date").size().reset_index(name="Comment Volume")

        fig5 = px.line(volume_trend,
                       x="date",
                       y="Comment Volume",
                       title="Comment Volume Over Time")

        st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# TAB 4 — ML DECISION ENGINE
# =====================================================
with tab4:

    st.subheader("Supervised Sentiment Classification Model")

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

    st.markdown("""
    The classifier has been trained on 12,000+ audience comments. 
    It generalizes sentiment detection at scale for acquisition analytics.
    """)

    user_input = st.text_area("Test Comment Classification")

    if st.button("Run Classification"):
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        predicted_label = le.inverse_transform(prediction)[0]
        st.write("Predicted Sentiment:", predicted_label)
