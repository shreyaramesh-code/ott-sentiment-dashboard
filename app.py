import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Content Intelligence Engine", layout="wide")

st.title("Content Acquisition Intelligence Engine")
st.markdown("Pre-Release OTT Investment Signal Dashboard – Indian Market")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_sentiment.csv")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.date
    return df

df = load_data()

# -------------------------------------------------
# VIDEO FILTER
# -------------------------------------------------
video_ids = df["video_id"].unique().tolist()

selected_video = st.sidebar.selectbox("Select Content (Video ID)", ["All"] + video_ids)

filtered_df = df.copy()
if selected_video != "All":
    filtered_df = filtered_df[filtered_df["video_id"] == selected_video]

# -------------------------------------------------
# CALCULATE INDICES
# -------------------------------------------------

# Sentiment Strength Index
ssi = filtered_df["compound"].mean()
ssi_scaled = (ssi + 1) / 2 * 100  # normalize from -1,1 → 0,100

# Polarization Risk Index
negative_ratio = (filtered_df["sentiment_label"] == "negative").mean()
sentiment_variance = filtered_df["compound"].var()
pri = (negative_ratio * 100) + (sentiment_variance * 50)

# Engagement Velocity Index
daily_counts = filtered_df.groupby("date").size().reset_index(name="volume")
if len(daily_counts) > 5:
    first_half = daily_counts["volume"].iloc[:len(daily_counts)//2].mean()
    second_half = daily_counts["volume"].iloc[len(daily_counts)//2:].mean()
    evi = ((second_half - first_half) / (first_half + 1)) * 100
else:
    evi = 0

# Emotional Excitement Index
if "joy" in filtered_df.columns:
    eei = (filtered_df["joy"].mean() + filtered_df["anticipation"].mean()) * 50
else:
    eei = 0

# Investment Confidence Score
ics = (0.30 * ssi_scaled/100 +
       0.25 * eei/100 +
       0.25 * (evi/100) -
       0.20 * (pri/100))

ics_score = round(max(min(ics * 10, 10), 0), 2)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Acquisition Overview",
    "Content Diagnostics",
    "Momentum & Growth",
    "Investment Model"
])

# -------------------------------------------------
# TAB 1 — OVERVIEW
# -------------------------------------------------
with tab1:

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentiment Strength Index", round(ssi_scaled,2))
    col2.metric("Polarization Risk Index", round(pri,2))
    col3.metric("Engagement Velocity Index", round(evi,2))
    col4.metric("Emotional Excitement Index", round(eei,2))

    st.divider()

    st.metric("Investment Confidence Score (0–10)", ics_score)

    if ics_score >= 7:
        st.success("Strategic Acquisition Opportunity – Strong Pre-Release Signal")
    elif ics_score >= 5:
        st.warning("Moderate Potential – Conditional Investment Recommended")
    else:
        st.error("High Risk – Acquisition Caution Advised")

# -------------------------------------------------
# TAB 2 — CONTENT DIAGNOSTICS
# -------------------------------------------------
with tab2:

    st.subheader("Sentiment Distribution")

    sentiment_counts = filtered_df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig1 = px.bar(sentiment_counts,
                  x="Sentiment",
                  y="Count",
                  color="Sentiment")
    st.plotly_chart(fig1, use_container_width=True)

    if "joy" in filtered_df.columns:
        st.subheader("Emotion Contribution Heatmap")

        emotion_cols = ["joy","trust","anticipation","sadness"]
        emotion_means = filtered_df[emotion_cols].mean()

        fig2 = go.Figure(data=go.Heatmap(
            z=[emotion_means.values],
            x=emotion_cols,
            y=["Intensity"],
            colorscale="Blues"
        ))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Compound Sentiment Distribution")

    fig3 = px.histogram(filtered_df,
                        x="compound",
                        nbins=30)
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# TAB 3 — MOMENTUM
# -------------------------------------------------
with tab3:

    if "date" in filtered_df.columns:
        daily_sentiment = filtered_df.groupby("date")["compound"].mean().reset_index()

        fig4 = px.line(daily_sentiment,
                       x="date",
                       y="compound",
                       title="Sentiment Momentum")
        st.plotly_chart(fig4, use_container_width=True)

        volume_trend = filtered_df.groupby("date").size().reset_index(name="Volume")
        fig5 = px.line(volume_trend,
                       x="date",
                       y="Volume",
                       title="Audience Signal Volume Growth")
        st.plotly_chart(fig5, use_container_width=True)

# -------------------------------------------------
# TAB 4 — INVESTMENT SIMULATION
# -------------------------------------------------
with tab4:

    st.subheader("Scenario Sensitivity Model")

    marketing_push = st.slider("Marketing Amplification Factor", 0.5, 2.0, 1.0)

    adjusted_evi = evi * marketing_push

    adjusted_ics = (0.30 * ssi_scaled/100 +
                    0.25 * eei/100 +
                    0.25 * (adjusted_evi/100) -
                    0.20 * (pri/100))

    adjusted_score = round(max(min(adjusted_ics * 10, 10), 0), 2)

    st.metric("Adjusted Investment Confidence Score", adjusted_score)

    st.markdown("""
    This simulation estimates how promotional amplification impacts acquisition confidence.
    """)
