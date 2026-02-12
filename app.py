import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pytrends.request import TrendReq

st.set_page_config(layout="wide")
st.title("Pre-Release OTT Content Intelligence Platform")
st.markdown("AI-Augmented Market & Audience Signal System")

# ============================
# LOAD YOUTUBE DATA
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_master_clean.csv")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.date
    return df

df = load_data()

# ============================
# GOOGLE TRENDS REAL-TIME
# ============================

st.sidebar.header("Market Intelligence")

selected_title = st.sidebar.selectbox(
    "Select Title for Market Analysis",
    [
        "Squid Game",
        "Squid Game 2",
        "Can This Love Be Translated"
    ]
)

pytrends = TrendReq(hl='en-IN', tz=330)

@st.cache_data(ttl=3600)
def get_trends(keyword):
    pytrends.build_payload([keyword], timeframe='today 3-m', geo='IN')
    interest_over_time = pytrends.interest_over_time()
    region_interest = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)
    return interest_over_time, region_interest

trend_data, region_data = get_trends(selected_title)

# ============================
# TABS
# ============================

tab1, tab2, tab3, tab4 = st.tabs([
    "Market Signals",
    "Audience Intelligence",
    "Investment Radar",
    "Benchmark & Velocity"
])

# ======================================================
# TAB 1 – MARKET SIGNALS (Google Trends)
# ======================================================

with tab1:

    st.subheader("Interest Over Time – India")

    if not trend_data.empty:
        fig1 = px.line(trend_data, y=selected_title,
                       title="Google Search Interest Trend")
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Geographic Opportunity Map")

    if not region_data.empty:
        region_df = region_data.reset_index()
        region_df.columns = ["State", "Interest"]

        fig2 = px.choropleth(
            region_df,
            locations="State",
            locationmode="country names",
            color="Interest",
            color_continuous_scale="Blues",
            title="Search Interest by Region (India)"
        )

        st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# TAB 2 – AUDIENCE INTELLIGENCE
# ======================================================

with tab2:

    positive_ratio = (df["sentiment_label"] == "positive").mean()
    negative_ratio = (df["sentiment_label"] == "negative").mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive Ratio", f"{round(positive_ratio*100,2)}%")
    col2.metric("Negative Risk", f"{round(negative_ratio*100,2)}%")
    col3.metric("Net Sentiment", round(df["compound"].mean(),3))

    if "joy" in df.columns:
        emotion_cols = ["joy","trust","anticipation","sadness"]
        emotion_means = df[emotion_cols].mean()

        fig3 = go.Figure(data=go.Bar(
            x=emotion_cols,
            y=emotion_means
        ))
        fig3.update_layout(title="Emotional Drivers")
        st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# TAB 3 – INVESTMENT RADAR
# ======================================================

with tab3:

    ssi = (df["compound"].mean()+1)/2
    eei = (df["joy"].mean()+df["anticipation"].mean())/2
    risk = negative_ratio
    velocity = df.groupby("date").size().pct_change().mean()

    categories = ['Sentiment Strength','Excitement','Momentum','Low Risk']
    values = [
        ssi,
        eei,
        velocity if not np.isnan(velocity) else 0,
        1-risk
    ]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig4.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False,
        title="Composite Investment Radar"
    )

    st.plotly_chart(fig4, use_container_width=True)

# ======================================================
# TAB 4 – BENCHMARK & VIRAL VELOCITY
# ======================================================

with tab4:

    st.subheader("Audience Signal Growth")

    volume_trend = df.groupby("date").size().reset_index(name="Volume")

    fig5 = px.line(volume_trend,
                   x="date",
                   y="Volume",
                   title="Comment Velocity")

    st.plotly_chart(fig5, use_container_width=True)

    if velocity and velocity > 0:
        st.success("Viral Acceleration Detected")
    else:
        st.warning("No Significant Viral Acceleration")


