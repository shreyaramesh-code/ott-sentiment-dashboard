import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pytrends.request import TrendReq

st.set_page_config(layout="wide")
st.title("Pre-Release OTT Content Intelligence Platform")
st.markdown("AI-Augmented Market & Audience Signal System")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("youtube_master_clean.csv")
    return df

df = load_data()

# =====================================================
# AUTO DETECT CRITICAL COLUMNS (NO ASSUMPTIONS)
# =====================================================

sentiment_col = None
for col in df.columns:
    if "sentiment" in col.lower():
        sentiment_col = col
        break

compound_col = None
for col in df.columns:
    if "compound" in col.lower():
        compound_col = col
        break

date_col = None
for col in df.columns:
    if "publish" in col.lower():
        date_col = col
        break

if sentiment_col is None:
    st.error("No sentiment column detected in dataset.")
    st.stop()

if compound_col is None:
    st.error("No compound score column detected.")
    st.stop()

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date"] = df[date_col].dt.date

# =====================================================
# GOOGLE TRENDS
# =====================================================

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
    return pytrends.interest_over_time()

@st.cache_data(ttl=3600)
def get_region_interest(keyword):
    pytrends.build_payload([keyword], timeframe='today 3-m', geo='IN')
    return pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)

trend_data = get_trends(selected_title)
region_data = get_region_interest(selected_title)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Market Signals",
    "Audience Intelligence",
    "Investment Radar",
    "Benchmark & Velocity"
])

# =====================================================
# TAB 1 – MARKET SIGNALS
# =====================================================

with tab1:

    st.subheader("Interest Over Time – India")

    if not trend_data.empty:
        fig1 = px.line(trend_data, y=selected_title)
        st.plotly_chart(fig1, use_container_width=True)

    # ==============================
    # 🔥 FIXED GEOGRAPHIC MAP PART
    # ==============================

    st.subheader("Geographic Opportunity – India")

    if not region_data.empty:

        region_df = region_data.reset_index()
        region_df.columns = ["State", "Interest"]
        region_df = region_df[region_df["Interest"] > 0]
        region_df["State"] = region_df["State"].str.strip()

        # If Google returns state codes
        if region_df["State"].str.len().max() <= 3:
            state_code_map = {
                "MH": "Maharashtra",
                "KA": "Karnataka",
                "TN": "Tamil Nadu",
                "DL": "Delhi",
                "UP": "Uttar Pradesh",
                "WB": "West Bengal",
                "RJ": "Rajasthan",
                "GJ": "Gujarat",
                "HR": "Haryana",
                "PB": "Punjab",
                "MP": "Madhya Pradesh",
                "AP": "Andhra Pradesh",
                "TS": "Telangana",
                "KL": "Kerala",
                "BR": "Bihar",
                "OR": "Odisha",
                "AS": "Assam"
            }
            region_df["State"] = region_df["State"].replace(state_code_map)

        fig2 = px.choropleth(
            region_df,
            locations="State",
            locationmode="country names",
            color="Interest",
            color_continuous_scale="Blues",
            scope="asia"
        )

        fig2.update_geos(
            visible=True,
            showcountries=True,
            showcoastlines=False,
            showland=True,
            fitbounds="locations",
            lataxis_range=[6, 38],
            lonaxis_range=[68, 98]
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Executive insight
        top_states = region_df.sort_values("Interest", ascending=False).head(5)
        st.markdown("### Top 5 High-Demand States")
        st.dataframe(top_states)

    else:
        st.warning("Google Trends returned no regional data.")

# =====================================================
# TAB 2 – AUDIENCE INTELLIGENCE
# =====================================================

with tab2:

    positive_ratio = (df[sentiment_col].str.lower() == "positive").mean()
    negative_ratio = (df[sentiment_col].str.lower() == "negative").mean()
    net_sentiment = df[compound_col].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive Ratio", f"{round(positive_ratio*100,2)}%")
    col2.metric("Negative Risk", f"{round(negative_ratio*100,2)}%")
    col3.metric("Net Sentiment Score", round(net_sentiment,3))

    emotion_cols = ["joy","trust","anticipation","sadness"]
    available_emotions = [col for col in emotion_cols if col in df.columns]

    if len(available_emotions) > 0:
        emotion_means = df[available_emotions].mean()
        fig3 = go.Figure(data=go.Bar(
            x=emotion_means.index,
            y=emotion_means.values
        ))
        fig3.update_layout(title="Emotional Drivers")
        st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# TAB 3 – INVESTMENT RADAR
# =====================================================

with tab3:

    positive_ratio = (df[sentiment_col].str.lower() == "positive").mean()
    negative_ratio = (df[sentiment_col].str.lower() == "negative").mean()

    ssi = (df[compound_col].mean() + 1) / 2

    if "joy" in df.columns and "anticipation" in df.columns:
        eei = (df["joy"].mean() + df["anticipation"].mean()) / 2
    else:
        eei = ssi

    if "date" in df.columns:
        velocity = df.groupby("date").size().pct_change().mean()
        velocity = 0 if pd.isna(velocity) else velocity
    else:
        velocity = 0

    categories = ['Sentiment Strength','Excitement','Momentum','Low Risk']
    values = [
        ssi,
        eei,
        velocity,
        1-negative_ratio
    ]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig4.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False
    )

    st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# TAB 4 – VELOCITY
# =====================================================

with tab4:

    if "date" in df.columns:
        volume_trend = df.groupby("date").size().reset_index(name="Volume")
        fig5 = px.line(volume_trend, x="date", y="Volume")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No timestamp column available for velocity analysis.")
