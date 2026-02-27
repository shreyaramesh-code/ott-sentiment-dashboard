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
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide")
st.title("OTT Content Marketing Analytics Dashboard")
st.markdown("Decision Analytics Framework for Korean Content – Indian Market")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("youtube_master_clean.csv")

df = load_data()

# =====================================================
# DETECT IMPORTANT COLUMNS
# =====================================================

sentiment_col = next((c for c in df.columns if "sentiment" in c.lower()), None)
compound_col = next((c for c in df.columns if "compound" in c.lower()), None)
text_col = next((c for c in df.columns if "text" in c.lower()), None)
date_col = next((c for c in df.columns if "publish" in c.lower()), None)

if not sentiment_col or not compound_col:
    st.error("Dataset missing sentiment columns.")
    st.stop()

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date"] = df[date_col].dt.date

# =====================================================
# CONTENT SELECTION
# =====================================================

st.sidebar.header("Content Selection")

selected_title = st.sidebar.selectbox(
    "Select Title",
    ["Squid Game", "Squid Game 2", "Can This Love Be Translated"]
)

# =====================================================
# FILTER DATA BY TITLE KEYWORD
# =====================================================

if text_col:
    filtered_df = df[df[text_col].str.contains(selected_title, case=False, na=False)]
    if filtered_df.empty:
        filtered_df = df.copy()
else:
    filtered_df = df.copy()

# =====================================================
# GOOGLE TRENDS
# =====================================================

pytrends = TrendReq(hl='en-IN', tz=330)

# Fixed timeframe until Jan 2026
TIMEFRAME = '2024-01-01 2026-01-31'


@st.cache_data(show_spinner=False)
def get_trends(keyword):

    try:
        pytrends.build_payload(
            [keyword],
            timeframe=TIMEFRAME,
            geo='IN'
        )
        data = pytrends.interest_over_time()
        return data

    except:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_region_interest(keyword):

    try:
        pytrends.build_payload(
            [keyword],
            timeframe=TIMEFRAME,
            geo='IN'
        )
        data = pytrends.interest_by_region(
            resolution='REGION',
            inc_low_vol=True
        )
        return data

    except:
        return pd.DataFrame()


trend_data = get_trends(selected_title)
region_data = get_region_interest(selected_title)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Demand",
    "Audience Perception",
    "Risk Analysis",
    "Content Momentum Engine",
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

        st.metric("Current Demand Index",
                  int(trend_data[selected_title].iloc[-1]))

    st.subheader("State-wise Demand Intensity")

    if not region_data.empty:

        region_df = region_data.reset_index()
        region_df.columns = ["State", "Interest"]
        region_df = region_df.sort_values("Interest", ascending=False)

        heatmap_fig = px.imshow(
            region_df[["Interest"]],
            labels=dict(color="Search Interest"),
            x=["Interest Score"],
            y=region_df["State"],
            color_continuous_scale="Reds",
            aspect="auto"
        )

        heatmap_fig.update_layout(height=600)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown("Top 5 High-Demand States")
        st.dataframe(region_df.head(5))

# =====================================================
# TAB 2 – AUDIENCE PERCEPTION
# =====================================================

with tab2:

    positive_ratio = (filtered_df[sentiment_col].str.lower() == "positive").mean()
    negative_ratio = (filtered_df[sentiment_col].str.lower() == "negative").mean()
    net_sentiment = filtered_df[compound_col].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive Sentiment %", f"{round(positive_ratio*100,2)}%")
    col2.metric("Negative Sentiment %", f"{round(negative_ratio*100,2)}%")
    col3.metric("Net Sentiment Score", round(net_sentiment,3))

    emotion_cols = [c for c in ["joy","anticipation","trust","sadness"] if c in filtered_df.columns]

    if emotion_cols:
        emotion_mean = filtered_df[emotion_cols].mean()
        fig = go.Figure(data=go.Bar(x=emotion_mean.index, y=emotion_mean.values))
        fig.update_layout(title="Dominant Emotional Drivers")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3 – RISK ANALYSIS
# =====================================================

with tab3:

    st.subheader("Sentiment Distribution")

    sentiment_counts = filtered_df[sentiment_col].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment Score Distribution")

    fig2 = px.histogram(filtered_df, x=compound_col, nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

    sentiment_std = filtered_df[compound_col].std()
    risk_adjusted_score = positive_ratio - negative_ratio

    col1, col2 = st.columns(2)
    #col1.metric("Polarization Index", round(sentiment_std,3))
    col2.metric("Risk Adjusted Sentiment", round(risk_adjusted_score,3))

# =====================================================
# TAB 4 – Content Momentum ENGINE
# =====================================================

with tab4:

    st.subheader("Content Momentum Evaluation")

    demand_score = (trend_data[selected_title].iloc[-1] / 100) if not trend_data.empty else 0
    sentiment_score = positive_ratio
    momentum_score = filtered_df.groupby("date").size().pct_change().mean() if "date" in filtered_df.columns else 0
    momentum_score = 0 if pd.isna(momentum_score) else min(max(momentum_score,0),1)
    risk_score = 1 - negative_ratio

    content_momentum_index = (
        0.3 * demand_score +
        0.3 * sentiment_score +
        0.2 * momentum_score +
        0.2 * risk_score
    )

    st.metric("Content_Momentum_Index", round(content_momentum_index,2))

    if content_momentum_index > 0.7:
        st.success("Recommendation: Strong content Candidate")
    elif content_momentum_index > 0.5:
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

        user_input = st.text_area(
            "Simulate Audience Reaction (Enter tagline / trailer caption / dialogue)"
        )

        if st.button("Simulate Sentiment"):

            if user_input.strip():
                vec = vectorizer.transform([user_input])
                pred = model.predict(vec)
                label = le.inverse_transform(pred)[0]
                st.success(f"Predicted Audience Sentiment: {label.upper()}")
            else:
                st.warning("Please enter text to simulate.")

        st.divider()
        st.subheader("Model Reliability Overview")

        X = vectorizer.transform(df[text_col])
        y = le.transform(df[sentiment_col])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = (y_pred == y_test).mean()
        st.metric("Model Accuracy", round(float(accuracy),3))

        cm = confusion_matrix(y_test, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=le.classes_,
            y=le.classes_,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size":16}
        ))

        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            title="Confusion Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)



