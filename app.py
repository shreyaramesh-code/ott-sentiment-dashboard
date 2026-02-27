import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
# FILTER DATA
# =====================================================

if text_col:
    filtered_df = df[df[text_col].str.contains(selected_title, case=False, na=False)]
    if filtered_df.empty:
        filtered_df = df.copy()
else:
    filtered_df = df.copy()

# =====================================================
# STATIC HISTORICAL MARKET DATA (DEMO SAFE)
# =====================================================

# Historical trend simulation (Jan 2024 - Jan 2026)
dates = pd.date_range(start="2024-01-01", end="2026-01-01", freq="M")

base = {
    "Squid Game": np.linspace(55, 90, len(dates)),
    "Squid Game 2": np.linspace(40, 85, len(dates)),
    "Can This Love Be Translated": np.linspace(30, 70, len(dates))
}

trend_data = pd.DataFrame({
    selected_title: base[selected_title]
}, index=dates)

# Simulated state demand data
region_templates = {
    "Squid Game": {
        "Maharashtra": 100,
        "Karnataka": 88,
        "Delhi": 81,
        "Tamil Nadu": 76,
        "Uttar Pradesh": 70,
        "West Bengal": 63,
        "Gujarat": 60,
        "Kerala": 58
    },
    "Squid Game 2": {
        "Maharashtra": 92,
        "Karnataka": 86,
        "Delhi": 79,
        "Tamil Nadu": 74,
        "Uttar Pradesh": 68,
        "West Bengal": 62,
        "Gujarat": 59,
        "Telangana": 55
    },
    "Can This Love Be Translated": {
        "Tamil Nadu": 85,
        "Kerala": 80,
        "Karnataka": 76,
        "Delhi": 72,
        "Maharashtra": 68,
        "West Bengal": 60,
        "Telangana": 58,
        "Gujarat": 50
    }
}

region_data = pd.DataFrame(region_templates[selected_title].items(), columns=["State", "Interest"])
region_data = region_data.sort_values("Interest", ascending=False)

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
    fig = px.line(trend_data, y=selected_title)
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Current Demand Index",
              int(trend_data[selected_title].iloc[-1]))

    st.subheader("State-wise Demand Intensity")

    heatmap_fig = px.imshow(
        region_data[["Interest"]],
        labels=dict(color="Search Interest"),
        x=["Interest Score"],
        y=region_data["State"],
        color_continuous_scale="Reds",
        aspect="auto"
    )

    heatmap_fig.update_layout(height=500)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown("Top 5 High-Demand States")
    st.dataframe(region_data.head(5))

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

    sentiment_counts = filtered_df[sentiment_col].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(filtered_df, x=compound_col, nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

    risk_adjusted_score = positive_ratio - negative_ratio
    st.metric("Risk Adjusted Sentiment", round(risk_adjusted_score,3))

# =====================================================
# TAB 4 – Content Momentum Engine
# =====================================================

with tab4:

    demand_score = trend_data[selected_title].iloc[-1] / 100
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
        st.success("Recommendation: Strong Content Candidate")
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
# TAB 5 – MODEL VALIDATION
# =====================================================

with tab5:

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

        user_input = st.text_area("Simulate Audience Reaction")

        if st.button("Simulate Sentiment"):
            if user_input.strip():
                vec = vectorizer.transform([user_input])
                pred = model.predict(vec)
                label = le.inverse_transform(pred)[0]
                st.success(f"Predicted Sentiment: {label.upper()}")
            else:
                st.warning("Enter text to simulate.")

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
            textfont={"size":14}
        ))

        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            title="Confusion Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)
