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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Demand",
    "Audience Perception",
    "Risk Analysis",
    "Acquisition Engine",
    "Sentiment Prediction",
    "Signal Reliability"
])


# =====================================================
# TAB 1 – MARKET SIGNALS
# =====================================================

with tab1:

    st.subheader("Interest Over Time – India")

    if not trend_data.empty:
        fig1 = px.line(trend_data, y=selected_title)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Geographic Demand Heatmap – India")

    if not region_data.empty:

        region_df = region_data.reset_index()
        region_df.columns = ["State", "Interest"]
        region_df = region_df[region_df["Interest"] > 0]
        region_df["State"] = region_df["State"].str.strip()

        # Handle state code case
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

        # Sort by interest
        region_df = region_df.sort_values("Interest", ascending=False)

        # Create Heatmap
        heatmap_fig = px.imshow(
            region_df[["Interest"]],
            labels=dict(color="Search Interest"),
            x=["Interest Score"],
            y=region_df["State"],
            color_continuous_scale="Reds",
            aspect="auto"
        )

        heatmap_fig.update_layout(
            height=600,
            xaxis_title="",
            yaxis_title="State"
        )

        st.plotly_chart(heatmap_fig, use_container_width=True)

        # Add executive insight
        st.markdown("### Top 5 High-Demand States")
        st.dataframe(region_df.head(5))

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

# ======================================================
# TAB 5 – LIVE SENTIMENT PREDICTION ENGINE
# ======================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

with tab5:

    st.subheader("AI Sentiment Classification Engine")
    st.markdown("Test real-time sentiment prediction using trained model.")

    # ============================
    # Detect required columns
    # ============================

    text_col = None
    for col in df.columns:
        if "text" in col.lower():
            text_col = col
            break

    sentiment_col = None
    for col in df.columns:
        if "sentiment" in col.lower():
            sentiment_col = col
            break

    if text_col is None or sentiment_col is None:
        st.error("Required text or sentiment column not found in dataset.")
        st.stop()

    # ============================
    # Train Model Once (Cached)
    # ============================

    @st.cache_resource
    def train_model(data):
        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        X = vectorizer.fit_transform(data[text_col])

        le = LabelEncoder()
        y = le.fit_transform(data[sentiment_col])

        model = LinearSVC()
        model.fit(X, y)

        return vectorizer, model, le

    vectorizer, model, le = train_model(df)

    # ============================
    # User Input
    # ============================

    user_input = st.text_area("Enter viewer comment to classify:")

    if st.button("Predict Sentiment"):

        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)
            predicted_label = le.inverse_transform(prediction)[0]

            st.success(f"Predicted Sentiment: {predicted_label.upper()}")

            # Confidence Score (Distance from hyperplane)
            decision_scores = model.decision_function(input_vector)

            if len(le.classes_) == 2:
                confidence = abs(decision_scores[0]) / np.max(abs(decision_scores))
            else:
                confidence = np.max(decision_scores) / np.sum(abs(decision_scores))

            st.metric("Prediction Confidence Score", round(float(confidence), 3))

            # Display probability-style indicator
            st.progress(float(min(max(confidence, 0), 1)))

# ======================================================
# TAB 6 – MODEL INTELLIGENCE (SVM ENGINE DIAGNOSTICS)
# ======================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

with tab6:

    st.subheader("SVM Sentiment Model Diagnostics")

    # ==============================
    # Detect required columns safely
    # ==============================

    text_col = None
    for col in df.columns:
        if "text" in col.lower():
            text_col = col
            break

    sentiment_col = None
    for col in df.columns:
        if "sentiment" in col.lower():
            sentiment_col = col
            break

    if text_col is None or sentiment_col is None:
        st.error("Required text or sentiment column not found.")
        st.stop()

    # ==============================
    # Train-Test Split
    # ==============================

    X_text = df[text_col]
    y_labels = df[sentiment_col]

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vectorizer.fit_transform(X_text)

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ==============================
    # Accuracy
    # ==============================

    accuracy = (y_pred == y_test).mean()
    st.metric("Model Accuracy", round(float(accuracy), 3))

    # ==============================
    # Classification Report
    # ==============================

    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Performance")
    st.dataframe(report_df)

    # ==============================
    # Confusion Matrix
    # ==============================

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.subheader("Confusion Matrix")
    st.pyplot(fig)

    # ==============================
    # Model-Based Sentiment Distribution
    # ==============================

    full_predictions = model.predict(X)
    predicted_labels = le.inverse_transform(full_predictions)

    pred_df = pd.Series(predicted_labels).value_counts().reset_index()
    pred_df.columns = ["Sentiment", "Count"]

    fig2 = px.bar(
        pred_df,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title="SVM-Based Sentiment Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    This module demonstrates how sentiment predictions are generated 
    and evaluated before being incorporated into acquisition analytics.
    """)




