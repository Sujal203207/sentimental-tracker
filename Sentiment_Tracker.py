def get_sample_data(scheme):
    # Provide static sample data for each scheme (expanded)
    sample = {
        "Ayushman Bharat": [
            {"Date": "2025-07-10", "Text": "Ayushman Bharat is a great initiative!", "Cleaned": "ayushman bharat is a great initiative"},
            {"Date": "2025-07-11", "Text": "Some hospitals are not accepting Ayushman Bharat cards.", "Cleaned": "some hospitals are not accepting ayushman bharat cards"},
            {"Date": "2025-07-12", "Text": "Ayushman Bharat helps many poor families.", "Cleaned": "ayushman bharat helps many poor families"},
            {"Date": "2025-07-13", "Text": "Ayushman Bharat coverage is increasing.", "Cleaned": "ayushman bharat coverage is increasing"}
        ],
        "Digital India": [
            {"Date": "2025-07-10", "Text": "Digital India is transforming the country.", "Cleaned": "digital india is transforming the country"},
            {"Date": "2025-07-11", "Text": "Internet speed is still an issue in rural areas.", "Cleaned": "internet speed is still an issue in rural areas"},
            {"Date": "2025-07-12", "Text": "Digital India campaign is impressive.", "Cleaned": "digital india campaign is impressive"},
            {"Date": "2025-07-13", "Text": "Digital India has made payments easier.", "Cleaned": "digital india has made payments easier"}
        ],
        "PM Awas Yojana": [
            {"Date": "2025-07-10", "Text": "PM Awas Yojana provides homes to the needy.", "Cleaned": "pm awas yojana provides homes to the needy"},
            {"Date": "2025-07-11", "Text": "Some people are still waiting for their houses.", "Cleaned": "some people are still waiting for their houses"},
            {"Date": "2025-07-12", "Text": "Good progress under PM Awas Yojana.", "Cleaned": "good progress under pm awas yojana"},
            {"Date": "2025-07-13", "Text": "PM Awas Yojana is a hope for many.", "Cleaned": "pm awas yojana is a hope for many"}
        ],
        "Startup India": [
            {"Date": "2025-07-10", "Text": "Startup India is boosting entrepreneurship.", "Cleaned": "startup india is boosting entrepreneurship"},
            {"Date": "2025-07-11", "Text": "Funding is still a challenge for startups.", "Cleaned": "funding is still a challenge for startups"},
            {"Date": "2025-07-12", "Text": "Startup India has created many jobs.", "Cleaned": "startup india has created many jobs"},
            {"Date": "2025-07-13", "Text": "Startup India is inspiring youth.", "Cleaned": "startup india is inspiring youth"}
        ],
        "Make in India": [
            {"Date": "2025-07-10", "Text": "Make in India is boosting manufacturing.", "Cleaned": "make in india is boosting manufacturing"},
            {"Date": "2025-07-11", "Text": "More jobs due to Make in India.", "Cleaned": "more jobs due to make in india"},
            {"Date": "2025-07-12", "Text": "Make in India needs more support.", "Cleaned": "make in india needs more support"}
        ],
        "Swachh Bharat": [
            {"Date": "2025-07-10", "Text": "Swachh Bharat has made cities cleaner.", "Cleaned": "swachh bharat has made cities cleaner"},
            {"Date": "2025-07-11", "Text": "Still need more awareness for Swachh Bharat.", "Cleaned": "still need more awareness for swachh bharat"},
            {"Date": "2025-07-12", "Text": "Swachh Bharat is a mass movement.", "Cleaned": "swachh bharat is a mass movement"}
        ],
        "Pradhan Mantri Jan Dhan Yojana": [
            {"Date": "2025-07-10", "Text": "Jan Dhan Yojana has increased financial inclusion.", "Cleaned": "jan dhan yojana has increased financial inclusion"},
            {"Date": "2025-07-11", "Text": "Zero balance accounts are helpful.", "Cleaned": "zero balance accounts are helpful"}
        ],
        "Pradhan Mantri Fasal Bima Yojana": [
            {"Date": "2025-07-10", "Text": "Fasal Bima Yojana secures farmers' crops.", "Cleaned": "fasal bima yojana secures farmers crops"},
            {"Date": "2025-07-11", "Text": "Insurance claim process should be faster.", "Cleaned": "insurance claim process should be faster"}
        ],
        "Ujjwala Yojana": [
            {"Date": "2025-07-10", "Text": "Ujjwala Yojana provides LPG to rural women.", "Cleaned": "ujjwala yojana provides lpg to rural women"},
            {"Date": "2025-07-11", "Text": "Refill cost is a concern for some families.", "Cleaned": "refill cost is a concern for some families"}
        ],
        "Atal Pension Yojana": [
            {"Date": "2025-07-10", "Text": "Atal Pension Yojana ensures old age security.", "Cleaned": "atal pension yojana ensures old age security"}
        ],
        "Beti Bachao Beti Padhao": [
            {"Date": "2025-07-10", "Text": "Beti Bachao Beti Padhao is changing mindsets.", "Cleaned": "beti bachao beti padhao is changing mindsets"}
        ],
        "Skill India Mission": [
            {"Date": "2025-07-10", "Text": "Skill India Mission is empowering youth.", "Cleaned": "skill india mission is empowering youth"}
        ],
        "National Rural Livelihood Mission": [
            {"Date": "2025-07-10", "Text": "NRLM is supporting rural women entrepreneurs.", "Cleaned": "nrlm is supporting rural women entrepreneurs"}
        ],
        "Saubhagya Yojana": [
            {"Date": "2025-07-10", "Text": "Saubhagya Yojana brings electricity to villages.", "Cleaned": "saubhagya yojana brings electricity to villages"}
        ],
        "Jal Jeevan Mission": [
            {"Date": "2025-07-10", "Text": "Jal Jeevan Mission provides tap water to homes.", "Cleaned": "jal jeevan mission provides tap water to homes"}
        ]
    }
    df = pd.DataFrame(sample.get(scheme, []))
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        # Use VADER for sentiment
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        df['Polarity'] = df['Cleaned'].apply(lambda x: sid.polarity_scores(x)['compound'])
        df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0.05 else ('Negative' if p < -0.05 else 'Neutral'))
    return df

# sentiment_tracker.py
# Install required packages first:
# pip install tweepy pandas matplotlib seaborn textblob streamlit wordcloud
# pip install pytrends nltk

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from pytrends.request import TrendReq
import tweepy
import re
import datetime

# Download NLTK data only if not present
import os
import nltk.data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Setup Twitter API v2 with Bearer Token
client = tweepy.Client(bearer_token="YcELWXxg%3DU5GPgWK9gvFhn5QMTgMDx2HF5vD78dkzNpJrAbc3YFaz1FiQ5k", wait_on_rate_limit=True)

# Streamlit UI
st.set_page_config(page_title="Sentiment Tracker for Indian Government Schemes", layout="wide")
st.title("🇮🇳 Sentiment Tracker for Indian Government Schemes")

all_schemes = [
    "Ayushman Bharat", "Digital India", "PM Awas Yojana", "Startup India",
    "Make in India", "Swachh Bharat", "Pradhan Mantri Jan Dhan Yojana",
    "Pradhan Mantri Fasal Bima Yojana", "Ujjwala Yojana", "Atal Pension Yojana",
    "Beti Bachao Beti Padhao", "Skill India Mission", "National Rural Livelihood Mission",
    "Saubhagya Yojana", "Jal Jeevan Mission"
]

# --- Sidebar with logo and design ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png", width=120)
st.sidebar.markdown("<h2 style='color:#0a5c2c;'>Govt. of India</h2>", unsafe_allow_html=True)
st.sidebar.title("About")
st.sidebar.info("""
This dashboard tracks public sentiment for major Indian government schemes using live Twitter data and Google Trends. Powered by VADER sentiment analysis for high accuracy.

Created by Sujal | Data Analyst & Visualisation Internship Project
""")

# --- Main UI ---
st.sidebar.markdown("---")
use_sample = st.sidebar.checkbox("Use Sample Data (Fast, Reliable Demo)", value=True)

col1, col2 = st.columns([2,1])
with col1:
    scheme = st.selectbox("Choose a Scheme", all_schemes)
with col2:
    tweet_limit = st.slider("Number of Tweets to Analyze", 10, 100, value=30, step=10)
    if tweet_limit > 50 and not use_sample:
        st.warning("Fetching more than 50 tweets may take longer and can hit Twitter API rate limits.")

# --- Date filter ---
this_year = datetime.datetime.now().year
years = list(range(this_year, this_year-5, -1))
months = ["All"] + [datetime.date(1900, m, 1).strftime('%B') for m in range(1,13)]
col3, col4 = st.columns(2)
with col3:
    year_filter = st.selectbox("Year", years)
with col4:
    month_filter = st.selectbox("Month", months)

@st.cache_data
def clean_tweet(text):
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # Remove mentions
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # Remove hashtags
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z ]", "", text)  # Remove special chars
    text = text.lower().strip()
    return text

@st.cache_data(ttl=300)
def fetch_tweets_v2(query, limit, year=None, month=None):
    tweets_data = []
    # Quote the query if it contains spaces (for multi-word schemes)
    if ' ' in query:
        search_query = f'"{query}"'
    else:
        search_query = query
    next_token = None
    fetched = 0
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    progress = st.progress(0, text="Fetching tweets...")
    while fetched < limit:
        try:
            response = client.search_recent_tweets(query=search_query, max_results=min(100, limit-fetched), tweet_fields=['created_at', 'text', 'author_id'], expansions=['author_id'], next_token=next_token)
        except Exception as e:
            st.error(f"Twitter API error: {e}")
            break
        if response.data is None:
            break
        users = {u['id']: u['username'] for u in response.includes['users']} if response.includes and 'users' in response.includes else {}
        for tweet in response.data:
            tweet_date = tweet.created_at
            if year and tweet_date.year != year:
                continue
            if month and month != "All" and tweet_date.strftime('%B') != month:
                continue
            clean = clean_tweet(tweet.text)
            polarity = sid.polarity_scores(clean)['compound']
            if polarity > 0.05:
                sentiment = "Positive"
            elif polarity < -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            tweets_data.append({
                "Date": tweet_date,
                "User": users.get(tweet.author_id, "Sujal_nagar"),
                "Text": tweet.text,
                "Cleaned": clean,
                "Polarity": polarity,
                "Sentiment": sentiment
            })
            fetched += 1
            progress.progress(min(fetched/limit, 1.0), text=f"Fetched {fetched} tweets...")
            if fetched >= limit:
                break
        next_token = response.meta.get('next_token') if hasattr(response, 'meta') and 'next_token' in response.meta else None
        if not next_token:
            break
    progress.empty()
    return pd.DataFrame(tweets_data)

# --- Tabs ---
tabs = st.tabs([
    "All Tweets", "Bar Graph", "Word Cloud", "Line Graph", "Trends & Download", "About"
])

if st.button("🔍 Analyze Now"):
    with st.spinner("Fetching and analyzing data..."):
        if use_sample:
            df = get_sample_data(scheme)
            df['User'] = 'SampleUser'
        else:
            df = fetch_tweets_v2(scheme, tweet_limit, year=year_filter, month=month_filter)
            if df.empty:
                st.info("No tweets found for this scheme. Showing sample data instead.")
                df = get_sample_data(scheme)
                df['User'] = 'SampleUser'
    if df.empty:
        st.warning("No data available for this scheme.")
    else:
        # --- Tab 1: All Tweets ---
        with tabs[0]:
            st.subheader(f"All Tweets for {scheme}")
            st.dataframe(df[['Date','User','Text','Sentiment']])
            st.markdown(f"**Total Tweets:** {len(df)}")
        # --- Tab 2: Bar Graph ---
        with tabs[1]:
            st.subheader("📈 Sentiment Distribution")
            sns.set_style("whitegrid")
            fig, ax = plt.subplots()
            sns.countplot(x="Sentiment", data=df, palette="Set2", ax=ax)
            st.pyplot(fig)
        # --- Tab 3: Word Cloud ---
        with tabs[2]:
            st.subheader("☁️ Word Cloud")
            words = ' '.join(df['Cleaned'])
            wc = WordCloud(width=800, height=300, background_color='white').generate(words)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)
        # --- Tab 4: Line Graph ---
        with tabs[3]:
            st.subheader("📊 Sentiment Over Time")
            df['Date'] = pd.to_datetime(df['Date'])
            df['Day'] = df['Date'].dt.date
            timeline = df.groupby(['Day', 'Sentiment']).size().unstack().fillna(0)
            st.line_chart(timeline)
        # --- Tab 5: Trends & Download ---
        with tabs[4]:
            st.subheader("🔍 Google Trends Popularity")
            pytrends = TrendReq()
            pytrends.build_payload([scheme], timeframe='today 3-m')
            trends_df = pytrends.interest_over_time()
            if not trends_df.empty:
                st.line_chart(trends_df[scheme])
            else:
                st.info("No trend data found for this topic.")
            st.subheader("⬇️ Download Data")
            st.download_button("Download as CSV", df.to_csv(index=False), file_name=f"{scheme}_sentiment_data.csv", mime="text/csv")
        # --- Tab 6: About ---
        with tabs[5]:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png", width=120)
            st.markdown("""
            <h2 style='color:#0a5c2c;'>About This Dashboard</h2>
            <p>This dashboard is created to analyze and visualize public sentiment for major Indian government schemes using real-time Twitter data and Google Trends. It is designed for policy makers, researchers, and the public to understand the impact and perception of government initiatives in both rural and urban India.</p>
            <ul>
            <li><b>Data Source:</b> Twitter API (recent tweets, up to 7 days)</li>
            <li><b>Sentiment Analysis:</b> VADER (best for social media)</li>
            <li><b>Visualization:</b> Streamlit, Seaborn, Matplotlib, WordCloud</li>
            </ul>
            <p><b>Created by Sujal</b> | Data Analyst & Visualisation Internship Project</p>
            <p><b>Contact:</b> <a href='mailto:sujalsingh0373@gmail.com'>sujalsingh0373@gmail.com</a></p>
            <p><b>For Government of India</b></p>
            """, unsafe_allow_html=True)
