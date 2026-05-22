import pandas as pd
import streamlit as st
import requests
import nltk
import plotly.graph_objects as go

from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="AI Reddit Sentiment Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ===================================
# CUSTOM DARK THEME
# ===================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background-color: #1F2937;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

/* Buttons */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}

/* Text Input */
.stTextInput>div>div>input {
    background-color: #1F2937;
    color: white;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #1F2937;
}

/* Headings */
h1, h2, h3 {
    color: white;
}

/* ── NEW: Color-coded metric cards ── */
div[data-testid="metric-container"]:nth-child(1) {
    border-left: 4px solid #22c55e !important;
}
div[data-testid="metric-container"]:nth-child(2) {
    border-left: 4px solid #ef4444 !important;
}
div[data-testid="metric-container"]:nth-child(3) {
    border-left: 4px solid #94a3b8 !important;
}

</style>
""", unsafe_allow_html=True)

# ===================================
# SESSION STATE (for sidebar buttons)
# ===================================

if "subreddit_input" not in st.session_state:
    st.session_state.subreddit_input = ""

# ===================================
# SIDEBAR
# ===================================

st.sidebar.title("🚀 Dashboard Controls")

st.sidebar.info(
    "Analyze Reddit discussions using AI-powered NLP sentiment analysis."
)

st.sidebar.subheader("🔥 Popular Subreddits")

# ── NEW: Clickable subreddit buttons that auto-fill the input ──
popular_subreddits = [
    "python", "technology", "artificial",
    "programming", "worldnews", "gaming",
    "machinelearning", "OpenAI"
]

for sub in popular_subreddits:
    if st.sidebar.button(f"r/{sub}", key=f"btn_{sub}"):
        st.session_state.subreddit_input = sub

# ===================================
# DOWNLOAD NLP MODEL
# ===================================

nltk.download('vader_lexicon', quiet=True)

# ===================================
# INITIALIZE SENTIMENT ANALYZER
# ===================================

sia = SentimentIntensityAnalyzer()

# ===================================
# TITLE
# ===================================

st.title("🚀 AI-Powered Reddit Sentiment Dashboard")

st.markdown("""
Fetches live Reddit discussions · Performs AI-based sentiment analysis · Generates visual insights
""")

st.divider()

# ===================================
# USER INPUTS
# ===================================

col_input1, col_input2 = st.columns([3, 1])

with col_input1:
    # ── NEW: Reads from session_state so sidebar buttons auto-fill this ──
    subreddit = st.text_input(
        "Enter Subreddit Name",
        placeholder="Example: python",
        value=st.session_state.subreddit_input
    )

with col_input2:
    post_limit = st.slider(
        "Posts",
        min_value=5,
        max_value=100,
        value=10
    )

# ===================================
# BUTTON
# ===================================

if st.button("🚀 Analyze Sentiment"):

    # ── NEW: Input validation ──
    if not subreddit.strip():
        st.warning("⚠️ Please enter a subreddit name before analyzing.")
        st.stop()

    # Reddit URL
    url = f"https://www.reddit.com/r/{subreddit}.json?limit={post_limit}"

    # Headers
    headers = {
        "User-Agent": "Mozilla/5.0"

    }

    # Loading Spinner
    with st.spinner("Fetching Reddit posts and analyzing sentiment..."):

        response = requests.get(url, headers=headers)

        # ===================================
        # SUCCESS CHECK
        # ===================================

        if response.status_code == 200:

            positive = 0
            negative = 0
            neutral = 0

            most_positive_post = ""
            most_positive_score = -1

            most_negative_post = ""
            most_negative_score = 1

            data = response.json()

            posts = data["data"]["children"]

            results = []

            all_titles = ""

            # ===================================
            # PROCESS POSTS  (logic unchanged)
            # ===================================

            for post in posts:

                title = post["data"]["title"]

                sentiment_score = sia.polarity_scores(title)

                compound = sentiment_score['compound']

                # Track most positive
                if compound > most_positive_score:
                    most_positive_score = compound
                    most_positive_post = title

                # Track most negative
                if compound < most_negative_score:
                    most_negative_score = compound
                    most_negative_post = title

                # Sentiment Classification
                if compound >= 0.05:
                    sentiment = "Positive 😊"
                    positive += 1

                elif compound <= -0.05:
                    sentiment = "Negative 😡"
                    negative += 1

                else:
                    sentiment = "Neutral 😐"
                    neutral += 1

                # Save Results
                results.append([title, sentiment])

                # Word Cloud Text
                all_titles += title + " "

            # ── NEW: Toast notification on success ──
            st.toast(f"✅ Fetched {len(posts)} posts from r/{subreddit}!", icon="🎉")

            # ===================================
            # TABS
            # ===================================

            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Dashboard",
                "📈 Charts",
                "📋 Data",
                "☁️ Word Cloud"
            ])

            # ===================================
            # TAB 1 - DASHBOARD  (logic unchanged)
            # ===================================

            with tab1:

                st.subheader("📊 Sentiment Overview")

                col1, col2, col3 = st.columns(3)

                col1.metric("Positive 😊", positive)
                col2.metric("Negative 😡", negative)
                col3.metric("Neutral 😐", neutral)

                st.divider()

                # ===================================
                # AI SUMMARY
                # ===================================

                st.subheader("🧠 AI Summary")

                total_posts = positive + negative + neutral

                positive_percent = (positive / total_posts) * 100
                negative_percent = (negative / total_posts) * 100
                neutral_percent = (neutral / total_posts) * 100

                if positive > negative and positive > neutral:
                    overall_sentiment = "largely POSITIVE 😊"

                elif negative > positive and negative > neutral:
                    overall_sentiment = "largely NEGATIVE 😡"

                else:
                    overall_sentiment = "mostly NEUTRAL 😐"

                st.info(
                    f"""
                    Overall Reddit sentiment for r/{subreddit} is {overall_sentiment}.

                    Positive discussions: {positive_percent:.1f}%

                    Negative discussions: {negative_percent:.1f}%

                    Neutral discussions: {neutral_percent:.1f}%

                    The dashboard analyzed {total_posts} Reddit posts in real time using NLP sentiment analysis.
                    """
                )

                st.divider()

                # ===================================
                # TOP TRENDING WORDS  (logic unchanged)
                # ===================================

                st.subheader("🔥 Top Trending Words")

                words = all_titles.lower().split()

                stop_words = [
                    "the", "a", "and", "is", "to", "of",
                    "in", "for", "on", "with", "this",
                    "that", "it", "you", "i", "my",
                    "we", "our", "from", "at", "be",
                    "are", "was", "were"
                ]

                filtered_words = []

                for word in words:

                    clean_word = word.strip(".,!?()[]{}:;\"'")

                    if clean_word not in stop_words and len(clean_word) > 3:
                        filtered_words.append(clean_word)

                word_freq = {}

                for word in filtered_words:

                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1

                sorted_words = sorted(
                    word_freq.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                top_words = sorted_words[:10]

                trend_df = pd.DataFrame(
                    top_words,
                    columns=["Word", "Frequency"]
                )

                st.dataframe(
                    trend_df,
                    use_container_width=True
                )

                st.divider()

                # ===================================
                # TOP POSTS
                # ===================================

                col_pos, col_neg = st.columns(2)

                with col_pos:
                    st.subheader("🔥 Most Positive Post")
                    st.success(most_positive_post)

                with col_neg:
                    st.subheader("😡 Most Negative Post")
                    st.error(most_negative_post)

            # ===================================
            # TAB 2 - CHARTS
            # ── NEW: Replaced matplotlib with Plotly for interactivity ──
            # ===================================

            with tab2:

                st.subheader("📈 Sentiment Visualizations")

                chart_col1, chart_col2 = st.columns(2)

                labels = ['Positive', 'Negative', 'Neutral']
                sizes  = [positive, negative, neutral]
                colors = ['#22c55e', '#ef4444', '#94a3b8']

                # DONUT CHART (replaces pie)
                with chart_col1:

                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=sizes,
                        hole=0.45,
                        marker=dict(colors=colors),
                        textfont=dict(color='white')
                    )])

                    fig_donut.update_layout(
                        title="Sentiment Distribution",
                        paper_bgcolor='#1F2937',
                        plot_bgcolor='#1F2937',
                        font=dict(color='white'),
                        legend=dict(font=dict(color='white')),
                        margin=dict(t=40, b=20, l=20, r=20)
                    )

                    st.plotly_chart(fig_donut, use_container_width=True)

                # BAR CHART
                with chart_col2:

                    fig_bar = go.Figure(data=[go.Bar(
                        x=labels,
                        y=sizes,
                        marker_color=colors,
                        text=sizes,
                        textposition='outside',
                        textfont=dict(color='white')
                    )])

                    fig_bar.update_layout(
                        title="Sentiment Comparison",
                        paper_bgcolor='#1F2937',
                        plot_bgcolor='#1F2937',
                        font=dict(color='white'),
                        yaxis=dict(
                            title="Number of Posts",
                            gridcolor='#374151',
                            color='white'
                        ),
                        xaxis=dict(color='white'),
                        margin=dict(t=40, b=20, l=20, r=20)
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)

            # ===================================
            # TAB 3 - DATA  (logic unchanged)
            # ===================================

            with tab3:

                st.subheader("📋 Detailed Results")

                df = pd.DataFrame(
                    results,
                    columns=["Post Title", "Sentiment"]
                )

                sentiment_filter = st.selectbox(
                    "Filter by Sentiment",
                    ["All", "Positive 😊", "Negative 😡", "Neutral 😐"]
                )

                if sentiment_filter != "All":
                    filtered_df = df[df["Sentiment"] == sentiment_filter]
                else:
                    filtered_df = df

                st.dataframe(
                    filtered_df,
                    use_container_width=True
                )

                csv = filtered_df.to_csv(index=False)

                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv,
                    file_name="reddit_sentiment_results.csv",
                    mime="text/csv"
                )

            # ===================================
            # TAB 4 - WORD CLOUD
            # ── NEW: Background matches app theme (#0E1117) ──
            # ===================================

            with tab4:

                st.subheader("☁️ Trending Topics Word Cloud")

                wordcloud = WordCloud(
                    width=1000,
                    height=500,
                    background_color='#0E1117',
                    colormap='Blues'
                ).generate(all_titles)

                fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                fig_wc.patch.set_facecolor('#0E1117')

                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis("off")

                st.pyplot(fig_wc)

        else:

            st.error(f"❌ Could not fetch r/{subreddit} — check the subreddit name and try again. (Status: {response.status_code})")