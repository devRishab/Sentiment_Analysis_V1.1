import os
import json
import feedparser
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import warnings
from pathlib import Path
import streamlit as st

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App Configuration & Styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide", page_icon="📊")

st.title("🦅 Live Market Intelligence & Sector Sentiment Analyzer")
st.markdown("Query the Nifty Total Market Universe to extract structural peers and analyze real-time news sentiment via **FinBERT**.")

# ─────────────────────────────────────────────────────────────────────────────
# Optimized Cached Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_universe_streamlit(csv_file) -> pd.DataFrame:
    """Loads and sanitizes the Nifty Stock Universe dataframe."""
    df = pd.read_csv(csv_file)
    df["MarketCap_INR"] = pd.to_numeric(df["MarketCap_INR"], errors="coerce")
    df["MarketCap_Cr"]  = pd.to_numeric(df["MarketCap_Cr"],  errors="coerce")
    return df

@st.cache_resource
def load_sentiment_pipeline():
    """Loads the FinBERT model once and keeps it cached in memory."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ─────────────────────────────────────────────────────────────────────────────
# Universe File Resolution (Local Root vs. User Upload Fallback)
# ─────────────────────────────────────────────────────────────────────────────
csv_filename = "nifty_total_market_universe.csv"
u = None

if Path(csv_filename).exists():
    u = load_universe_streamlit(csv_filename)
else:
    st.warning(f"⚠️ `{csv_filename}` was not found in your repository root folder.")
    uploaded_file = st.file_uploader("Please upload your Universe CSV file to proceed:", type=["csv"])
    if uploaded_file is not None:
        u = load_universe_streamlit(uploaded_file)

# Stop execution if data universe isn't available yet
if u is not None:
    
    # Sidebar Configuration Options
    st.sidebar.header("🔧 Analysis Parameters")
    top_n_peers = st.sidebar.slider("Number of Peers to Evaluate", min_value=2, max_value=10, value=5)
    lookback_days = st.sidebar.slider("News Lookback Window (Days)", min_value=5, max_value=30, value=15)
    confidence_threshold = st.sidebar.slider("FinBERT Confidence Filter", min_value=0.50, max_value=0.95, value=0.80, step=0.05)

    # ─────────────────────────────────────────────────────────────────────────────
    # Core Notebook Logic Wrapper Functions
    # ─────────────────────────────────────────────────────────────────────────────
    def search_by_name(query: str, universe: pd.DataFrame, top_n: int = 1) -> pd.DataFrame:
        mask = universe["CompanyName"].str.contains(query, case=False, na=False) | \
               universe["Symbol"].str.lower().eq(query.lower().strip())
        return universe[mask].head(top_n)

    def find_peers(symbol: str, universe: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        symbol = symbol.upper().strip()
        row = universe[universe["Symbol"] == symbol]
        if row.empty:
            return pd.DataFrame()

        r = row.iloc[0]
        target_industry = r["Industry"]
        target_capsize = r["CapSize"]

        # Echo Target Profile parameters to UI
        st.write(f"🎯 **Target Selected:** {r['CompanyName']} (`{symbol}`)")
        st.caption(f"**Industry Group:** {target_industry} | **Classification:** {target_capsize} | **Market Cap:** ₹{r['MarketCap_Cr']:,.2f} Cr")

        peers = (
            universe[
                (universe["Industry"] == target_industry) &
                (universe["CapSize"] == target_capsize)
            ]
            .head(top_n)
            [["Symbol", "CompanyName", "MarketCap_Cr", "CapSize", "Industry"]]
            .reset_index(drop=True)
        )
        return peers

    # ─────────────────────────────────────────────────────────────────────────────
    # User Input & Execution Action Run
    # ─────────────────────────────────────────────────────────────────────────────
    target_stock_name = st.text_input("🔍 Search Stock (Type Full Name or NSE Symbol):", value="Zen Technologies Limited")

    if st.button("Run Quantitative Sentiment Analysis", type="primary"):
        res = search_by_name(target_stock_name, u, top_n=1)

        if not res.empty:
            symbol = res['Symbol'].iloc[0]
            res1 = find_peers(symbol, u, top_n=top_n_peers)
            
            # Map notebook variables cleanly to semantic keys
            peers = res1[['CompanyName', 'Symbol']].rename(
                columns={'CompanyName': 'name', 'Symbol': 'ticker'}
            ).to_dict('records')

            with st.spinner("Initializing FinBERT Deep Learning Weights..."):
                analyzer = load_sentiment_pipeline()

            results_summary = []
            all_raw_headlines = {}

            # Processing loop status UI tracker
            status_container = st.status("Gathering news artifacts and crunching sentiment...", expanded=True)
            
            for company in peers:
                name = company["name"]
                ticker = company["ticker"]

                status_container.write(f"Scraping & scoring headlines for **{name}**...")
                
                search_term = f"{name} {ticker}".strip()
                query = urllib.parse.quote(f"{search_term} stock when:{lookback_days}d")
                rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

                feed = feedparser.parse(rss_url)
                entries = feed.entries[:20]

                bullish_wt, bearish_wt, neutral_wt = 0, 0, 0
                results = []

                for entry in entries:
                    title = entry.title
                    pub_date = getattr(entry, 'published', 'N/A')

                    sentiment = analyzer(title)[0]
                    label = sentiment['label']
                    score = sentiment['score']

                    if score < confidence_threshold:
                        continue

                    if label == 'positive':
                        bullish_wt += score
                    elif label == 'negative':
                        bearish_wt += score
                    elif label == 'neutral':
                        neutral_wt += score

                    results.append({
                        "Date": pub_date,
                        "Headline": title,
                        "Sentiment": label.upper(),
                        "Confidence": round(score, 3)
                    })

                all_raw_headlines[name] = pd.DataFrame(results)

                total = bullish_wt + bearish_wt + neutral_wt
                bull_pct = (bullish_wt / total * 100) if total > 0 else 0
                bear_pct = (bearish_wt / total * 100) if total > 0 else 0
                neut_pct = (neutral_wt / total * 100) if total > 0 else 0

                results_summary.append({
                    "Company": name,
                    "Bullish (%)": round(bull_pct, 1),
                    "Neutral (%)": round(neut_pct, 1),
                    "Bearish (%)": round(bear_pct, 1)
                })

            df_peers = pd.DataFrame(results_summary)
            status_container.update(label="Analysis Finished Successfully!", state="complete", expanded=False)

            # ── Main Dashboard Visualization Rendering ───────────────────────────
            st.subheader("📊 Cross-Peer Sentiment Distribution Matrix")
            
            col_matrix, col_spacer = st.columns([2, 1])
            with col_matrix:
                st.dataframe(df_peers, use_container_width=True, hide_index=True)

            # Generate Matplotlib chart object inside Streamlit framework safely
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(df_peers["Company"]))
            width = 0.25

            bars_bull = ax.bar(x - width, df_peers["Bullish (%)"], width, label='Bullish', color='#2ca02c', edgecolor='black', zorder=3)
            bars_neut = ax.bar(x, df_peers["Neutral (%)"], width, label='Neutral', color='#ff7f0e', edgecolor='black', zorder=3)
            bars_bear = ax.bar(x + width, df_peers["Bearish (%)"], width, label='Bearish', color='#d62728', edgecolor='black', zorder=3)

            for bars in (bars_bull, bars_neut, bars_bear):
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

            for i, row in df_peers.iterrows():
                scores = {"Bullish": row["Bullish (%)"], "Neutral": row["Neutral (%)"], "Bearish": row["Bearish (%)"]}
                dominant = max(scores, key=scores.get)
                colour = {'Bullish': '#2ca02c', 'Neutral': '#ff7f0e', 'Bearish': '#d62728'}[dominant]
                ax.text(x[i], -5, f"▲ {dominant}" if dominant == "Bullish" else f"● {dominant}" if dominant == "Neutral" else f"▼ {dominant}", ha='center', va='top', fontsize=8, fontweight='bold', color=colour)

            ax.set_ylabel('Sentiment Score (%)', fontweight='bold')
            ax.set_title(f'Peer Group Sentiment Comparison\n(Confidence ≥ {confidence_threshold} · Last {lookback_days} Days)', fontsize=13, fontweight='bold', pad=14)
            ax.set_xticks(x)
            ax.set_xticklabels(df_peers["Company"], rotation=15, ha='right', fontsize=9)
            ax.set_ylim(0, max(df_peers[["Bullish (%)", "Neutral (%)", "Bearish (%)"]].max().max() + 15, 30))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
            plt.tight_layout()
            
            st.pyplot(fig)

            # ── Drill-down Data Dropdowns ─────────────────────────────────────────
            st.subheader("📰 Granular Pipeline Audits (Raw Scraped Headlines)")
            for comp_name, df_headlines in all_raw_headlines.items():
                with st.expander(f"View Scraped Headlines for {comp_name}"):
                    if not df_headlines.empty:
                        st.dataframe(df_headlines, use_container_width=True, hide_index=True)
                    else:
                        st.write("No headlines cleared the current confidence threshold filters.")
        else:
            st.error(f"❌ '{target_stock_name}' did not match any listings inside your asset universe file.")
