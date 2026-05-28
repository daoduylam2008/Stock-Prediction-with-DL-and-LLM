import warnings
from datetime import datetime
from typing import List, Dict

warnings.filterwarnings("ignore")

import feedparser
import yfinance as yf

from RAG.mock_data import get_mock_fundamentals, get_mock_prices, get_mock_articles


RSS_FEEDS = [
    "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://finance.yahoo.com/news/rssindex",
]


def fetch_fundamentals(ticker: str) -> Dict:
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None:
            raise ValueError("empty info")
        hist = yf.Ticker(ticker).history(period="1y")
        hi52 = float(hist["High"].max()) if not hist.empty else 0.0
        lo52 = float(hist["Low"].min())  if not hist.empty else 0.0
        divs = yf.Ticker(ticker).dividends
        latest_div = float(divs.iloc[-1]) if not divs.empty else 0.0
        return {
            "market_cap_B":   round((info.get("marketCap") or 0) / 1e9, 2),
            "pe_ratio":       round(info.get("trailingPE") or 0.0, 2),
            "dividend_yield": round((info.get("dividendYield") or 0) * 100, 4),
            "52w_high":       round(info.get("fiftyTwoWeekHigh") or hi52, 2),
            "52w_low":        round(info.get("fiftyTwoWeekLow")  or lo52, 2),
            "latest_div":     round(latest_div, 4),
            "current_price":  round(info.get("currentPrice") or info.get("regularMarketPrice") or 0, 2),
            "volume":         info.get("volume") or 0,
            "beta":           round(info.get("beta") or 1.0, 3),
            "name":           info.get("longName") or ticker,
            "_source": "live",
        }
    except Exception as e:
        print(f"  [warn] Live fundamentals failed ({e}), using mock data.")
        return {**get_mock_fundamentals(ticker), "_source": "mock"}


def fetch_prices(ticker: str, days: int = 60) -> List[Dict]:
    try:
        hist = yf.Ticker(ticker).history(period=f"{days}d")
        if hist.empty:
            raise ValueError("empty history")
        recs = []
        for dt, row in hist.iterrows():
            recs.append({
                "date":  dt.strftime("%Y-%m-%d"),
                "open":  round(row["Open"],  2),
                "close": round(row["Close"], 2),
                "high":  round(row["High"],  2),
                "low":   round(row["Low"],   2),
                "volume": int(row["Volume"]),
            })
        return recs
    except Exception as e:
        print(f"  [warn] Live prices failed ({e}), using mock data.")
        return get_mock_prices(ticker)



def fetch_news(ticker: str, company: str = "", max_n: int = 20) -> List[Dict]:
    keywords = {ticker.lower()}
    if company:
        keywords.add(company.lower().split()[0])   # first word of company name
    articles: List[Dict] = []
    seen: set = set()

    for url_tpl in RSS_FEEDS:
        url = url_tpl.format(ticker=ticker)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title   = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link    = entry.get("link", "")
                pub     = entry.get("published", datetime.utcnow().isoformat())
                if not any(kw in (title + summary).lower() for kw in keywords):
                    continue
                key = title[:60]
                if key in seen:
                    continue
                seen.add(key)
                articles.append({
                    "headline": title,
                    "summary":  summary[:400],
                    "link": link,
                    "published": pub,
                    "source": url.split("/")[2],
                })
                if len(articles) >= max_n:
                    break
        except Exception:
            continue
        if len(articles) >= max_n:
            break

    if not articles:
        print("  [warn] No live news found, using mock data.")
        articles = get_mock_articles(ticker)
    return articles