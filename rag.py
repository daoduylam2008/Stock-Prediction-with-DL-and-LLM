import math, warnings
from datetime import datetime, timedelta, timezone
from typing import Dict

warnings.filterwarnings("ignore")

from RAG import analysis, data_loader, vectordb


def query(ticker: str, days: int = 60) -> Dict:
    ticker = ticker.upper().strip()

    # Fecthing fundamentals
    fund = data_loader.fetch_fundamentals(ticker)
    company = fund.get("name", ticker)

    # Fecthing price history
    prices = data_loader.fetch_prices(ticker, days)

    # Fetching news
    articles = data_loader.fetch_news(ticker, company, max_n=20)
    
    # Building vector store
    print(f"({len(articles)} articles, {len(prices)} price bars)…")
    vs = vectordb.build_vectorstore(articles, prices, fund, ticker)

    # RAG sub-questions
    q1 = f"What are the latest news events impacting {ticker} stock price?"
    q2 = f"What do financial metrics and price trend indicate about {ticker}?"
    ev1 = vectordb.query(vs, q1)
    ev2 = vectordb.query(vs, q2)

    # News corpus analysis
    corpus = []
    for art in articles[:12]:
        combined = art["headline"] + " " + art["summary"]
        sent = analysis.analyse_sentiment(combined)
        corpus.append({
            "headline":          art["headline"],
            "event_category":    analysis.classify_event(combined),
            "sentiment_label":   sent["label"],
            "sentiment_score":   sent["score"],
            "reasoning_summary": (f"TextBlob polarity={sent['score']:.3f}; "
                                  f"category={analysis.classify_event(combined)}; "
                                  f"source={art['source']}"),
            "source_relevance":  analysis.source_relevance(art["source"], ticker, art["headline"]),
        })
    # Aggregated sentiment
    scores = [c["sentiment_score"] for c in corpus] or [0.0]
    ssum   = round(sum(scores), 4)
    smax   = round(max(scores), 4)
    smin   = round(min(scores), 4)
    votes  = [c["sentiment_label"] for c in corpus]
    major  = max(set(votes), key=votes.count) if votes else "UNKNOWN"

    # Signals
    drift    = analysis.price_drift_bp(prices)
    beta_val = fund.get("beta", 1.0)
    regime   = analysis.vol_regime(prices, beta_val)
    conf     = round(min(1.0, len(articles)/20) * (0.70 if regime=="high" else 0.90), 3)

    # Decision
    if ssum > 0 and drift > 0:
        decision, outlook, rp = "buy",  14, "risk-seeking"
        reason = (f"Positive sentiment (sum={ssum}) + upward momentum "
                  f"({drift} bps) → bullish bias.")
    elif ssum < 0 and drift < 0:
        decision, outlook, rp = "sell",  7, "risk-averse"
        reason = (f"Negative sentiment (sum={ssum}) + downward drift "
                  f"({drift} bps) → elevated downside risk.")
    else:
        decision, outlook = "hold", 10
        rp     = "risk-averse" if regime=="high" else "risk-seeking"
        reason = (f"Mixed signals: sentiment={ssum}, drift={drift} bps. "
                  f"No clear directional conviction.")

    # Data window
    wstart = prices[0]["date"]  if prices else (datetime.utcnow()-timedelta(days=60)).strftime("%Y-%m-%d")
    wend   = prices[-1]["date"] if prices else datetime.utcnow().strftime("%Y-%m-%d")

    pe = fund.get("pe_ratio", 0.0)
    if isinstance(pe, float) and math.isnan(pe):
        pe = 0.0

    result = {
        "analysis_metadata": {
            "ticker":      ticker,
            "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_window": f"{wstart} to {wend}",
        },
        "fundamental_indicators": {
            "market_capitalization_billions": fund.get("market_cap_B", 0.0),
            "pe_ratio":                       pe,
            "dividend_yield_percent":         fund.get("dividend_yield", 0.0),
            "fifty_two_week_high":            fund.get("52w_high", 0.0),
            "fifty_two_week_low":             fund.get("52w_low", 0.0),
            "latest_quarterly_dividend":      fund.get("latest_div", 0.0),
        },
        "news_corpus_analysis": corpus,
        "daily_aggregated_sentiment": {
            "sentiment_sum":  ssum,
            "sentiment_max":  smax,
            "sentiment_min":  smin,
            "majority_vote":  major,
            "news_count":     len(corpus),
        },
        "predictive_signals": {
            "query_decomposition": [
                f"sub_question_1: {q1} | evidence_retrieval: {ev1[:220]}…",
                f"sub_question_2: {q2} | numerical_computation: drift={drift}bps, beta={beta_val}",
            ],
            "price_drift_prediction":    drift,
            "confidence_score":          conf,
            "implied_volatility_regime": regime,
        },
        "investment_recommendation": {
            "investment_decision":     decision,
            "summary_reason":          reason,
            "short_term_outlook_days": outlook,
            "risk_profile":            rp,
            "memory_indices":          analysis.mem_indices(drift, ssum),
        },
    }

    return result