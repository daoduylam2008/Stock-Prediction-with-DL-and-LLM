from datetime import datetime, timedelta, timezone
import random


def _price_series(base, days=60, seed=42):
    random.seed(seed)
    records = []
    price = base
    today = datetime.now(timezone.utc)
    
    for i in range(days, 0, -1):
        dt = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        chg = random.gauss(0, 0.012)
        open_p  = round(price, 2)
        close_p = round(price * (1 + chg), 2)
        high_p  = round(max(open_p, close_p) * (1 + random.uniform(0, 0.008)), 2)
        low_p   = round(min(open_p, close_p) * (1 - random.uniform(0, 0.008)), 2)
        records.append({"date": dt, "open": open_p, "close": close_p,
                        "high": high_p, "low": low_p,
                        "volume": random.randint(40_000_000, 120_000_000)})
        price = close_p

    return records


MOCK_FUNDAMENTALS = {
    "AAPL": {
        "market_cap_B": 3214.5, "pe_ratio": 28.4, "dividend_yield": 0.52,
        "52w_high": 237.23, "52w_low": 164.08, "latest_div": 0.25,
        "current_price": 201.12, "volume": 72_000_000, "beta": 1.24,
        "name": "Apple Inc.",
    },
    "MSFT": {
        "market_cap_B": 3100.2, "pe_ratio": 34.1, "dividend_yield": 0.71,
        "52w_high": 468.35, "52w_low": 344.79, "latest_div": 0.75,
        "current_price": 415.60, "volume": 22_000_000, "beta": 0.89,
        "name": "Microsoft Corporation",
    },
    "TSLA": {
        "market_cap_B": 680.1, "pe_ratio": 55.2, "dividend_yield": 0.0,
        "52w_high": 358.64, "52w_low": 138.80, "latest_div": 0.0,
        "current_price": 218.45, "volume": 90_000_000, "beta": 2.1,
        "name": "Tesla Inc.",
    },
    "GOOGL": {
        "market_cap_B": 2100.3, "pe_ratio": 22.8, "dividend_yield": 0.48,
        "52w_high": 207.05, "52w_low": 155.63, "latest_div": 0.20,
        "current_price": 174.90, "volume": 25_000_000, "beta": 1.04,
        "name": "Alphabet Inc.",
    },
}
DEFAULT_FUND = {
    "market_cap_B": 100.0, "pe_ratio": 20.0, "dividend_yield": 1.0,
    "52w_high": 150.0, "52w_low": 90.0, "latest_div": 0.10,
    "current_price": 120.0, "volume": 10_000_000, "beta": 1.0,
    "name": "Unknown Corp",
}

MOCK_PRICES = {
    "AAPL": _price_series(170, seed=1),
    "MSFT": _price_series(380, seed=2),
    "TSLA": _price_series(195, seed=3),
    "GOOGL": _price_series(160, seed=4),
}


def _articles(ticker, company):
    headlines = [
        (f"{company} Reports Record Quarterly Revenue, Beats Analyst Estimates",
         "Earnings surged driven by strong product sales and services growth. Wall Street reacted positively.",
         0.7),
        (f"{ticker} Stock Upgrade: Analyst Raises Price Target to $250",
         "Leading bank upgrades outlook citing AI-driven growth and robust demand pipeline.",
         0.6),
        (f"Macro Headwinds: Rising Interest Rates Pressure Tech Stocks Including {ticker}",
         "Federal Reserve signals further rate hikes, weighing on high-multiple tech valuations.",
         -0.4),
        (f"{company} Unveils New AI-Powered Product Line at Annual Conference",
         "Innovation in generative AI and hardware integration expected to drive next growth cycle.",
         0.5),
        (f"Regulatory Scrutiny Mounts for {ticker} Amid Antitrust Investigation",
         "DOJ and FTC probing market practices; legal costs and strategic risk increase.",
         -0.5),
        (f"{company} Dividend Increased by 5% — Signals Management Confidence",
         "Capital return program expanded. Board approved buyback of $10B in additional shares.",
         0.4),
        (f"Supply Chain Disruptions Could Weigh on {ticker} Near-Term Margins",
         "Component shortages in Asia impacting production schedules into next quarter.",
         -0.3),
        (f"Institutional Investors Increase Stakes in {ticker} Following Dip",
         "Several large funds disclosed increased holdings, signaling long-term confidence.",
         0.35),
        (f"{ticker} Partnership with Major Cloud Provider Expands Global Reach",
         "Strategic agreement expected to add $2B annual recurring revenue over three years.",
         0.55),
        (f"Bearish Sentiment Grows as {ticker} Misses Revenue Guidance",
         "Weaker-than-expected forward guidance spooked investors; stock fell 4% after hours.",
         -0.6),
    ]

    results = []
    for i, (h, s, _) in enumerate(headlines):
        results.append({
            "headline": h,
            "summary": s,
            "link": f"https://finance.example.com/{ticker.lower()}-{i}",
            "published": (datetime.utcnow() - timedelta(days=i*2)).isoformat(),
            "source": ["reuters.com","cnbc.com","seekingalpha.com","bloomberg.com","wsj.com"][i % 5],
        })

    return results


def get_mock_fundamentals(ticker):
    return MOCK_FUNDAMENTALS.get(ticker, {**DEFAULT_FUND, "name": f"{ticker} Corp"})


def get_mock_prices(ticker):
    if ticker not in MOCK_PRICES:
        MOCK_PRICES[ticker] = _price_series(100, seed=hash(ticker) % 100)

    return MOCK_PRICES[ticker]

def get_mock_articles(ticker):
    fund = get_mock_fundamentals(ticker)

    return _articles(ticker, fund["name"])
