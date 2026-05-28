import warnings
from typing import List, Dict

warnings.filterwarnings("ignore")

from textblob import TextBlob

EVENT_KW = {
    "earnings":   ["earnings","eps","revenue","profit","loss","quarterly","beat","miss"],
    "merger_acq": ["merger","acquisition","buyout","takeover","deal","acquire"],
    "regulatory": ["sec","ftc","doj","regulation","antitrust","lawsuit","fine","penalty"],
    "macro":      ["fed","interest rate","inflation","gdp","recession","economy"],
    "product":    ["launch","product","release","innovation","patent"],
    "leadership": ["ceo","cfo","executive","resign","appoint","hire"],
    "analyst":    ["upgrade","downgrade","price target","analyst","rating"],
}

def classify_event(text: str) -> str:
    tl = text.lower()
    for cat, kws in EVENT_KW.items():
        if any(k in tl for k in kws):
            return cat
    return "general"

def analyse_sentiment(text: str) -> Dict:
    score = round(TextBlob(text).sentiment.polarity, 4)
    pos_w = ["surge","soar","beat","record","upgrade","strong","growth","profit","buy","expanded"]
    neg_w = ["drop","fall","miss","downgrade","loss","decline","risk","sell","cut","bearish"]
    tl    = text.lower()
    ph, nh = sum(1 for w in pos_w if w in tl), sum(1 for w in neg_w if w in tl)
    if score > 0.1 or ph > nh:
        label = "YES"
    elif score < -0.1 or nh > ph:
        label = "NO"
    else:
        label = "UNKNOWN"
    return {"score": score, "label": label}

def source_relevance(source: str, ticker: str, headline: str) -> int:
    score = 50
    if any(p in source.lower() for p in ["reuters","bloomberg","wsj","ft.com","cnbc","seekingalpha"]):
        score += 20
    if ticker.lower() in headline.lower():
        score += 20
    if "finance" in source.lower() or "market" in source.lower():
        score += 10
    return min(score, 100)

def price_drift_bp(prices: List[Dict]) -> float:
    if len(prices) < 2:
        return 0.0
    closes = [r["close"] for r in prices[-10:]]
    return round((closes[-1] - closes[0]) / (closes[0] or 1) * 10_000, 1)

def vol_regime(prices: List[Dict], beta: float) -> str:
    if len(prices) < 5:
        return "high" if beta > 1.2 else "low"
    atr = [abs(r["high"] - r["low"]) / (r["close"] or 1) for r in prices[-14:]]
    return "high" if (sum(atr)/len(atr)) > 0.025 or beta > 1.2 else "low"

def mem_indices(drift: float, ssum: float) -> Dict:
    s = min(100, max(0, int(50 + drift / 20)))
    m = min(100, max(0, int(50 + ssum * 10)))
    return {"short_term": s, "mid_term": m, "long_term": min(100, max(0, (s+m)//2))}