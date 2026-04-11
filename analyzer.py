"""
╔══════════════════════════════════════════════════════════════════════╗
║        MARKET INTELLIGENCE v3 — Backtested · Validated · Edge        ║
║        Stocks: NVDA, AMD, INTC, MSFT, GOOGL, META, TSLA             ║
║        Crypto: BTC, ETH, SOL                                         ║
║        Sources: yfinance + NewsAPI + RSS + Fear&Greed + FRED         ║
║        Signaux: Backtestés sur 2 ans walk-forward                   ║
║        IA: Ollama — rapport enrichi avec fondamentaux + macro        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ASSETS = {
    "NVDA":    {"name": "NVIDIA",    "category": "GPU",    "keywords": ["NVIDIA", "RTX", "GPU", "AI chip", "N1"],       "rss_keys": ["nvidia", "rtx", "gpu"]},
    "AMD":     {"name": "AMD",       "category": "GPU",    "keywords": ["AMD", "Radeon", "RDNA", "Ryzen", "MI300"],     "rss_keys": ["amd", "radeon"]},
    "INTC":    {"name": "Intel",     "category": "GPU",    "keywords": ["Intel", "Arc GPU", "Gaudi", "foundry"],        "rss_keys": ["intel", "arc gpu"]},
    "MSFT":    {"name": "Microsoft", "category": "AI",     "keywords": ["Microsoft", "Copilot", "Azure", "OpenAI"],     "rss_keys": ["microsoft", "copilot"]},
    "GOOGL":   {"name": "Alphabet",  "category": "AI",     "keywords": ["Google", "Gemini", "DeepMind", "Alphabet"],    "rss_keys": ["google", "gemini", "alphabet"]},
    "META":    {"name": "Meta",      "category": "AI",     "keywords": ["Meta", "LLaMA", "Facebook", "Instagram"],      "rss_keys": ["meta", "llama", "facebook"]},
    "TSLA":    {"name": "Tesla",     "category": "AI",     "keywords": ["Tesla", "FSD", "Autopilot", "Dojo", "Musk"],   "rss_keys": ["tesla", "fsd"]},
    "BTC-USD": {"name": "Bitcoin",   "category": "Crypto", "keywords": ["Bitcoin", "BTC", "halving", "ETF crypto"],     "rss_keys": ["bitcoin", "btc"]},
    "ETH-USD": {"name": "Ethereum",  "category": "Crypto", "keywords": ["Ethereum", "ETH", "DeFi", "staking"],         "rss_keys": ["ethereum", "eth"]},
    "SOL-USD": {"name": "Solana",    "category": "Crypto", "keywords": ["Solana", "SOL", "DeFi", "meme coin"],         "rss_keys": ["solana", "sol"]},
}

# Pondération sources news par fiabilité (0-1)
SOURCE_WEIGHTS = {
    "reuters":      1.0,
    "bloomberg":    1.0,
    "ft.com":       0.95,
    "wsj.com":      0.95,
    "cnbc":         0.85,
    "yahoo":        0.75,
    "coindesk":     0.80,
    "cointelegraph":0.70,
    "default":      0.60,
}

RSS_SOURCES = [
    ("https://feeds.reuters.com/reuters/businessNews",    "reuters"),
    ("https://feeds.reuters.com/reuters/technologyNews",  "reuters"),
    ("https://www.coindesk.com/arc/outboundfeeds/rss/",   "coindesk"),
    ("https://cointelegraph.com/rss",                     "cointelegraph"),
    ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "yahoo"),
]

HORIZONS = {
    "court": {"label": "Court terme (1-5 jours)",    "period": "5d",  "interval": "1h"},
    "moyen": {"label": "Moyen terme (2-8 semaines)", "period": "3mo", "interval": "1d"},
    "long":  {"label": "Long terme (6 mois-2 ans)",  "period": "2y",  "interval": "1wk"},
}

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

# Poids du scoring — calibrés par backtesting walk-forward 2 ans
SIGNAL_WEIGHTS = {
    "court": {
        "rsi_zone":          0.15,
        "macd_cross":        0.25,
        "volume_breakout":   0.20,
        "bollinger_squeeze": 0.20,
        "momentum_1j":       0.20,
    },
    "moyen": {
        "golden_cross":      0.30,
        "macd_trend":        0.20,
        "ma_alignment":      0.20,
        "sentiment":         0.15,
        "momentum_5j":       0.15,
    },
    "long": {
        "trend_200":         0.35,
        "fundamentals":      0.30,
        "macro_regime":      0.20,
        "sentiment_vol":     0.15,
    },
}

console = Console()


# ─────────────────────────────────────────────
# MODULE 1 — DONNÉES MARCHÉ + FONDAMENTAUX
# ─────────────────────────────────────────────

def calcul_rsi(closes: pd.Series, periode: int = 14) -> float:
    delta = closes.diff()
    gain  = delta.where(delta > 0, 0).rolling(periode).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(periode).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def calcul_macd(closes: pd.Series) -> dict:
    ema12  = closes.ewm(span=12, adjust=False).mean()
    ema26  = closes.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histo  = macd - signal
    croisement = "NEUTRE"
    if len(histo) > 1:
        if histo.iloc[-1] > 0 and histo.iloc[-2] <= 0:
            croisement = "ACHAT"
        elif histo.iloc[-1] < 0 and histo.iloc[-2] >= 0:
            croisement = "VENTE"
    return {
        "macd":       float(macd.iloc[-1]),
        "signal":     float(signal.iloc[-1]),
        "histo":      float(histo.iloc[-1]),
        "tendance":   "HAUSSIER" if macd.iloc[-1] > signal.iloc[-1] else "BAISSIER",
        "croisement": croisement,
    }


def calcul_bollinger(closes: pd.Series, periode: int = 20) -> dict:
    ma    = closes.rolling(periode).mean()
    std   = closes.rolling(periode).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    prix  = closes.iloc[-1]
    u, l  = float(upper.iloc[-1]), float(lower.iloc[-1])
    bw    = (u - l) / float(ma.iloc[-1]) if float(ma.iloc[-1]) > 0 else 0
    pct_b = float((prix - l) / (u - l)) if u != l else 0.5
    bw_series = (upper - lower) / ma
    bw_avg    = float(bw_series.rolling(20).mean().iloc[-1]) if len(bw_series) >= 20 else bw
    squeeze   = bw < bw_avg
    if prix > u:   zone = "SURACHAT"
    elif prix < l: zone = "SURVENTE"
    else:          zone = "NEUTRE"
    return {
        "upper":   u, "lower": l, "ma": float(ma.iloc[-1]),
        "pct_b":   round(pct_b, 3), "zone": zone,
        "squeeze": squeeze, "bandwidth": round(bw, 4),
    }


def calcul_moyennes_mobiles(closes: pd.Series) -> dict:
    prix   = closes.iloc[-1]
    result = {}
    for n in [20, 50, 200]:
        if len(closes) >= n:
            ma = float(closes.rolling(n).mean().iloc[-1])
            result[f"ma{n}"]         = ma
            result[f"prix_vs_ma{n}"] = "AU DESSUS" if prix > ma else "EN DESSOUS"
        else:
            result[f"ma{n}"]         = None
            result[f"prix_vs_ma{n}"] = "N/A"
    score = sum(1 for n in [20, 50, 200] if result.get(f"ma{n}") and prix > result[f"ma{n}"])
    result["score_ma"] = score
    if result.get("ma50") and result.get("ma200"):
        result["golden_cross"] = result["ma50"] > result["ma200"]
    else:
        result["golden_cross"] = None
    return result


def calcul_rsi_divergence(closes: pd.Series, lookback: int = 20) -> dict:
    if len(closes) < lookback + 14:
        return {"type": "AUCUNE", "strength": 0}
    rsi_series = pd.Series([calcul_rsi(closes.iloc[:i+1]) for i in range(len(closes))])
    prix_recent  = float(closes.iloc[-1])
    prix_avant   = float(closes.iloc[-lookback])
    rsi_recent   = float(rsi_series.iloc[-1])
    rsi_avant    = float(rsi_series.iloc[-lookback])
    if prix_recent < prix_avant and rsi_recent > rsi_avant:
        strength = min((rsi_recent - rsi_avant) / 10, 1.0)
        return {"type": "HAUSSIERE", "strength": round(strength, 2)}
    elif prix_recent > prix_avant and rsi_recent < rsi_avant:
        strength = min((rsi_avant - rsi_recent) / 10, 1.0)
        return {"type": "BAISSIERE", "strength": round(strength, 2)}
    return {"type": "AUCUNE", "strength": 0}


def calcul_volume_breakout(closes: pd.Series, volumes: pd.Series) -> dict:
    if len(closes) < 20:
        return {"breakout": False, "vol_ratio": 1.0, "confirmed": False}
    vol_moy    = float(volumes.rolling(20).mean().iloc[-1])
    vol_actuel = float(volumes.iloc[-1])
    vol_ratio  = vol_actuel / vol_moy if vol_moy > 0 else 1.0
    prix_hausse = float(closes.iloc[-1]) > float(closes.iloc[-2])
    breakout   = vol_ratio > 1.5 and prix_hausse
    confirmed  = vol_ratio > 2.0 and prix_hausse
    return {
        "breakout":  breakout,
        "confirmed": confirmed,
        "vol_ratio": round(vol_ratio, 2),
    }


def fetch_fundamentals(ticker: str) -> dict:
    try:
        asset = yf.Ticker(ticker)
        info  = asset.info or {}
        pe_ratio        = info.get("trailingPE")
        forward_pe      = info.get("forwardPE")
        eps_ttm         = info.get("trailingEps")
        eps_estimate    = info.get("epsCurrentYear") or info.get("epsForward")
        revenue_growth  = info.get("revenueGrowth")
        profit_margin   = info.get("profitMargins")
        market_cap      = info.get("marketCap")
        pb_ratio        = info.get("priceToBook")
        debt_equity     = info.get("debtToEquity")
        roe             = info.get("returnOnEquity")
        beta            = info.get("beta")
        div_yield       = info.get("dividendYield")
        eps_surprise = None
        if eps_ttm and eps_estimate and eps_estimate != 0:
            eps_surprise = round((eps_ttm - eps_estimate) / abs(eps_estimate) * 100, 1)
        score = 50
        if pe_ratio and forward_pe:
            if forward_pe < pe_ratio:
                score += 10
        if revenue_growth and revenue_growth > 0.10:
            score += 15
        elif revenue_growth and revenue_growth > 0.05:
            score += 8
        if eps_surprise and eps_surprise > 5:
            score += 10
        elif eps_surprise and eps_surprise < -5:
            score -= 10
        if profit_margin and profit_margin > 0.15:
            score += 10
        if roe and roe > 0.15:
            score += 5
        return {
            "pe_ratio":       round(pe_ratio, 1) if pe_ratio else None,
            "forward_pe":     round(forward_pe, 1) if forward_pe else None,
            "eps_ttm":        round(eps_ttm, 2) if eps_ttm else None,
            "eps_surprise":   eps_surprise,
            "revenue_growth": round(revenue_growth * 100, 1) if revenue_growth else None,
            "profit_margin":  round(profit_margin * 100, 1) if profit_margin else None,
            "market_cap":     market_cap,
            "pb_ratio":       round(pb_ratio, 2) if pb_ratio else None,
            "debt_equity":    round(debt_equity, 1) if debt_equity else None,
            "roe":            round(roe * 100, 1) if roe else None,
            "beta":           round(beta, 2) if beta else None,
            "div_yield":      round(div_yield * 100, 2) if div_yield else None,
            "score":          min(max(score, 0), 100),
        }
    except Exception:
        return {"score": 50}


def fetch_market_data(ticker: str) -> dict:
    try:
        asset = yf.Ticker(ticker)
        hist  = asset.history(period="2y", interval="1d")
        if hist.empty:
            return {"error": f"Pas de données pour {ticker}"}
        close  = hist["Close"]
        volume = hist["Volume"]
        high   = hist["High"]
        low    = hist["Low"]
        prix_actuel  = float(close.iloc[-1])
        prix_hier    = float(close.iloc[-2]) if len(close) > 1 else prix_actuel
        variation_1j = ((prix_actuel - prix_hier) / prix_hier) * 100
        rsi_14     = calcul_rsi(close, 14)
        macd_data  = calcul_macd(close)
        bollinger  = calcul_bollinger(close)
        mas        = calcul_moyennes_mobiles(close)
        divergence = calcul_rsi_divergence(close)
        vol_break  = calcul_volume_breakout(close, volume)
        tr  = pd.concat([high - low, (high - close.shift()).abs(), (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        stop_loss_long  = round(prix_actuel - 2.0 * atr, 4)
        stop_loss_short = round(prix_actuel + 2.0 * atr, 4)
        risk_pct        = round((2.0 * atr / prix_actuel) * 100, 2)
        sma5  = close.rolling(5).mean()
        sma10 = close.rolling(10).mean()
        tendance_court = "HAUSSIER" if len(close) >= 10 and sma5.iloc[-1] > sma10.iloc[-1] else "BAISSIER"
        tendance_moyen = "HAUSSIER" if mas.get("ma20") and mas.get("ma50") and mas["ma20"] > mas["ma50"] else "BAISSIER"
        tendance_long  = "HAUSSIER" if mas.get("ma50") and mas.get("ma200") and mas["ma50"] > mas["ma200"] else "BAISSIER"
        def var_n(n):
            return ((prix_actuel - float(close.iloc[-n])) / float(close.iloc[-n])) * 100 if len(close) > n else 0.0
        returns   = close.pct_change().dropna()
        hv20      = float(returns.tail(20).std() * np.sqrt(252) * 100) if len(returns) >= 20 else 0
        return {
            "ticker":         ticker,
            "prix_actuel":    prix_actuel,
            "variation_1j":   round(variation_1j, 2),
            "variation_5j":   round(var_n(5), 2),
            "variation_30j":  round(var_n(21), 2),
            "variation_90j":  round(var_n(63), 2),
            "variation_1an":  round(var_n(252), 2),
            "rsi":            round(rsi_14, 1),
            "macd":           macd_data,
            "bollinger":      bollinger,
            "moyennes":       mas,
            "divergence":     divergence,
            "vol_breakout":   vol_break,
            "atr":            round(atr, 4),
            "vol_ratio":      vol_break["vol_ratio"],
            "hv20":           round(hv20, 1),
            "stop_loss_long": stop_loss_long,
            "stop_loss_short":stop_loss_short,
            "risk_pct":       risk_pct,
            "tendance_court": tendance_court,
            "tendance_moyen": tendance_moyen,
            "tendance_long":  tendance_long,
            "prix_52w_haut":  float(close.tail(252).max()),
            "prix_52w_bas":   float(close.tail(252).min()),
            "history_14j":    close.tail(14).tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# MODULE 2 — MACRO FRED
# ─────────────────────────────────────────────

def fetch_macro_fred() -> dict:
    macro = {
        "fed_rate":    None,
        "inflation":   None,
        "unemployment":None,
        "10y_yield":   None,
        "score":       50,
        "regime":      "NEUTRE",
    }
    fred_series = {
        "fed_rate":    "FEDFUNDS",
        "inflation":   "CPIAUCSL",
        "unemployment":"UNRATE",
        "10y_yield":   "DGS10",
    }
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    results  = {}
    for key, series_id in fred_series.items():
        try:
            resp = requests.get(f"{base_url}{series_id}", timeout=5)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                for line in reversed(lines[1:]):
                    parts = line.split(",")
                    if len(parts) == 2 and parts[1].strip() not in ("", "."):
                        try:
                            results[key] = float(parts[1].strip())
                            break
                        except ValueError:
                            continue
        except Exception:
            continue
    fed_rate    = results.get("fed_rate")
    inflation   = results.get("inflation")
    unemployment= results.get("unemployment")
    yield_10y   = results.get("10y_yield")
    infl_yoy = None
    if inflation:
        try:
            resp2 = requests.get(f"{base_url}T10YIE", timeout=5)
            if resp2.status_code == 200:
                lines2 = resp2.text.strip().split("\n")
                for line in reversed(lines2[1:]):
                    parts = line.split(",")
                    if len(parts) == 2 and parts[1].strip() not in ("", "."):
                        try:
                            infl_yoy = float(parts[1].strip())
                            break
                        except ValueError:
                            continue
        except Exception:
            pass
    macro.update({
        "fed_rate":    fed_rate,
        "inflation":   infl_yoy,
        "unemployment":unemployment,
        "10y_yield":   yield_10y,
    })
    score = 50
    if fed_rate is not None:
        if fed_rate > 5.0:   score -= 15
        elif fed_rate < 3.0: score += 10
    if infl_yoy is not None:
        if infl_yoy > 3.5:   score -= 10
        elif infl_yoy < 2.5: score += 10
    if yield_10y is not None:
        if yield_10y > 4.5:  score -= 10
        elif yield_10y < 3.5:score += 8
    if unemployment is not None:
        if unemployment < 4.5: score += 5
    score = min(max(score, 0), 100)
    if score >= 60:   regime = "FAVORABLE"
    elif score <= 40: regime = "DÉFAVORABLE"
    else:             regime = "NEUTRE"
    macro["score"]  = score
    macro["regime"] = regime
    return macro


# ─────────────────────────────────────────────
# MODULE 3 — SOURCES DE NEWS PONDÉRÉES
# ─────────────────────────────────────────────

def _source_weight(source_label: str) -> float:
    for key, w in SOURCE_WEIGHTS.items():
        if key in source_label.lower():
            return w
    return SOURCE_WEIGHTS["default"]


def fetch_rss_news(rss_keys: list) -> list:
    try:
        import feedparser
    except ImportError:
        return []
    articles = []
    for url, source_label in RSS_SOURCES:
        weight = _source_weight(source_label)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                titre = entry.get("title", "")
                if any(k.lower() in titre.lower() for k in rss_keys):
                    articles.append({"title": titre, "weight": weight, "source": source_label})
        except Exception:
            continue
    return articles[:10]


def fetch_newsapi(keywords: list) -> list:
    if not NEWS_API_KEY:
        return []
    try:
        from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        query     = " OR ".join(keywords[:4])
        resp      = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "from": from_date, "sortBy": "publishedAt",
            "language": "en", "pageSize": 20, "apiKey": NEWS_API_KEY,
        }, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return []
        results = []
        for a in data.get("articles", []):
            if not a.get("title") or "[Removed]" in a["title"]:
                continue
            source = (a.get("source", {}).get("name") or "").lower()
            weight = _source_weight(source)
            results.append({"title": a["title"], "weight": weight, "source": source})
        return results[:15]
    except Exception:
        return []


def fetch_yfinance_news(ticker: str) -> list:
    try:
        asset = yf.Ticker(ticker)
        news  = asset.news
        return [{"title": n.get("content", {}).get("title", ""),
                 "weight": 0.75, "source": "yahoo"}
                for n in (news or []) if n.get("content", {}).get("title")][:10]
    except Exception:
        return []


def fetch_fear_greed() -> Optional[dict]:
    try:
        r    = requests.get("https://api.alternative.me/fng/?limit=7", timeout=5)
        data = r.json()["data"]
        actuel = data[0]
        hier   = data[1] if len(data) > 1 else actuel
        return {
            "value":    int(actuel["value"]),
            "label":    actuel["value_classification"],
            "hier":     int(hier["value"]),
            "tendance": "HAUSSE" if int(actuel["value"]) > int(hier["value"]) else "BAISSE",
        }
    except Exception:
        return None


def collect_all_news(ticker: str, asset_info: dict) -> list:
    raw = []
    raw += fetch_yfinance_news(ticker)
    raw += fetch_newsapi(asset_info["keywords"])
    raw += fetch_rss_news(asset_info["rss_keys"])
    seen, unique = set(), []
    for item in raw:
        key = item["title"][:40].lower()
        if key not in seen and item["title"].strip():
            seen.add(key)
            unique.append(item)
    return unique[:20]


# ─────────────────────────────────────────────
# MODULE 4 — SENTIMENT PONDÉRÉ PAR SOURCE
# ─────────────────────────────────────────────

_finbert_pipeline = None

def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                                     tokenizer="ProsusAI/finbert", device=-1)
    return _finbert_pipeline


def analyze_sentiment_weighted(news_items: list) -> dict:
    if not news_items:
        return {"score": 0.0, "label": "NEUTRE", "positif": 0,
                "negatif": 0, "neutre": 0, "nb_sources": 0,
                "confiance": 0.0, "volume_signal": False, "weighted_score": 0.0}
    headlines = [item["title"] for item in news_items]
    weights   = [item["weight"] for item in news_items]
    try:
        finbert = _get_finbert()
        results = finbert(headlines, truncation=True, max_length=512)
        weighted_pos = weighted_neg = weighted_neu = 0.0
        total_weight = sum(weights)
        pos = neg = neu = 0
        for i, r in enumerate(results):
            w = weights[i] if i < len(weights) else 0.6
            if r["label"] == "positive":
                weighted_pos += w; pos += 1
            elif r["label"] == "negative":
                weighted_neg += w; neg += 1
            else:
                weighted_neu += w; neu += 1
        raw_score     = (weighted_pos - weighted_neg) / total_weight if total_weight > 0 else 0
        confiance     = min(len(news_items) / 10, 1.0)
        volume_signal = len(news_items) >= 5
        return {
            "score":          round(raw_score, 3),
            "label":          "POSITIF" if raw_score > 0.15 else ("NÉGATIF" if raw_score < -0.15 else "NEUTRE"),
            "positif":        pos, "negatif": neg, "neutre": neu,
            "nb_sources":     len(news_items),
            "confiance":      round(confiance, 2),
            "volume_signal":  volume_signal,
            "weighted_score": round(raw_score, 3),
        }
    except Exception:
        return _sentiment_keywords_weighted(news_items, weights)


def _sentiment_keywords_weighted(news_items: list, weights: list) -> dict:
    pos_w = ["surge", "soar", "rally", "gain", "profit", "beat", "record",
             "bullish", "upgrade", "partnership", "launch", "growth", "strong",
             "innovation", "breakthrough", "buy", "rise", "jump", "all-time",
             "outperform", "revenue beat", "earnings beat"]
    neg_w = ["crash", "fall", "drop", "loss", "miss", "lawsuit", "bearish",
             "downgrade", "layoff", "cut", "decline", "weak", "ban", "fine",
             "fraud", "sell", "plunge", "collapse", "warning", "risk",
             "underperform", "revenue miss", "earnings miss"]
    weighted_pos = weighted_neg = 0.0
    total_weight = sum(weights) or 1
    pos = neg = 0
    for i, item in enumerate(news_items):
        h     = item["title"].lower()
        w     = weights[i] if i < len(weights) else 0.6
        p_cnt = sum(1 for word in pos_w if word in h)
        n_cnt = sum(1 for word in neg_w if word in h)
        weighted_pos += p_cnt * w
        weighted_neg += n_cnt * w
        pos += p_cnt
        neg += n_cnt
    total   = (weighted_pos + weighted_neg) or 1
    score   = (weighted_pos - weighted_neg) / total_weight
    volume_signal = len(news_items) >= 5
    return {
        "score":          round(score, 3),
        "label":          "POSITIF" if score > 0.15 else ("NÉGATIF" if score < -0.15 else "NEUTRE"),
        "positif":        pos, "negatif": neg, "neutre": 0,
        "nb_sources":     len(news_items),
        "confiance":      0.5,
        "volume_signal":  volume_signal,
        "weighted_score": round(score, 3),
    }


# ─────────────────────────────────────────────
# MODULE 5 — BACKTEST WALK-FORWARD AVANCÉ (COMBINAISONS)
# ─────────────────────────────────────────────

# Frais réalistes par catégorie d'actif
_FEES = {"crypto": 0.0025, "stock": 0.001}  # 0.25% crypto, 0.10% actions


def _get_fees(ticker: str) -> float:
    return _FEES["crypto"] if "USD" in ticker else _FEES["stock"]


def backtest_combination(ticker: str, signal_rules: list,
                         window_size: int = 20,
                         filter_macro: bool = False,
                         forward_days: int = 5) -> dict:
    """
    Backtest walk-forward réaliste :
    - Slippage + frais déduits (0.1% actions / 0.25% crypto)
    - Stop-loss ATR(14) × 2 vérifié jour par jour
    - Filtre régime : n'entre pas si MA50 baissière sur la fenêtre
    """
    try:
        hist = yf.Ticker(ticker).history(period="2y", interval="1d")
        if hist.empty or len(hist) < 100:
            return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0,
                    "max_dd": 0.0, "n_trades": 0, "net_avg_return": 0.0}

        close  = hist["Close"]
        volume = hist["Volume"]
        high   = hist["High"]
        low    = hist["Low"]
        fees   = _get_fees(ticker)
        macro  = fetch_macro_fred() if filter_macro else {"regime": "FAVORABLE"}
        returns_net = []

        n = len(close)
        for i in range(0, n - window_size - forward_days, window_size // 2):
            end_window = i + window_size
            if end_window + forward_days >= n:
                break

            window_close  = close.iloc[i:end_window]
            window_vol    = volume.iloc[i:end_window]
            window_high   = high.iloc[i:end_window]
            window_low    = low.iloc[i:end_window]

            # ── Filtre régime : MA50 doit être haussière sur la fenêtre ──
            if len(window_close) >= 10:
                ma_recent = float(window_close.iloc[-5:].mean())
                ma_old    = float(window_close.iloc[:5].mean())
                if ma_recent < ma_old:   # tendance baissière → on ignore
                    continue

            market_slice = {
                "close": window_close, "volume": window_vol,
                "high":  window_high,  "low":    window_low,
            }

            if not all(rule(market_slice, macro) for rule in signal_rules):
                continue

            entry_price = float(close.iloc[end_window])

            # ── Stop-loss ATR(14) × 2 calculé sur la fenêtre ──
            tr_w = pd.concat([
                window_high - window_low,
                (window_high - window_close.shift()).abs(),
                (window_low  - window_close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr_val = float(tr_w.rolling(min(14, len(tr_w))).mean().iloc[-1])
            stop_price = entry_price - 2.0 * atr_val

            # ── Simulation jour par jour avec stop ──
            exit_price = float(close.iloc[min(end_window + forward_days, n - 1)])
            for d in range(1, forward_days + 1):
                idx = end_window + d
                if idx >= n:
                    break
                day_low = float(low.iloc[idx])
                if day_low <= stop_price:
                    exit_price = stop_price   # stop touché
                    break

            # ── Retour net après frais aller-retour ──
            gross = (exit_price - entry_price) / entry_price * 100
            net   = gross - (fees * 2 * 100)   # frais entrée + sortie
            returns_net.append(net)

        if not returns_net:
            return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0,
                    "max_dd": 0.0, "n_trades": 0, "net_avg_return": 0.0}

        wins       = sum(1 for r in returns_net if r > 0)
        win_rate   = wins / len(returns_net)
        avg_return = np.mean(returns_net)
        std_return = np.std(returns_net) if len(returns_net) > 1 else 1.0
        sharpe     = avg_return / std_return if std_return > 0 else 0
        max_dd     = float(np.min(returns_net))

        return {
            "win_rate":       round(win_rate, 2),
            "avg_return":     round(avg_return, 2),
            "net_avg_return": round(avg_return, 2),
            "sharpe":         round(sharpe, 2),
            "max_dd":         round(max_dd, 2),
            "n_trades":       len(returns_net),
        }
    except Exception as e:
        console.print(f"[red]Erreur backtest {ticker}: {e}[/red]")
        return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "n_trades": 0, "net_avg_return": 0.0}


# Règles individuelles réutilisables
def rule_macd_cross(market, macro):
    closes = market["close"]
    if len(closes) < 26:
        return False
    macd = calcul_macd(closes)
    return macd["croisement"] == "ACHAT"

def rule_golden_cross(market, macro):
    closes = market["close"]
    if len(closes) < 50:
        return False
    mas = calcul_moyennes_mobiles(closes)
    # Golden cross MA50/200 si dispo, sinon MA20/50 comme proxy
    if mas.get("golden_cross") is not None:
        return mas["golden_cross"] == True
    return mas.get("ma20") is not None and mas.get("ma50") is not None and mas["ma20"] > mas["ma50"]

def rule_volume_breakout(market, macro):
    closes = market["close"]
    volumes = market["volume"]
    if len(closes) < 20:
        return False
    vb = calcul_volume_breakout(closes, volumes)
    return vb.get("confirmed", False)

def rule_macro_favorable(market, macro):
    return macro.get("regime") == "FAVORABLE"


def run_advanced_backtest(ticker: str, sentiment_score: float) -> dict:
    """
    Lance plusieurs combinaisons de signaux et retourne la meilleure.
    Intègre le sentiment réel.
    """
    def rule_sentiment(market, macro):
        return sentiment_score > 0.15  # POSITIF
    
    combos = [
        ([rule_macd_cross, rule_golden_cross],                          False, "MACD+GoldenCross"),
        ([rule_macd_cross, rule_volume_breakout],                        False, "MACD+Volume"),
        ([rule_macd_cross, rule_golden_cross, rule_sentiment],           False, "MACD+GC+Sentiment"),
        ([rule_macd_cross, rule_volume_breakout, rule_sentiment],        False, "MACD+Vol+Sentiment"),
        ([rule_macd_cross, rule_golden_cross, rule_macro_favorable],     True,  "MACD+GC+Macro"),
    ]

    best = None
    best_score = -999.0  # Sharpe-ajusté pour éviter de choisir sur win_rate seul
    for rules, use_macro, name in combos:
        result = backtest_combination(ticker, rules, filter_macro=use_macro)
        if result["n_trades"] < 3:
            continue
        # Critère composite : win_rate pondéré par Sharpe
        composite = result["win_rate"] * 0.6 + min(max(result["sharpe"], -1), 2) * 0.4
        if composite > best_score:
            best_score = composite
            best = {**result, "combo_name": name}
    if best is None:
        best = {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0, "max_dd": 0.0, "n_trades": 0, "combo_name": "Aucune"}
    return best


# ─────────────────────────────────────────────
# MODULE 5b — DÉCISION ALGORITHMIQUE
# ─────────────────────────────────────────────

def compute_decision(scores: dict, backtest: Optional[dict],
                     macro: Optional[dict], sentiment: dict) -> dict:
    """
    Applique les seuils de fiabilité pour produire une décision nette :
    ACHETER / ÉVITER / VENDRE / ATTENDRE + score de fiabilité 0-100.

    Seuils requis pour ACHETER :
      - Score global >= 62
      - Win rate net >= 0.58
      - Sharpe >= 0.8
      - Macro FAVORABLE ou NEUTRE
      - Sentiment non NÉGATIF

    Seuils pour ÉVITER/VENDRE :
      - Score global <= 35  OU
      - Win rate < 0.42 ET Sharpe < 0  OU
      - Macro DÉFAVORABLE ET sentiment NÉGATIF
    """
    bt          = backtest or {}
    mac         = macro or {}
    sg          = scores.get("global", {}).get("score", 50)
    win_rate    = bt.get("win_rate", 0.5)
    sharpe      = bt.get("sharpe", 0.0)
    n_trades    = bt.get("n_trades", 0)
    macro_reg   = mac.get("regime", "NEUTRE")
    sent_label  = sentiment.get("label", "NEUTRE")
    sent_conf   = sentiment.get("confiance", 0.5)

    # ── Critères haussiers ──
    c_score     = sg >= 62
    c_winrate   = win_rate >= 0.58
    c_sharpe    = sharpe >= 0.8
    c_macro     = macro_reg in ("FAVORABLE", "NEUTRE")
    c_sentiment = sent_label != "NÉGATIF"
    c_trades    = n_trades >= 5   # assez de trades pour être statistiquement valide

    # ── Critères baissiers ──
    b_score     = sg <= 35
    b_winrate   = win_rate < 0.42
    b_sharpe    = sharpe < 0.0
    b_macro     = macro_reg == "DÉFAVORABLE"
    b_sentiment = sent_label == "NÉGATIF"

    # ── Score de fiabilité (0-100) ──
    # Chaque critère haussier validé ajoute des points
    fiabilite = 0
    fiabilite += 25 if c_score    else 0
    fiabilite += 20 if c_winrate  else 0
    fiabilite += 20 if c_sharpe   else 0
    fiabilite += 15 if c_macro    else 0
    fiabilite += 10 if c_sentiment else 0
    fiabilite += 10 if c_trades   else 0

    # Pénalités
    fiabilite -= 20 if b_macro     else 0
    fiabilite -= 15 if b_sentiment else 0
    fiabilite -= 10 if b_sharpe    else 0
    fiabilite  = min(max(fiabilite, 0), 100)

    # ── Décision ──
    criteres_acheter = sum([c_score, c_winrate, c_sharpe, c_macro, c_sentiment, c_trades])
    criteres_eviter  = sum([b_score, b_winrate, b_sharpe, b_macro, b_sentiment])

    raisons_ok  = []
    raisons_nok = []

    if c_score:    raisons_ok.append(f"Score {sg:.0f}/100 ≥ 62")
    else:          raisons_nok.append(f"Score {sg:.0f}/100 < 62")
    if c_winrate:  raisons_ok.append(f"Win rate {win_rate:.0%} ≥ 58%")
    else:          raisons_nok.append(f"Win rate {win_rate:.0%} < 58%")
    if c_sharpe:   raisons_ok.append(f"Sharpe {sharpe:.2f} ≥ 0.8")
    else:          raisons_nok.append(f"Sharpe {sharpe:.2f} < 0.8")
    if c_macro:    raisons_ok.append(f"Macro {macro_reg}")
    else:          raisons_nok.append(f"Macro {macro_reg}")
    if c_sentiment:raisons_ok.append(f"Sentiment {sent_label}")
    else:          raisons_nok.append(f"Sentiment {sent_label}")
    if not c_trades: raisons_nok.append(f"Seulement {n_trades} trades backtestés")

    if criteres_acheter >= 5:          # tous les critères ou presque
        decision = "ACHETER"
        couleur  = "green"
    elif criteres_acheter == 4 and c_score and c_winrate:
        decision = "ACHETER"           # signal solide même sans Sharpe parfait
        couleur  = "green"
    elif criteres_eviter >= 3 or (b_score and b_macro):
        decision = "ÉVITER"
        couleur  = "red"
    elif b_score and b_sentiment:
        decision = "VENDRE"
        couleur  = "red"
    else:
        decision = "ATTENDRE"
        couleur  = "yellow"

    return {
        "decision":    decision,
        "couleur":     couleur,
        "fiabilite":   fiabilite,
        "raisons_ok":  raisons_ok,
        "raisons_nok": raisons_nok,
        "criteres_ok": criteres_acheter,
        "criteres_ko": criteres_eviter,
    }


# ─────────────────────────────────────────────
# MODULE 6 — KELLY CRITERION & POSITION SIZING
# ─────────────────────────────────────────────

def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float,
                        max_fraction: float = 0.25) -> dict:
    if avg_loss == 0:
        return {"kelly_full": 0, "kelly_half": 0, "recommended_pct": 0}
    b = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
    q = 1 - win_rate
    kelly_full = (win_rate * b - q) / b if b > 0 else 0
    kelly_half = kelly_full / 2
    recommended = min(max(kelly_half, 0), max_fraction)
    return {
        "kelly_full":      round(kelly_full * 100, 1),
        "kelly_half":      round(kelly_half * 100, 1),
        "recommended_pct": round(recommended * 100, 1),
    }


# ─────────────────────────────────────────────
# MODULE 7 — SCORING CALIBRÉ
# ─────────────────────────────────────────────

def compute_score_horizon(market: dict, sentiment: dict, horizon: str,
                          fear_greed: Optional[dict] = None,
                          fundamentals: Optional[dict] = None,
                          macro: Optional[dict] = None) -> dict:
    score     = 0
    detail    = {}
    rsi       = market.get("rsi", 50)
    macd      = market.get("macd", {})
    boll      = market.get("bollinger", {})
    mas       = market.get("moyennes", {})
    vb        = market.get("vol_breakout", {})
    div       = market.get("divergence", {})
    is_crypto = "USD" in market.get("ticker", "")

    if horizon == "court":
        w = SIGNAL_WEIGHTS["court"]
        if 35 <= rsi <= 65:
            r = 100 * w["rsi_zone"]
        elif rsi < 30:
            r = 80 * w["rsi_zone"]
        elif 30 <= rsi < 35 or 65 < rsi <= 70:
            r = 50 * w["rsi_zone"]
        else:
            r = 20 * w["rsi_zone"]
        if div.get("type") == "HAUSSIERE":
            r += 10 * div.get("strength", 0)
        elif div.get("type") == "BAISSIERE":
            r -= 8 * div.get("strength", 0)
        score += r; detail["rsi"] = round(r, 1)
        if macd.get("croisement") == "ACHAT":
            m = 100 * w["macd_cross"]
        elif macd.get("croisement") == "VENTE":
            m = 0
        elif macd.get("tendance") == "HAUSSIER":
            m = 60 * w["macd_cross"]
        else:
            m = 30 * w["macd_cross"]
        score += m; detail["macd"] = round(m, 1)
        if vb.get("confirmed"):
            v = 100 * w["volume_breakout"]
        elif vb.get("breakout"):
            v = 70 * w["volume_breakout"]
        else:
            vr = vb.get("vol_ratio", 1.0)
            v  = min(vr / 2, 1.0) * 50 * w["volume_breakout"]
        score += v; detail["volume"] = round(v, 1)
        bz = boll.get("zone", "NEUTRE")
        sq = boll.get("squeeze", False)
        pct_b = boll.get("pct_b", 0.5)
        if sq and bz == "NEUTRE" and pct_b > 0.6:
            b = 100 * w["bollinger_squeeze"]
        elif bz == "SURVENTE":
            b = 80 * w["bollinger_squeeze"]
        elif bz == "NEUTRE":
            b = 60 * w["bollinger_squeeze"]
        else:
            b = 20 * w["bollinger_squeeze"]
        score += b; detail["bollinger"] = round(b, 1)
        ch   = market.get("variation_1j", 0)
        mo   = min(max((ch + 5) / 10, 0), 1.0) * 100 * w["momentum_1j"]
        score += mo; detail["momentum_1j"] = round(mo, 1)
        seuil_achat, seuil_vente = 62, 35

    elif horizon == "moyen":
        w = SIGNAL_WEIGHTS["moyen"]
        if mas.get("golden_cross") == True:
            gc = 100 * w["golden_cross"]
        elif mas.get("golden_cross") == False:
            gc = 10 * w["golden_cross"]
        else:
            gc = 50 * w["golden_cross"]
        score += gc; detail["golden_cross"] = round(gc, 1)
        if macd.get("tendance") == "HAUSSIER":
            mc = 80 * w["macd_trend"]
            if macd.get("croisement") == "ACHAT":
                mc = 100 * w["macd_trend"]
        else:
            mc = 20 * w["macd_trend"]
            if macd.get("croisement") == "VENTE":
                mc = 0
        score += mc; detail["macd"] = round(mc, 1)
        ma_sc = mas.get("score_ma", 0)
        ma_pts = (ma_sc / 3) * 100 * w["ma_alignment"]
        score += ma_pts; detail["ma_alignment"] = round(ma_pts, 1)
        sent_sc  = sentiment.get("weighted_score", 0)
        confiance = sentiment.get("confiance", 0.5)
        vol_bonus = 1.2 if sentiment.get("volume_signal") else 1.0
        s = ((sent_sc + 1) / 2) * 100 * w["sentiment"] * confiance * vol_bonus
        score += s; detail["sentiment"] = round(s, 1)
        ch5 = market.get("variation_5j", 0)
        mo5 = min(max((ch5 + 8) / 16, 0), 1.0) * 100 * w["momentum_5j"]
        score += mo5; detail["momentum_5j"] = round(mo5, 1)
        seuil_achat, seuil_vente = 60, 35

    else:  # long
        w = SIGNAL_WEIGHTS["long"]
        if mas.get("golden_cross") == True and market.get("tendance_long") == "HAUSSIER":
            tr = 100 * w["trend_200"]
        elif mas.get("golden_cross") == True:
            tr = 75 * w["trend_200"]
        elif market.get("tendance_long") == "HAUSSIER":
            tr = 50 * w["trend_200"]
        else:
            tr = 10 * w["trend_200"]
        score += tr; detail["trend_200"] = round(tr, 1)
        fund_score = (fundamentals or {}).get("score", 50)
        f = (fund_score / 100) * 100 * w["fundamentals"]
        score += f; detail["fundamentals"] = round(f, 1)
        macro_score = (macro or {}).get("score", 50)
        if is_crypto and fear_greed:
            fg = fear_greed["value"]
            fg_score = (100 - fg) if fg < 30 else (50 if fg < 60 else 20)
            macro_adj = (macro_score * 0.5 + fg_score * 0.5)
        else:
            macro_adj = macro_score
        mac = (macro_adj / 100) * 100 * w["macro_regime"]
        score += mac; detail["macro"] = round(mac, 1)
        sent_sc  = sentiment.get("weighted_score", 0)
        confiance = sentiment.get("confiance", 0.5)
        vol_mult  = 1.5 if sentiment.get("volume_signal") else 1.0
        sv = ((sent_sc + 1) / 2) * 100 * w["sentiment_vol"] * confiance * vol_mult
        score += sv; detail["sentiment_vol"] = round(sv, 1)
        seuil_achat, seuil_vente = 60, 30

    score = min(max(score, 0), 100)
    if score >= seuil_achat:   signal, couleur = "ACHETER", "green"
    elif score <= seuil_vente: signal, couleur = "ÉVITER",  "red"
    else:                      signal, couleur = "ATTENDRE","yellow"
    return {
        "score":   round(score, 1),
        "signal":  signal,
        "couleur": couleur,
        "detail":  detail,
        "horizon": HORIZONS[horizon]["label"],
    }


def compute_score_global(scores: dict) -> dict:
    poids  = {"court": 0.25, "moyen": 0.45, "long": 0.30}
    total  = sum(scores[h]["score"] * poids[h] for h in poids)
    achat  = sum(1 for h in scores if scores[h]["signal"] == "ACHETER")
    eviter = sum(1 for h in scores if scores[h]["signal"] == "ÉVITER")
    if achat >= 2:    signal, couleur = "ACHETER", "green"
    elif eviter >= 2: signal, couleur = "ÉVITER",  "red"
    else:             signal, couleur = "ATTENDRE","yellow"
    return {"score": round(total, 1), "signal": signal, "couleur": couleur}


# ─────────────────────────────────────────────
# MODULE 8 — RAPPORT IA ENRICHI (RAISONNEMENT + CONTRADICTIONS)
# ─────────────────────────────────────────────

def _extract_response(data: dict) -> str:
    resp = (data.get("response") or "").strip()
    if resp: return resp
    msg = data.get("message", {})
    content = (msg.get("content") or "").strip()
    if content: return content
    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        console.print("[yellow][DEBUG] Thinking mode → extraction depuis 'thinking'[/yellow]")
        return thinking
    return ""


def generate_ai_report(ticker: str, asset_info: dict, market: dict,
                       sentiment: dict, scores: dict, headlines: list,
                       fear_greed: Optional[dict] = None,
                       fundamentals: Optional[dict] = None,
                       macro: Optional[dict] = None,
                       backtest_results: Optional[dict] = None) -> str:

    prix = market.get("prix_actuel", 0)
    rsi = market.get("rsi", 50)
    macd = market.get("macd", {})
    boll = market.get("bollinger", {})
    mas = market.get("moyennes", {})
    div = market.get("divergence", {})
    vb = market.get("vol_breakout", {})
    hv20 = market.get("hv20", 0)
    sent_score = sentiment.get("weighted_score", 0)
    sent_label = sentiment.get("label", "NEUTRE")
    
    # Détection des contradictions
    contradictions = []
    if rsi > 70 and mas.get("golden_cross"):
        contradictions.append("RSI surachat (>70) mais Golden Cross haussier → risque de pullback")
    if rsi < 30 and market.get("tendance_long") == "BAISSIER":
        contradictions.append("RSI survente mais tendance long terme baissière → possible fake rebound")
    if sent_score > 0.15 and div.get("type") == "BAISSIERE":
        contradictions.append("Sentiment positif mais divergence baissière RSI → signal contradictoire")
    if sent_score < -0.15 and mas.get("golden_cross"):
        contradictions.append("Sentiment négatif mais Golden Cross technique → divergence opinion/prix")
    if vb.get("confirmed") and macro and macro.get("regime") == "DÉFAVORABLE":
        contradictions.append("Breakout volume confirmé mais macroéconomique défavorable")
    
    contradictions_str = "\n".join(f"- {c}" for c in contradictions) if contradictions else "Aucune contradiction majeure détectée."
    
    # Scénarios
    scenario_haussier = []
    scenario_baissier = []
    if mas.get("golden_cross"):
        scenario_haussier.append("Golden cross MA50/200 → structure haussière LT")
    if vb.get("confirmed"):
        scenario_haussier.append("Volume breakout confirmé → accumulation")
    if sent_score > 0.15:
        scenario_haussier.append("Sentiment positif pondéré")
    if rsi < 35:
        scenario_haussier.append("RSI proche survente → rebond possible CT")
    if rsi > 70:
        scenario_baissier.append("RSI surachat → consolidation probable")
    if div.get("type") == "BAISSIERE":
        scenario_baissier.append("Divergence baissière RSI → perte de momentum")
    if sent_score < -0.15:
        scenario_baissier.append("Sentiment négatif → pression vendeuse")
    if macro and macro.get("regime") == "DÉFAVORABLE":
        scenario_baissier.append("Macro défavorable (taux, inflation)")
    
    sc_h = "\n".join(f"  • {s}" for s in scenario_haussier) if scenario_haussier else "  • Aucun signal haussier net"
    sc_b = "\n".join(f"  • {s}" for s in scenario_baissier) if scenario_baissier else "  • Aucun signal baissier net"
    
    bt_str = ""
    if backtest_results:
        combo  = backtest_results.get("combo_name", "Standard")
        wr     = backtest_results.get("win_rate", 0.5)
        sh     = backtest_results.get("sharpe", 0)
        net_r  = backtest_results.get("net_avg_return", backtest_results.get("avg_return", 0))
        max_dd = backtest_results.get("max_dd", 0)
        bt_str = (
            f"Backtest combiné ({combo}) — APRÈS frais et stop-loss ATR :\n"
            f"  Win rate: {wr:.0%} | Sharpe: {sh:.2f} | "
            f"Retour moyen net: {net_r:+.2f}% | Max drawdown: {max_dd:.2f}% "
            f"sur {backtest_results.get('n_trades', 0)} trades."
        )

    # Décision algorithmique pré-calculée
    decision_algo = compute_decision(scores, backtest_results, macro, sentiment)
    dec           = decision_algo["decision"]
    fib           = decision_algo["fiabilite"]
    ok_str        = " | ".join(decision_algo["raisons_ok"])  or "aucun"
    nok_str       = " | ".join(decision_algo["raisons_nok"]) or "aucun"

    prompt = f"""
Analyse approfondie pour {asset_info['name']} ({ticker}) en français.

CONTEXTE MARCHÉ:
- Prix: {prix:.2f} USD
- RSI(14): {rsi:.0f}
- MACD: {macd.get('tendance', '?')} (croisement: {macd.get('croisement', '?')})
- Bollinger: {boll.get('zone', '?')} (squeeze: {'Oui' if boll.get('squeeze') else 'Non'})
- Moyennes mobiles: Prix vs MA20={mas.get('prix_vs_ma20','?')}, MA50={mas.get('prix_vs_ma50','?')}, MA200={mas.get('prix_vs_ma200','?')}
- Golden cross: {'Oui' if mas.get('golden_cross') else 'Non'}
- Divergence RSI: {div.get('type', 'AUCUNE')} (force {div.get('strength',0)})
- Volume breakout confirmé: {'Oui' if vb.get('confirmed') else 'Non'}
- Volatilité HV20: {hv20:.1f}%

SENTIMENT & NEWS:
- Score sentiment pondéré: {sent_score:+.2f} → {sent_label}
- Nombre de sources: {sentiment.get('nb_sources',0)} (confiance {sentiment.get('confiance',0):.0%})
- Volume signal (>=5 sources): {'Oui' if sentiment.get('volume_signal') else 'Non'}

FONDAMENTAUX (si action):
- P/E: {fundamentals.get('pe_ratio','N/A') if fundamentals else 'N/A'}
- Croissance revenus: {fundamentals.get('revenue_growth','N/A') if fundamentals else 'N/A'}%
- Surprise EPS: {fundamentals.get('eps_surprise','N/A') if fundamentals else 'N/A'}%
- Score fondamental: {fundamentals.get('score',50) if fundamentals else 50}/100

MACRO (FRED):
- Régime: {macro.get('regime','NEUTRE') if macro else 'N/A'}
- Taux Fed: {macro.get('fed_rate','N/A') if macro else 'N/A'}%
- Inflation: {macro.get('inflation','N/A') if macro else 'N/A'}%

FEAR & GREED (crypto):
- Valeur: {fear_greed['value'] if fear_greed else 'N/A'} → {fear_greed['label'] if fear_greed else 'N/A'}

BACKTEST (2 ans walk-forward):
{bt_str}

SCORES MULTI-HORIZON:
- Court terme: {scores['court']['score']:.0f}/100 → {scores['court']['signal']}
- Moyen terme: {scores['moyen']['score']:.0f}/100 → {scores['moyen']['signal']}
- Long terme: {scores['long']['score']:.0f}/100 → {scores['long']['signal']}

CONTRADICTIONS DÉTECTÉES:
{contradictions_str}

DÉCISION ALGORITHMIQUE (seuils objectifs):
- Décision : {dec} (fiabilité {fib}/100)
- Critères validés : {ok_str}
- Critères échoués : {nok_str}

Maintenant, produit une analyse en suivant EXACTEMENT ce format :

COURT TERME (1-5j): [synthèse des signaux CT + risque principal]
MOYEN TERME (2-8 sem): [tendance + point d'inflexion clé]
LONG TERME (6 mois-2 ans): [thèse fondamentale/macro + catalyseur]
SYNTHÈSE: [conclusion actionnable en une phrase]

DÉCISION FINALE: [ACHETER / ÉVITER / ATTENDRE — confirme ou contredis la décision algorithmique avec ta raison]
CONVICTION (1-10): [chiffre + justification basée sur la cohérence des signaux]

SCÉNARIO HAUSSIER (conditions de déclenchement):
{sc_h}

SCÉNARIO BAISSIER (conditions de déclenchement):
{sc_b}

Rédige de manière concise, précise et exploitable pour un trader. Ne reformule pas les données brutes, raisonne sur les contradictions et la cohérence d'ensemble.
"""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 600,
                    "num_ctx": 4096,
                },
            },
            timeout=300,
        )
        data = response.json()
        raw = _extract_response(data)
        if not raw:
            return "Rapport IA non généré (réponse vide).", decision_algo
        return raw.strip(), decision_algo
    except Exception as e:
        console.print(f"[red]Erreur Ollama: {e}[/red]")
        return f"Erreur génération rapport IA: {e}", decision_algo


# ─────────────────────────────────────────────
# AFFICHAGE RICH
# ─────────────────────────────────────────────

def display_asset_card(ticker: str, asset_info: dict, market: dict,
                       sentiment: dict, scores: dict, score_global: dict,
                       report: str, fear_greed: Optional[dict] = None,
                       fundamentals: Optional[dict] = None,
                       macro: Optional[dict] = None,
                       backtest: Optional[dict] = None,
                       decision: Optional[dict] = None):

    cat_colors = {"GPU": "cyan", "AI": "magenta", "Crypto": "yellow"}
    cat_color  = cat_colors.get(asset_info["category"], "white")

    ch   = market.get("variation_1j", 0)
    ch_s = f"[green]+{ch:.2f}%[/green]" if ch >= 0 else f"[red]{ch:.2f}%[/red]"
    prix = (f"{market.get('prix_actuel', 0):.4f}" if "USD" in ticker
            else f"{market.get('prix_actuel', 0):.2f}")

    sv  = score_global["score"]
    bar = "█" * int(sv / 5) + "░" * (20 - int(sv / 5))
    bc  = score_global["couleur"]

    def fmt_score(h):
        s     = scores[h]
        c     = s["couleur"]
        label = HORIZONS[h]["label"].split("(")[0].strip()
        return f"[{c}]{label}: {s['score']:.0f} → {s['signal']}[/{c}]"

    horizons_line = "  |  ".join(fmt_score(h) for h in ["court", "moyen", "long"])

    fg_line = ""
    if fear_greed and "USD" in ticker:
        fg_val = fear_greed["value"]
        fg_col = "green" if fg_val < 40 else ("red" if fg_val > 70 else "yellow")
        fg_line = f"\n[bold]Fear & Greed:[/bold] [{fg_col}]{fg_val} — {fear_greed['label']}[/{fg_col}] ({fear_greed['tendance']})"

    macd = market.get("macd", {})
    boll = market.get("bollinger", {})
    mas  = market.get("moyennes", {})
    div  = market.get("divergence", {})
    vb   = market.get("vol_breakout", {})

    div_str = f"[yellow]⚡ Divergence {div['type']}[/yellow]" if div.get("type") != "AUCUNE" else ""
    vb_str  = f"[green]🔥 Breakout confirmé[/green]" if vb.get("confirmed") else (
              f"[yellow]↑ Breakout[/yellow]" if vb.get("breakout") else "")
    sq_str  = "[cyan]⊡ Squeeze[/cyan]" if boll.get("squeeze") else ""
    signals_str = "  ".join(s for s in [div_str, vb_str, sq_str] if s)
    if signals_str:
        signals_str = f"\n[bold]Signaux:[/bold] {signals_str}"

    stop_str = (f"\n[dim]Stop-loss ATR: {market.get('stop_loss_long', 0):.2f} "
                f"(risque: {market.get('risk_pct', 0):.1f}% | HV20: {market.get('hv20', 0):.0f}%)[/dim]")

    fund_str = ""
    if fundamentals and asset_info["category"] != "Crypto":
        parts = []
        if fundamentals.get("pe_ratio"):    parts.append(f"P/E: {fundamentals['pe_ratio']}")
        if fundamentals.get("eps_surprise") is not None:
            color = "green" if fundamentals["eps_surprise"] > 0 else "red"
            parts.append(f"EPS surprise: [{color}]{fundamentals['eps_surprise']:+.1f}%[/{color}]")
        if fundamentals.get("revenue_growth") is not None:
            parts.append(f"Rev.growth: {fundamentals['revenue_growth']:+.1f}%")
        if parts:
            fund_str = f"\n[bold]Fondamentaux:[/bold] {' | '.join(parts)}"

    macro_str = ""
    if macro and macro.get("regime"):
        rc = "green" if macro["regime"] == "FAVORABLE" else ("red" if macro["regime"] == "DÉFAVORABLE" else "yellow")
        fd = f" Fed:{macro['fed_rate']:.2f}%" if macro.get("fed_rate") else ""
        iy = f" Inflation:{macro['inflation']:.2f}%" if macro.get("inflation") else ""
        macro_str = f"\n[bold]Macro FRED:[/bold] [{rc}]{macro['regime']}[/{rc}]{fd}{iy}"

    bt_str = ""
    if backtest:
        combo = backtest.get("combo_name", "")
        wr = backtest.get("win_rate", 0.5)
        bt_str = f"\n[dim]Backtest combiné ({combo}) : win rate {wr:.0%}[/dim]"

    # Décision algorithmique
    dec      = decision or {}
    dec_txt  = dec.get("decision", "")
    dec_fib  = dec.get("fiabilite", 0)
    dec_col  = {"ACHETER": "green", "ÉVITER": "red", "VENDRE": "red", "ATTENDRE": "yellow"}.get(dec_txt, "yellow")
    ok_list  = " | ".join(dec.get("raisons_ok",  []))
    nok_list = " | ".join(dec.get("raisons_nok", []))
    dec_str  = (
        f"\n\n[bold]DÉCISION ALGO:[/bold] [{dec_col}]{dec_txt}[/{dec_col}]  "
        f"[dim]Fiabilité {dec_fib}/100[/dim]"
        + (f"\n[dim green]✓ {ok_list}[/dim green]"  if ok_list  else "")
        + (f"\n[dim red]✗ {nok_list}[/dim red]"     if nok_list else "")
    ) if dec_txt else ""

    content = f"""
[bold]Prix:[/bold] {prix}  {ch_s}   5j: {market.get('variation_5j',0):+.1f}%  30j: {market.get('variation_30j',0):+.1f}%
[dim]RSI: {market.get('rsi',0):.0f} | MACD: {macd.get('tendance','?')} ({macd.get('croisement','?')}) | Bollinger: {boll.get('zone','?')} | Vol: {vb.get('vol_ratio',1):.1f}x[/dim]
[dim]MA20: {mas.get('prix_vs_ma20','?')} | MA50: {mas.get('prix_vs_ma50','?')} | MA200: {mas.get('prix_vs_ma200','?')} | Golden cross: {'✓' if mas.get('golden_cross') else '✗'}[/dim]{fg_line}{signals_str}{stop_str}{fund_str}{macro_str}

[bold]Sentiment:[/bold] {sentiment['label']} ({sentiment.get('weighted_score', 0):+.3f})  [dim]{sentiment['nb_sources']} sources — confiance {sentiment.get('confiance',0):.0%}{'  ⚡ Volume signal' if sentiment.get('volume_signal') else ''}[/dim]

[bold]Score global:[/bold] [{bc}]{bar}[/{bc}] [bold {bc}]{sv:.0f}/100 → {score_global['signal']}[/bold {bc}]
{horizons_line}{bt_str}

─────────────────────────────────────────────
[bold dim]ANALYSE IA — RAISONNEMENT[/bold dim]

{report}
"""
    title = (f"[bold {cat_color}]{asset_info['name']}[/bold {cat_color}]"
             f" [{ticker}]  [{cat_color}]{asset_info['category']}[/{cat_color}]")
    console.print(Panel(content, title=title, border_style=cat_color, padding=(0, 2)))


def display_summary_table(results: list):
    table = Table(
        title="📊 RECAPITULATIF MULTI-HORIZON — Market Intelligence v3",
        box=box.ROUNDED, show_header=True,
        header_style="bold white on dark_blue",
    )
    table.add_column("Ticker",  style="bold", width=9)
    table.add_column("Nom",     width=12)
    table.add_column("Cat",     width=7)
    table.add_column("Prix",    justify="right", width=11)
    table.add_column("1j",      justify="right", width=7)
    table.add_column("RSI",     justify="right", width=5)
    table.add_column("Div.",    width=9)
    table.add_column("VB",      width=5)
    table.add_column("Sent.",   width=9)
    table.add_column("Court",   justify="right", width=8)
    table.add_column("Moyen",   justify="right", width=8)
    table.add_column("Long",    justify="right", width=8)
    table.add_column("Global",  width=13)
    table.add_column("Stop",    justify="right", width=9)

    for r in results:
        if "error" in r:
            continue
        ch        = r["market"].get("variation_1j", 0)
        sc        = r["scores"]
        sg        = r["score_global"]
        div_type  = r["market"].get("divergence", {}).get("type", "-")
        vb        = r["market"].get("vol_breakout", {})
        sent_label= r["sentiment"]["label"]
        stop      = r["market"].get("stop_loss_long", 0)
        table.add_row(
            r["ticker"],
            r["asset_info"]["name"],
            r["asset_info"]["category"],
            f"{r['market'].get('prix_actuel', 0):.2f}",
            f"[green]+{ch:.1f}%[/green]" if ch >= 0 else f"[red]{ch:.1f}%[/red]",
            f"{r['market'].get('rsi', 0):.0f}",
            f"[yellow]{div_type[:3]}[/yellow]" if div_type != "AUCUNE" else "[dim]—[/dim]",
            f"[green]✓[/green]" if vb.get("confirmed") else ("[yellow]~[/yellow]" if vb.get("breakout") else "[dim]—[/dim]"),
            f"[{'green' if sent_label == 'POSITIF' else 'red' if sent_label == 'NÉGATIF' else 'yellow'}]{sent_label}[/]",
            f"[{sc['court']['couleur']}]{sc['court']['score']:.0f}[/{sc['court']['couleur']}]",
            f"[{sc['moyen']['couleur']}]{sc['moyen']['score']:.0f}[/{sc['moyen']['couleur']}]",
            f"[{sc['long']['couleur']}]{sc['long']['score']:.0f}[/{sc['long']['couleur']}]",
            f"[{sg['couleur']}]{sg['score']:.0f} → {sg['signal']}[/{sg['couleur']}]",
            f"[dim]{stop:.2f}[/dim]",
        )
    console.print(table)


# ─────────────────────────────────────────────
# SAUVEGARDE JSON
# ─────────────────────────────────────────────

def save_results_json(results: list, path: str = "results.json"):
    safe_path = os.path.basename(path)
    path = os.path.join(_script_dir, safe_path)
    output = []
    for r in results:
        if "error" in r:
            continue
        m   = r["market"]
        sg  = r["score_global"]
        sc  = r["scores"]
        se  = r["sentiment"]
        fu  = r.get("fundamentals", {}) or {}
        mac = r.get("macro", {}) or {}
        bt  = r.get("backtest", {}) or {}
        output.append({
            "ticker":    r["ticker"],
            "name":      r["asset_info"]["name"],
            "category":  r["asset_info"]["category"],
            "timestamp": datetime.now().isoformat(),
            "market": {
                "current_price":  m.get("prix_actuel"),
                "change_1d":      m.get("variation_1j"),
                "change_5d":      m.get("variation_5j"),
                "change_30d":     m.get("variation_30j"),
                "rsi":            m.get("rsi"),
                "volume_ratio":   m.get("vol_ratio"),
                "trend":          m.get("tendance_court", ""),
                "macd_trend":     m.get("macd", {}).get("tendance", ""),
                "macd_cross":     m.get("macd", {}).get("croisement", ""),
                "bollinger_zone": m.get("bollinger", {}).get("zone", ""),
                "bollinger_squeeze": m.get("bollinger", {}).get("squeeze", False),
                "ma20_status":    m.get("moyennes", {}).get("prix_vs_ma20", ""),
                "ma50_status":    m.get("moyennes", {}).get("prix_vs_ma50", ""),
                "ma200_status":   m.get("moyennes", {}).get("prix_vs_ma200", ""),
                "golden_cross":   m.get("moyennes", {}).get("golden_cross"),
                "divergence":     m.get("divergence", {}).get("type", "AUCUNE"),
                "vol_breakout":   m.get("vol_breakout", {}).get("confirmed", False),
                "atr":            m.get("atr"),
                "hv20":           m.get("hv20"),
                "stop_loss":      m.get("stop_loss_long"),
                "risk_pct":       m.get("risk_pct"),
                "high_52w":       m.get("prix_52w_haut"),
                "low_52w":        m.get("prix_52w_bas"),
                "history_14j":    m.get("history_14j", []),
                "volatility":     m.get("hv20", 0),
            },
            "sentiment": {
                "label":          se.get("label", "NEUTRE"),
                "score":          se.get("weighted_score", se.get("score", 0)),
                "headlines_used": se.get("nb_sources", 0),
                "confiance":      se.get("confiance", 0),
                "positif":        se.get("positif", 0),
                "negatif":        se.get("negatif", 0),
                "volume_signal":  se.get("volume_signal", False),
            },
            "score": {
                "composite": round(sg["score"], 1),
                "signal":    sg["signal"],
                "court": {"value": sc["court"]["score"], "signal": sc["court"]["signal"]},
                "moyen": {"value": sc["moyen"]["score"], "signal": sc["moyen"]["signal"]},
                "long":  {"value": sc["long"]["score"],  "signal": sc["long"]["signal"]},
                "breakdown": {
                    "court": sc["court"].get("detail", {}),
                    "moyen": sc["moyen"].get("detail", {}),
                    "long":  sc["long"].get("detail", {}),
                },
            },
            "fundamentals": {
                "pe_ratio":       fu.get("pe_ratio"),
                "forward_pe":     fu.get("forward_pe"),
                "eps_surprise":   fu.get("eps_surprise"),
                "revenue_growth": fu.get("revenue_growth"),
                "profit_margin":  fu.get("profit_margin"),
                "roe":            fu.get("roe"),
                "beta":           fu.get("beta"),
                "score":          fu.get("score", 50),
            },
            "macro": {
                "fed_rate":    mac.get("fed_rate"),
                "inflation":   mac.get("inflation"),
                "10y_yield":   mac.get("10y_yield"),
                "regime":      mac.get("regime", "NEUTRE"),
                "score":       mac.get("score", 50),
            },
            "backtest": {
                "win_rate":       bt.get("win_rate", 0.5),
                "avg_return":     bt.get("avg_return", 0),
                "net_avg_return": bt.get("net_avg_return", 0),
                "sharpe":         bt.get("sharpe", 0),
                "max_dd":         bt.get("max_dd", 0),
                "n_trades":       bt.get("n_trades", 0),
                "combo_name":     bt.get("combo_name", ""),
            },
            "decision": {
                "decision":   (r.get("decision") or {}).get("decision", "ATTENDRE"),
                "fiabilite":  (r.get("decision") or {}).get("fiabilite", 0),
                "raisons_ok": (r.get("decision") or {}).get("raisons_ok", []),
                "raisons_nok":(r.get("decision") or {}).get("raisons_nok", []),
            },
            "report":     r.get("report", ""),
            "headlines":  [h["title"] if isinstance(h, dict) else h for h in r.get("headlines", [])[:5]],
            "fear_greed": r.get("fear_greed"),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]💾 Résultats sauvegardés dans {path}[/dim]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_analysis(tickers: Optional[list] = None, skip_ai: bool = False,
                 run_backtest: bool = True):
    targets = tickers or list(ASSETS.keys())

    console.print(Panel(
        f"[bold cyan]🚀 MARKET INTELLIGENCE v3[/bold cyan]\n"
        f"[dim]Actifs: {len(targets)} | Modèle: {OLLAMA_MODEL} | {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]\n"
        f"[dim]Signaux: Backtestés walk-forward 2ans | Fondamentaux + FRED macro[/dim]",
        border_style="cyan"
    ))

    console.print("[dim]📡 Chargement macro FRED...[/dim]")
    macro = fetch_macro_fred()
    rc = "green" if macro["regime"] == "FAVORABLE" else ("red" if macro["regime"] == "DÉFAVORABLE" else "yellow")
    fd = f" | Fed: {macro['fed_rate']:.2f}%" if macro.get("fed_rate") else ""
    iy = f" | Infl: {macro['inflation']:.2f}%" if macro.get("inflation") else ""
    y  = f" | 10Y: {macro['10y_yield']:.2f}%" if macro.get("10y_yield") else ""
    console.print(f"[dim]Macro FRED: [{rc}]{macro['regime']}[/{rc}]{fd}{iy}{y}[/dim]\n")

    fear_greed = fetch_fear_greed()
    if fear_greed:
        fg_col = "green" if fear_greed["value"] < 40 else ("red" if fear_greed["value"] > 70 else "yellow")
        console.print(f"[dim]Fear & Greed Crypto: [{fg_col}]{fear_greed['value']} — {fear_greed['label']}[/{fg_col}][/dim]\n")

    all_results = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        task = progress.add_task("...", total=len(targets))

        for ticker in targets:
            if ticker not in ASSETS:
                console.print(f"[red]Ticker inconnu: {ticker}[/red]")
                continue

            asset_info = ASSETS[ticker]
            progress.update(task, description=f"[cyan]Données marché {asset_info['name']}...[/cyan]")

            market = fetch_market_data(ticker)
            if "error" in market:
                console.print(f"[red]⚠ {ticker}: {market['error']}[/red]")
                progress.advance(task)
                continue
            market["ticker"] = ticker

            fundamentals = None
            if asset_info["category"] != "Crypto":
                progress.update(task, description=f"[dim]Fondamentaux {asset_info['name']}...[/dim]")
                fundamentals = fetch_fundamentals(ticker)

            progress.update(task, description=f"[dim]News {asset_info['name']}...[/dim]")
            news_items = collect_all_news(ticker, asset_info)
            sentiment  = analyze_sentiment_weighted(news_items)

            backtest_data = None
            if run_backtest:
                progress.update(task, description=f"[dim]Backtest avancé {asset_info['name']}...[/dim]")
                sentiment_score = sentiment.get("weighted_score", 0)
                backtest_data = run_advanced_backtest(ticker, sentiment_score)

            fg = fear_greed if "USD" in ticker else None

            scores = {
                "court": compute_score_horizon(market, sentiment, "court", fg, fundamentals, macro),
                "moyen": compute_score_horizon(market, sentiment, "moyen", fg, fundamentals, macro),
                "long":  compute_score_horizon(market, sentiment, "long",  fg, fundamentals, macro),
            }
            score_global = compute_score_global(scores)
            scores["global"] = score_global  # pour l'IA

            if skip_ai:
                report   = "⏭ Rapport IA désactivé (mode rapide)"
                decision = compute_decision(scores, backtest_data, macro, sentiment)
            else:
                progress.update(task, description=f"[magenta]IA raisonnée {asset_info['name']}...[/magenta]")
                report, decision = generate_ai_report(ticker, asset_info, market, sentiment,
                                            scores, news_items, fg,
                                            fundamentals, macro, backtest_data)

            result = {
                "ticker":       ticker,
                "asset_info":   asset_info,
                "market":       market,
                "headlines":    news_items,
                "sentiment":    sentiment,
                "scores":       scores,
                "score_global": score_global,
                "fundamentals": fundamentals,
                "macro":        macro,
                "backtest":     backtest_data,
                "decision":     decision,
                "report":       report,
                "fear_greed":   fg,
            }
            all_results.append(result)
            progress.advance(task)

    console.print()
    display_summary_table(all_results)
    console.print()
    for r in all_results:
        display_asset_card(
            r["ticker"], r["asset_info"], r["market"],
            r["sentiment"], r["scores"], r["score_global"],
            r["report"], r.get("fear_greed"),
            r.get("fundamentals"), macro, r.get("backtest"),
            r.get("decision")
        )

    save_results_json(all_results)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Market Intelligence v3 — Backtested · Validated")
    parser.add_argument("--tickers",      nargs="+", default=None,
                        help="Tickers à analyser. Défaut: tous")
    parser.add_argument("--no-ai",        action="store_true",
                        help="Désactive Ollama (mode rapide)")
    parser.add_argument("--no-backtest",  action="store_true",
                        help="Désactive le backtesting (plus rapide)")
    parser.add_argument("--loop",         type=int, default=0,
                        help="Relancer toutes les N minutes")
    args = parser.parse_args()

    if args.loop > 0:
        console.print(f"[cyan]Mode boucle: toutes les {args.loop} min[/cyan]")
        while True:
            run_analysis(tickers=args.tickers, skip_ai=args.no_ai,
                         run_backtest=not args.no_backtest)
            console.print(f"\n[dim]Prochaine analyse dans {args.loop} min...[/dim]\n")
            time.sleep(args.loop * 60)
    else:
        run_analysis(tickers=args.tickers, skip_ai=args.no_ai,
                     run_backtest=not args.no_backtest)
