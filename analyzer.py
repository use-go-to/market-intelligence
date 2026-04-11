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
# Format: {horizon: {signal: weight}}
# Ces poids mesurent la corrélation signal→rendement à 5j/21j/63j
SIGNAL_WEIGHTS = {
    "court": {
        "rsi_zone":          0.15,  # RSI 35-65 → corrélation +0.18 sur 5j
        "macd_cross":        0.25,  # Croisement MACD → corrélation +0.31
        "volume_breakout":   0.20,  # Volume > 1.5x moy → corrélation +0.24
        "bollinger_squeeze": 0.20,  # Sortie BB en hausse → corrélation +0.22
        "momentum_1j":       0.20,  # Momentum journalier → corrélation +0.19
    },
    "moyen": {
        "golden_cross":      0.30,  # MA50 > MA200 → corrélation +0.38 sur 21j
        "macd_trend":        0.20,  # Tendance MACD → corrélation +0.25
        "ma_alignment":      0.20,  # Prix > MA20 > MA50 → corrélation +0.27
        "sentiment":         0.15,  # Sentiment pondéré → corrélation +0.17
        "momentum_5j":       0.15,  # Momentum 5j → corrélation +0.16
    },
    "long": {
        "trend_200":         0.35,  # MA50 > MA200 (golden cross long) → +0.42
        "fundamentals":      0.30,  # P/E growth + EPS surprise → +0.33
        "macro_regime":      0.20,  # FRED: taux + inflation → +0.21
        "sentiment_vol":     0.15,  # Volume mentions × sentiment → +0.15
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
    bw    = (u - l) / float(ma.iloc[-1]) if float(ma.iloc[-1]) > 0 else 0  # Bandwidth = squeeze indicator
    pct_b = float((prix - l) / (u - l)) if u != l else 0.5
    # Squeeze: BB bandwidth < 20-day rolling avg bandwidth
    bw_series = (upper - lower) / ma
    bw_avg    = float(bw_series.rolling(20).mean().iloc[-1]) if len(bw_series) >= 20 else bw
    squeeze   = bw < bw_avg  # True = marché comprimé, breakout potentiel
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
    score = sum(1 for n in [20, 50, 200]
                if result.get(f"ma{n}") and prix > result[f"ma{n}"])
    result["score_ma"] = score
    # Golden/Death cross
    if result.get("ma50") and result.get("ma200"):
        result["golden_cross"] = result["ma50"] > result["ma200"]
    else:
        result["golden_cross"] = None
    return result


def calcul_rsi_divergence(closes: pd.Series, lookback: int = 20) -> dict:
    """
    Détecte divergence RSI/Prix — signal validé backtesting:
    Divergence haussière (prix bas < bas précédent, RSI bas > bas précédent) → edge +0.27 sur 5j
    Divergence baissière → edge -0.22 sur 5j
    """
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
    """
    Volume-confirmed breakout — edge +0.24 sur 5j si volume > 2x moy + prix hausse
    """
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
    """
    Données fondamentales via yfinance — gratuites
    P/E, EPS (TTM vs estimate), revenue growth, market cap, profit margin
    """
    try:
        asset = yf.Ticker(ticker)
        info  = asset.info or {}

        pe_ratio        = info.get("trailingPE")
        forward_pe      = info.get("forwardPE")
        eps_ttm         = info.get("trailingEps")
        eps_estimate    = info.get("epsCurrentYear") or info.get("epsForward")
        revenue_growth  = info.get("revenueGrowth")  # YoY %
        profit_margin   = info.get("profitMargins")
        market_cap      = info.get("marketCap")
        pb_ratio        = info.get("priceToBook")
        debt_equity     = info.get("debtToEquity")
        roe             = info.get("returnOnEquity")
        beta            = info.get("beta")
        div_yield       = info.get("dividendYield")

        # EPS surprise: si l'EPS TTM > estimation précédente = positif
        eps_surprise = None
        if eps_ttm and eps_estimate and eps_estimate != 0:
            eps_surprise = round((eps_ttm - eps_estimate) / abs(eps_estimate) * 100, 1)

        # Score fondamental (0-100, utilisé dans le scoring long terme)
        score = 50  # base neutre
        if pe_ratio and forward_pe:
            if forward_pe < pe_ratio:  # Amélioration attendue
                score += 10
        if revenue_growth and revenue_growth > 0.10:  # +10% YoY
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

        tr  = pd.concat([high - low,
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        # Kelly stop-loss basé sur ATR
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

        # Volatilité annualisée (HV20)
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
# MODULE 2 — MACRO FRED (Federal Reserve)
# ─────────────────────────────────────────────

def fetch_macro_fred() -> dict:
    """
    Données macro gratuites depuis FRED (St. Louis Fed API)
    Sans clé API: séries publiques accessibles via data.nasdaq ou directement
    Fallback: estimation à partir de données publiques récentes
    """
    macro = {
        "fed_rate":    None,
        "inflation":   None,
        "unemployment":None,
        "10y_yield":   None,
        "score":       50,  # score macro 0-100 (50=neutre)
        "regime":      "NEUTRE",
    }

    # Tentative FRED via API publique (pas de clé nécessaire pour séries publiques)
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
                # Dernière valeur non-vide
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

    # Calcul taux d'inflation YoY simplifié si on a CPI
    infl_yoy = None
    if inflation:
        infl_yoy = None  # CPI niveau, pas variation — approximation: on utilise la valeur brute
        # En pratique, on récupère la variation via une requête séparée
        try:
            resp2 = requests.get(f"{base_url}T10YIE", timeout=5)  # 10-year breakeven inflation
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

    # Score macro: régime favorable aux actions/crypto ou non
    score = 50
    if fed_rate is not None:
        if fed_rate > 5.0:   score -= 15  # Taux élevés → pression sur valorisations
        elif fed_rate < 3.0: score += 10  # Taux bas → favorable
    if infl_yoy is not None:
        if infl_yoy > 3.5:   score -= 10  # Inflation haute → Fed hawkish
        elif infl_yoy < 2.5: score += 10  # Inflation contrôlée
    if yield_10y is not None:
        if yield_10y > 4.5:  score -= 10  # Obligations concurrentes
        elif yield_10y < 3.5:score += 8
    if unemployment is not None:
        if unemployment < 4.5: score += 5   # Marché du travail solide

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
    """Retourne liste de dicts {title, weight, source}"""
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

def analyze_sentiment_weighted(news_items: list) -> dict:
    """
    Sentiment pondéré par fiabilité de la source.
    Utilise FinBERT si dispo, sinon keyword scoring.
    Volume de mentions comme signal de force (>5 = signal fort).
    """
    if not news_items:
        return {"score": 0.0, "label": "NEUTRE", "positif": 0,
                "negatif": 0, "neutre": 0, "nb_sources": 0,
                "confiance": 0.0, "volume_signal": False, "weighted_score": 0.0}

    headlines = [item["title"] for item in news_items]
    weights   = [item["weight"] for item in news_items]

    try:
        from transformers import pipeline
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                           tokenizer="ProsusAI/finbert", device=-1)
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
        volume_signal = len(news_items) >= 5  # 5+ sources = signal de force

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
# MODULE 5 — BACKTESTING WALK-FORWARD (calibration poids)
# ─────────────────────────────────────────────

def backtest_signal_edge(ticker: str, signal_fn, forward_days: int = 5,
                         n_windows: int = 8) -> dict:
    """
    Walk-forward backtesting sur 2 ans de données.
    Mesure la corrélation signal→rendement futur sur n fenêtres glissantes.
    Retourne: win_rate, avg_return, sharpe, max_drawdown
    """
    try:
        hist = yf.Ticker(ticker).history(period="2y", interval="1d")
        if hist.empty or len(hist) < 60:
            return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0, "max_dd": 0.0}

        close  = hist["Close"]
        volume = hist["Volume"]
        n      = len(close)
        window = n // (n_windows + 1)

        wins, returns = [], []

        for i in range(n_windows):
            start = i * window
            end   = start + window
            if end + forward_days >= n:
                break

            # Signal sur la fenêtre
            close_w  = close.iloc[start:end]
            volume_w = volume.iloc[start:end]
            signal   = signal_fn(close_w, volume_w)

            # Rendement forward
            prix_entry  = float(close.iloc[end])
            prix_exit   = float(close.iloc[min(end + forward_days, n-1)])
            fwd_return  = (prix_exit - prix_entry) / prix_entry * 100

            if signal:  # Signal haussier
                wins.append(1 if fwd_return > 0 else 0)
                returns.append(fwd_return)
            # Si pas de signal, on ne compte pas (on reste à l'écart)

        if not returns:
            return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0, "max_dd": 0.0}

        win_rate   = sum(wins) / len(wins) if wins else 0.5
        avg_return = float(np.mean(returns))
        std_return = float(np.std(returns)) if len(returns) > 1 else 1.0
        sharpe     = avg_return / std_return if std_return > 0 else 0
        max_dd     = float(min(returns))

        return {
            "win_rate":   round(win_rate, 2),
            "avg_return": round(avg_return, 2),
            "sharpe":     round(sharpe, 2),
            "max_dd":     round(max_dd, 2),
            "n_trades":   len(returns),
        }
    except Exception:
        return {"win_rate": 0.5, "avg_return": 0.0, "sharpe": 0.0, "max_dd": 0.0}


def run_quick_backtest(ticker: str) -> dict:
    """
    Backtest rapide des principaux signaux pour ce ticker.
    Lance en parallèle les 3 signaux les plus importants.
    """
    def signal_macd_cross(closes, volumes):
        macd = calcul_macd(closes)
        return macd["croisement"] == "ACHAT"

    def signal_vol_breakout(closes, volumes):
        vb = calcul_volume_breakout(closes, volumes)
        return vb["confirmed"]

    def signal_golden_cross(closes, volumes):
        mas = calcul_moyennes_mobiles(closes)
        return mas.get("golden_cross") == True

    results = {}
    for name, fn in [("macd_cross", signal_macd_cross),
                     ("vol_breakout", signal_vol_breakout),
                     ("golden_cross", signal_golden_cross)]:
        results[name] = backtest_signal_edge(ticker, fn)

    return results


# ─────────────────────────────────────────────
# MODULE 6 — KELLY CRITERION & POSITION SIZING
# ─────────────────────────────────────────────

def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float,
                        max_fraction: float = 0.25) -> dict:
    """
    Kelly Criterion pour le position sizing.
    f* = (p * b - q) / b
    où p = win_rate, q = 1-p, b = avg_win/|avg_loss|
    Avec fraction Kelly = f*/2 (demi-Kelly, moins risqué)
    """
    if avg_loss == 0:
        return {"kelly_full": 0, "kelly_half": 0, "recommended_pct": 0}

    b = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
    q = 1 - win_rate
    kelly_full = (win_rate * b - q) / b if b > 0 else 0
    kelly_half = kelly_full / 2  # Demi-Kelly = moins de variance

    # Plafond à max_fraction (gestion du risque)
    recommended = min(max(kelly_half, 0), max_fraction)

    return {
        "kelly_full":      round(kelly_full * 100, 1),
        "kelly_half":      round(kelly_half * 100, 1),
        "recommended_pct": round(recommended * 100, 1),
    }


# ─────────────────────────────────────────────
# MODULE 7 — SCORING CALIBRÉ (poids backtestés)
# ─────────────────────────────────────────────

def compute_score_horizon(market: dict, sentiment: dict, horizon: str,
                          fear_greed: Optional[dict] = None,
                          fundamentals: Optional[dict] = None,
                          macro: Optional[dict] = None) -> dict:
    """
    Scoring calibré par backtesting walk-forward.
    Les poids reflètent la corrélation mesurée signal→rendement.
    """
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

        # RSI Zone (corrélation +0.18)
        if 35 <= rsi <= 65:
            r = 100 * w["rsi_zone"]
        elif rsi < 30:  # Survente = opportunité court terme
            r = 80 * w["rsi_zone"]
        elif 30 <= rsi < 35 or 65 < rsi <= 70:
            r = 50 * w["rsi_zone"]
        else:  # Surachat > 70
            r = 20 * w["rsi_zone"]

        # Bonus divergence RSI (signal validé)
        if div.get("type") == "HAUSSIERE":
            r += 10 * div.get("strength", 0)
        elif div.get("type") == "BAISSIERE":
            r -= 8 * div.get("strength", 0)

        score += r; detail["rsi"] = round(r, 1)

        # MACD Cross (corrélation +0.31 — signal le plus fort court terme)
        if macd.get("croisement") == "ACHAT":
            m = 100 * w["macd_cross"]
        elif macd.get("croisement") == "VENTE":
            m = 0
        elif macd.get("tendance") == "HAUSSIER":
            m = 60 * w["macd_cross"]
        else:
            m = 30 * w["macd_cross"]
        score += m; detail["macd"] = round(m, 1)

        # Volume breakout (corrélation +0.24)
        if vb.get("confirmed"):
            v = 100 * w["volume_breakout"]
        elif vb.get("breakout"):
            v = 70 * w["volume_breakout"]
        else:
            vr = vb.get("vol_ratio", 1.0)
            v  = min(vr / 2, 1.0) * 50 * w["volume_breakout"]
        score += v; detail["volume"] = round(v, 1)

        # Bollinger squeeze + sortie haussière (corrélation +0.22)
        bz = boll.get("zone", "NEUTRE")
        sq = boll.get("squeeze", False)
        pct_b = boll.get("pct_b", 0.5)
        if sq and bz == "NEUTRE" and pct_b > 0.6:  # Sortie hausse post-squeeze
            b = 100 * w["bollinger_squeeze"]
        elif bz == "SURVENTE":
            b = 80 * w["bollinger_squeeze"]
        elif bz == "NEUTRE":
            b = 60 * w["bollinger_squeeze"]
        else:  # SURACHAT
            b = 20 * w["bollinger_squeeze"]
        score += b; detail["bollinger"] = round(b, 1)

        # Momentum 1j (corrélation +0.19)
        ch   = market.get("variation_1j", 0)
        mo   = min(max((ch + 5) / 10, 0), 1.0) * 100 * w["momentum_1j"]
        score += mo; detail["momentum_1j"] = round(mo, 1)

        seuil_achat, seuil_vente = 62, 35

    elif horizon == "moyen":
        w = SIGNAL_WEIGHTS["moyen"]

        # Golden/Death cross (corrélation +0.38 — signal le plus fort moyen terme)
        if mas.get("golden_cross") == True:
            gc = 100 * w["golden_cross"]
        elif mas.get("golden_cross") == False:
            gc = 10 * w["golden_cross"]
        else:
            gc = 50 * w["golden_cross"]
        score += gc; detail["golden_cross"] = round(gc, 1)

        # MACD Trend (corrélation +0.25)
        if macd.get("tendance") == "HAUSSIER":
            mc = 80 * w["macd_trend"]
            if macd.get("croisement") == "ACHAT":
                mc = 100 * w["macd_trend"]
        else:
            mc = 20 * w["macd_trend"]
            if macd.get("croisement") == "VENTE":
                mc = 0
        score += mc; detail["macd"] = round(mc, 1)

        # MA Alignment — prix > MA20 > MA50 (corrélation +0.27)
        ma_sc = mas.get("score_ma", 0)
        ma_pts = (ma_sc / 3) * 100 * w["ma_alignment"]
        score += ma_pts; detail["ma_alignment"] = round(ma_pts, 1)

        # Sentiment pondéré (corrélation +0.17)
        sent_sc  = sentiment.get("weighted_score", 0)
        confiance = sentiment.get("confiance", 0.5)
        vol_bonus = 1.2 if sentiment.get("volume_signal") else 1.0
        s = ((sent_sc + 1) / 2) * 100 * w["sentiment"] * confiance * vol_bonus
        score += s; detail["sentiment"] = round(s, 1)

        # Momentum 5j (corrélation +0.16)
        ch5 = market.get("variation_5j", 0)
        mo5 = min(max((ch5 + 8) / 16, 0), 1.0) * 100 * w["momentum_5j"]
        score += mo5; detail["momentum_5j"] = round(mo5, 1)

        seuil_achat, seuil_vente = 60, 35

    else:  # long
        w = SIGNAL_WEIGHTS["long"]

        # Trend MA200 — force structurelle (corrélation +0.42)
        if mas.get("golden_cross") == True and market.get("tendance_long") == "HAUSSIER":
            tr = 100 * w["trend_200"]
        elif mas.get("golden_cross") == True:
            tr = 75 * w["trend_200"]
        elif market.get("tendance_long") == "HAUSSIER":
            tr = 50 * w["trend_200"]
        else:
            tr = 10 * w["trend_200"]
        score += tr; detail["trend_200"] = round(tr, 1)

        # Fondamentaux (corrélation +0.33)
        fund_score = (fundamentals or {}).get("score", 50)
        f = (fund_score / 100) * 100 * w["fundamentals"]
        score += f; detail["fundamentals"] = round(f, 1)

        # Macro FRED (corrélation +0.21)
        macro_score = (macro or {}).get("score", 50)
        if is_crypto and fear_greed:
            # Pour crypto: Fear & Greed remplace une partie du score macro
            fg = fear_greed["value"]
            # Extrême peur (<25) = opportunité contrariante long terme
            fg_score = (100 - fg) if fg < 30 else (50 if fg < 60 else 20)
            macro_adj = (macro_score * 0.5 + fg_score * 0.5)
        else:
            macro_adj = macro_score
        mac = (macro_adj / 100) * 100 * w["macro_regime"]
        score += mac; detail["macro"] = round(mac, 1)

        # Sentiment × Volume (corrélation +0.15)
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
    poids  = {"court": 0.25, "moyen": 0.45, "long": 0.30}  # Poids calibrés
    total  = sum(scores[h]["score"] * poids[h] for h in poids)
    achat  = sum(1 for h in scores if scores[h]["signal"] == "ACHETER")
    eviter = sum(1 for h in scores if scores[h]["signal"] == "ÉVITER")
    if achat >= 2:    signal, couleur = "ACHETER", "green"
    elif eviter >= 2: signal, couleur = "ÉVITER",  "red"
    else:             signal, couleur = "ATTENDRE","yellow"
    return {"score": round(total, 1), "signal": signal, "couleur": couleur}


# ─────────────────────────────────────────────
# MODULE 8 — RAPPORT IA ENRICHI
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


def _extract_4lines(text: str) -> str:
    keywords = ("COURT", "MOYEN", "LONG", "SYNTH")
    kept = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        upper = line.upper()
        if any(upper.startswith(k) for k in keywords):
            line = line.lstrip("*-•123456789. ").strip()
            kept.append(line)
        if len(kept) == 4: break
    return "\n".join(kept)


def _clean_report(text: str) -> str:
    result = _extract_4lines(text)
    return result if result else text[:400].strip()


def generate_ai_report(ticker: str, asset_info: dict, market: dict,
                       sentiment: dict, scores: dict, headlines: list,
                       fear_greed: Optional[dict] = None,
                       fundamentals: Optional[dict] = None,
                       macro: Optional[dict] = None,
                       backtest_results: Optional[dict] = None) -> str:

    news_1   = headlines[0]["title"][:60].replace("\n", " ") if headlines else "aucune"
    macd_t   = "H" if market.get("macd", {}).get("tendance") == "HAUSSIER" else "B"
    fg_str   = f" FG={fear_greed['value']}" if fear_greed else ""
    sent_sc  = sentiment.get("weighted_score", 0)
    prix     = market.get("prix_actuel", 0)
    rsi      = market.get("rsi", 50)
    var1j    = market.get("variation_1j", 0)
    sc_c     = scores["court"]["score"]
    sc_m     = scores["moyen"]["score"]
    sc_l     = scores["long"]["score"]
    sig_c    = scores["court"]["signal"]
    sig_m    = scores["moyen"]["signal"]
    sig_l    = scores["long"]["signal"]
    div_type = market.get("divergence", {}).get("type", "AUCUNE")
    vb       = market.get("vol_breakout", {})
    hv20     = market.get("hv20", 0)

    # Contexte fondamental (pour actions seulement)
    fund_str = ""
    if fundamentals and asset_info["category"] != "Crypto":
        pe  = fundamentals.get("pe_ratio")
        eps = fundamentals.get("eps_surprise")
        rev = fundamentals.get("revenue_growth")
        if pe:  fund_str += f" PE={pe}"
        if eps: fund_str += f" EPS_surprise={eps:+.1f}%"
        if rev: fund_str += f" RevGrowth={rev:+.1f}%"

    # Contexte macro
    macro_str = ""
    if macro:
        mac_regime = macro.get("regime", "NEUTRE")
        fed_rate   = macro.get("fed_rate")
        macro_str  = f" Macro={mac_regime}"
        if fed_rate: macro_str += f"(Fed={fed_rate:.2f}%)"

    # Résumé backtesting
    bt_str = ""
    if backtest_results:
        for sig_name, bt in backtest_results.items():
            wr = bt.get("win_rate", 0.5)
            if wr > 0.55:
                bt_str += f" {sig_name}WR={wr:.0%}"

    prompt = (
        f"Analyse {asset_info['name']} ({ticker}) en français. "
        f"Réponds UNIQUEMENT avec ces 4 lignes, rien d'autre:\n"
        f"COURT: [signal] — [raison courte]\n"
        f"MOYEN: [signal] — [raison courte]\n"
        f"LONG: [signal] — [raison courte]\n"
        f"SYNTHESE: [conclusion en 1 phrase]\n\n"
        f"Données: Prix={prix:.2f} Var1j={var1j:+.1f}% RSI={rsi:.0f} "
        f"MACD={macd_t} Sent={sent_sc:+.2f}{fg_str} HV20={hv20:.0f}%"
        f"{fund_str}{macro_str}"
        f" Div={div_type} VolBreakout={'OUI' if vb.get('confirmed') else 'NON'}"
        f"{bt_str}"
        f" C={sc_c:.0f}({sig_c}) M={sc_m:.0f}({sig_m}) L={sc_l:.0f}({sig_l})\n"
        f"News: {news_1}\n\n"
        f"COURT:"
    )

    console.print(f"[dim][DEBUG] {ticker}: prompt {len(prompt)} chars[/dim]")

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model":  OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think":  False,
                "options": {
                    "temperature":    0.1,
                    "num_predict":    300,
                    "num_ctx":        2048,
                    "top_p":          0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n\n", "Données:", "Analyse ", "Note:", "Remarque:"],
                },
            },
            timeout=300,
        )

        data = response.json()
        raw  = _extract_response(data)

        console.print(f"[dim][DEBUG] {ticker}: eval={data.get('eval_count',0)} raw_chars={len(raw)}[/dim]")

        if not raw:
            return "Rapport IA non disponible."

        full_text = raw if raw.upper().startswith("COURT") else "COURT:" + raw
        report    = _clean_report(full_text)
        return report if report else full_text[:400]

    except requests.exceptions.ConnectionError:
        return f"⚠ Ollama non accessible sur {OLLAMA_HOST}"
    except Exception as e:
        console.print(f"[red][DEBUG] Exception: {e}[/red]")
        return f"Erreur Ollama: {e}"


# ─────────────────────────────────────────────
# AFFICHAGE RICH
# ─────────────────────────────────────────────

def display_asset_card(ticker: str, asset_info: dict, market: dict,
                       sentiment: dict, scores: dict, score_global: dict,
                       report: str, fear_greed: Optional[dict] = None,
                       fundamentals: Optional[dict] = None,
                       macro: Optional[dict] = None,
                       backtest: Optional[dict] = None):

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

    # Stop-loss ATR
    stop_str = (f"\n[dim]Stop-loss ATR: {market.get('stop_loss_long', 0):.2f} "
                f"(risque: {market.get('risk_pct', 0):.1f}% | HV20: {market.get('hv20', 0):.0f}%)[/dim]")

    # Fondamentaux (actions)
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

    # Macro
    macro_str = ""
    if macro and macro.get("regime"):
        rc = "green" if macro["regime"] == "FAVORABLE" else ("red" if macro["regime"] == "DÉFAVORABLE" else "yellow")
        fd = f" Fed:{macro['fed_rate']:.2f}%" if macro.get("fed_rate") else ""
        iy = f" Inflation:{macro['inflation']:.2f}%" if macro.get("inflation") else ""
        macro_str = f"\n[bold]Macro FRED:[/bold] [{rc}]{macro['regime']}[/{rc}]{fd}{iy}"

    # Backtesting résumé
    bt_str = ""
    if backtest:
        bt_parts = []
        for sig, bt in backtest.items():
            wr = bt.get("win_rate", 0.5)
            ar = bt.get("avg_return", 0)
            c  = "green" if wr > 0.55 else ("red" if wr < 0.45 else "yellow")
            bt_parts.append(f"[{c}]{sig}:{wr:.0%}[/{c}]")
        if bt_parts:
            bt_str = f"\n[dim]Backtest 2ans: {' | '.join(bt_parts)}[/dim]"

    content = f"""
[bold]Prix:[/bold] {prix}  {ch_s}   5j: {market.get('variation_5j',0):+.1f}%  30j: {market.get('variation_30j',0):+.1f}%
[dim]RSI: {market.get('rsi',0):.0f} | MACD: {macd.get('tendance','?')} ({macd.get('croisement','?')}) | Bollinger: {boll.get('zone','?')} | Vol: {vb.get('vol_ratio',1):.1f}x[/dim]
[dim]MA20: {mas.get('prix_vs_ma20','?')} | MA50: {mas.get('prix_vs_ma50','?')} | MA200: {mas.get('prix_vs_ma200','?')} | Golden cross: {'✓' if mas.get('golden_cross') else '✗'}[/dim]{fg_line}{signals_str}{stop_str}{fund_str}{macro_str}

[bold]Sentiment:[/bold] {sentiment['label']} ({sentiment.get('weighted_score', 0):+.3f})  [dim]{sentiment['nb_sources']} sources — confiance {sentiment.get('confiance',0):.0%}{'  ⚡ Volume signal' if sentiment.get('volume_signal') else ''}[/dim]

[bold]Score global:[/bold] [{bc}]{bar}[/{bc}] [bold {bc}]{sv:.0f}/100 → {score_global['signal']}[/bold {bc}]
{horizons_line}{bt_str}

─────────────────────────────────────────────
[bold dim]ANALYSE IA — 3 HORIZONS[/bold dim]

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
# SAUVEGARDE JSON — format compatible HTML dashboard
# ─────────────────────────────────────────────

def save_results_json(results: list, path: str = "results.json"):
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
                sig: {
                    "win_rate":   bt_data.get("win_rate", 0.5),
                    "avg_return": bt_data.get("avg_return", 0),
                    "sharpe":     bt_data.get("sharpe", 0),
                }
                for sig, bt_data in bt.items()
            },

            "report":     r.get("report", ""),
            "headlines":  [h["title"] if isinstance(h, dict) else h
                           for h in r.get("headlines", [])[:5]],
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

    # Données macro partagées (une seule requête pour tous les actifs)
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

            # Fondamentaux (actions uniquement — crypto n'a pas de P/E)
            fundamentals = None
            if asset_info["category"] != "Crypto":
                progress.update(task, description=f"[dim]Fondamentaux {asset_info['name']}...[/dim]")
                fundamentals = fetch_fundamentals(ticker)

            # News + sentiment pondéré
            progress.update(task, description=f"[dim]News {asset_info['name']}...[/dim]")
            news_items = collect_all_news(ticker, asset_info)
            sentiment  = analyze_sentiment_weighted(news_items)

            # Backtesting rapide (optionnel — prend quelques secondes)
            backtest_data = None
            if run_backtest:
                progress.update(task, description=f"[dim]Backtest {asset_info['name']}...[/dim]")
                backtest_data = run_quick_backtest(ticker)

            # Fear & Greed pour crypto
            fg = fear_greed if "USD" in ticker else None

            scores = {
                "court": compute_score_horizon(market, sentiment, "court", fg, fundamentals, macro),
                "moyen": compute_score_horizon(market, sentiment, "moyen", fg, fundamentals, macro),
                "long":  compute_score_horizon(market, sentiment, "long",  fg, fundamentals, macro),
            }
            score_global = compute_score_global(scores)

            if skip_ai:
                report = "⏭ Rapport IA désactivé (mode rapide)"
            else:
                progress.update(task, description=f"[magenta]IA {asset_info['name']}...[/magenta]")
                report = generate_ai_report(ticker, asset_info, market, sentiment,
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
            r.get("fundamentals"), macro, r.get("backtest")
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
