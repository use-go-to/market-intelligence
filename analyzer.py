"""
╔══════════════════════════════════════════════════════════════════════╗
║        MARKET INTELLIGENCE v2 — Multi-Horizon + Multi-Sources        ║
║        Stocks: NVDA, AMD, INTC, MSFT, GOOGL, META, TSLA             ║
║        Crypto: BTC, ETH, SOL                                         ║
║        Sources: yfinance + NewsAPI + RSS + Fear&Greed + FinBERT      ║
║        IA: Ollama/Gemma4 — 3 horizons (court/moyen/long terme)       ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALLATION:
    pip install yfinance requests transformers torch pandas numpy
    pip install newsapi-python python-dotenv rich colorama feedparser

CONFIG (.env):
    NEWS_API_KEY=your_key   (gratuit sur newsapi.org)
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=gemma4:latest
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

# Sources RSS gratuites — finances + crypto
RSS_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://www.investing.com/rss/news.rss",
]

# 3 horizons d'analyse avec leurs paramètres
HORIZONS = {
    "court":  {"label": "Court terme (1–5 jours)",    "period": "5d",  "interval": "1h",  "ma_short": 10,  "ma_long": 20},
    "moyen":  {"label": "Moyen terme (2–8 semaines)", "period": "3mo", "interval": "1d",  "ma_short": 20,  "ma_long": 50},
    "long":   {"label": "Long terme (6 mois–2 ans)",  "period": "2y",  "interval": "1wk", "ma_short": 50,  "ma_long": 200},
}

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

console = Console()


# ─────────────────────────────────────────────
# MODULE 1 — DONNÉES MARCHÉ MULTI-HORIZON
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
    return {
        "macd":    float(macd.iloc[-1]),
        "signal":  float(signal.iloc[-1]),
        "histo":   float(histo.iloc[-1]),
        "tendance": "HAUSSIER" if macd.iloc[-1] > signal.iloc[-1] else "BAISSIER",
        "croisement": "ACHAT" if histo.iloc[-1] > 0 and histo.iloc[-2] <= 0
                      else ("VENTE" if histo.iloc[-1] < 0 and histo.iloc[-2] >= 0 else "NEUTRE")
                      if len(histo) > 1 else "NEUTRE",
    }


def calcul_bollinger(closes: pd.Series, periode: int = 20) -> dict:
    ma   = closes.rolling(periode).mean()
    std  = closes.rolling(periode).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    prix  = closes.iloc[-1]
    pct_b = float((prix - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) if upper.iloc[-1] != lower.iloc[-1] else 0.5
    if prix > upper.iloc[-1]:
        zone = "SURACHAT"
    elif prix < lower.iloc[-1]:
        zone = "SURVENTE"
    else:
        zone = "NEUTRE"
    return {"upper": float(upper.iloc[-1]), "lower": float(lower.iloc[-1]),
            "ma": float(ma.iloc[-1]), "pct_b": round(pct_b, 3), "zone": zone}


def calcul_moyennes_mobiles(closes: pd.Series) -> dict:
    prix  = closes.iloc[-1]
    result = {}
    for n in [20, 50, 200]:
        if len(closes) >= n:
            ma = float(closes.rolling(n).mean().iloc[-1])
            result[f"ma{n}"] = ma
            result[f"prix_vs_ma{n}"] = "AU DESSUS" if prix > ma else "EN DESSOUS"
        else:
            result[f"ma{n}"] = None
            result[f"prix_vs_ma{n}"] = "N/A"
    # Score 0-3 : combien de MA le prix dépasse
    score = sum(1 for n in [20, 50, 200] if result.get(f"ma{n}") and prix > result[f"ma{n}"])
    result["score_ma"] = score
    return result


def fetch_market_data(ticker: str) -> dict:
    """Récupère données pour les 3 horizons + indicateurs complets."""
    try:
        asset = yf.Ticker(ticker)

        # Données principales sur 2 ans journalières
        hist = asset.history(period="2y", interval="1d")
        if hist.empty:
            return {"error": f"Pas de données pour {ticker}"}

        close  = hist["Close"]
        volume = hist["Volume"]

        prix_actuel = float(close.iloc[-1])
        prix_hier   = float(close.iloc[-2]) if len(close) > 1 else prix_actuel
        variation_1j = ((prix_actuel - prix_hier) / prix_hier) * 100

        # Indicateurs techniques
        rsi_14       = calcul_rsi(close, 14)
        macd_data    = calcul_macd(close)
        bollinger    = calcul_bollinger(close)
        mas          = calcul_moyennes_mobiles(close)

        # Volatilité ATR
        high = hist["High"]
        low  = hist["Low"]
        tr   = pd.concat([high - low,
                          (high - close.shift()).abs(),
                          (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr  = float(tr.rolling(14).mean().iloc[-1])

        # Volume
        vol_moy     = float(volume.rolling(20).mean().iloc[-1])
        vol_actuel  = float(volume.iloc[-1])
        vol_ratio   = vol_actuel / vol_moy if vol_moy > 0 else 1.0

        # Tendance multi-horizon (SMA)
        tendance_court = "HAUSSIER" if len(close) >= 10 and close.rolling(5).mean().iloc[-1] > close.rolling(10).mean().iloc[-1] else "BAISSIER"
        tendance_moyen = "HAUSSIER" if mas.get("ma20") and mas.get("ma50") and mas["ma20"] > mas["ma50"] else "BAISSIER"
        tendance_long  = "HAUSSIER" if mas.get("ma50") and mas.get("ma200") and mas["ma50"] > mas["ma200"] else "BAISSIER"

        # Variation sur différentes périodes
        var_5j   = ((prix_actuel - float(close.iloc[-5]))  / float(close.iloc[-5]))  * 100 if len(close) > 5  else 0
        var_30j  = ((prix_actuel - float(close.iloc[-21])) / float(close.iloc[-21])) * 100 if len(close) > 21 else 0
        var_90j  = ((prix_actuel - float(close.iloc[-63])) / float(close.iloc[-63])) * 100 if len(close) > 63 else 0
        var_1an  = ((prix_actuel - float(close.iloc[-252]))/ float(close.iloc[-252]))* 100 if len(close) > 252 else 0

        return {
            "ticker":          ticker,
            "prix_actuel":     prix_actuel,
            "variation_1j":    round(variation_1j, 2),
            "variation_5j":    round(var_5j, 2),
            "variation_30j":   round(var_30j, 2),
            "variation_90j":   round(var_90j, 2),
            "variation_1an":   round(var_1an, 2),
            "rsi":             round(rsi_14, 1),
            "macd":            macd_data,
            "bollinger":       bollinger,
            "moyennes":        mas,
            "atr":             round(atr, 4),
            "vol_ratio":       round(vol_ratio, 2),
            "tendance_court":  tendance_court,
            "tendance_moyen":  tendance_moyen,
            "tendance_long":   tendance_long,
            "prix_52w_haut":   float(close.tail(252).max()),
            "prix_52w_bas":    float(close.tail(252).min()),
            "history_14j":     close.tail(14).tolist(),
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# MODULE 2 — SOURCES DE NEWS MULTIPLES
# ─────────────────────────────────────────────

def fetch_rss_news(rss_keys: list[str]) -> list[str]:
    """Scrape les flux RSS gratuits et filtre par mots-clés."""
    try:
        import feedparser
    except ImportError:
        return []

    articles = []
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                titre = entry.get("title", "")
                if any(k.lower() in titre.lower() for k in rss_keys):
                    articles.append(titre)
        except Exception:
            continue
    return articles[:10]


def fetch_newsapi(keywords: list[str]) -> list[str]:
    """Récupère news via NewsAPI (source payante mais plus précise)."""
    if not NEWS_API_KEY:
        return []
    try:
        from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        query = " OR ".join(keywords[:4])
        resp  = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "from": from_date, "sortBy": "publishedAt",
            "language": "en", "pageSize": 20, "apiKey": NEWS_API_KEY,
        }, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return []
        return [a["title"] for a in data.get("articles", [])
                if a.get("title") and "[Removed]" not in a["title"]][:15]
    except Exception:
        return []


def fetch_yfinance_news(ticker: str) -> list[str]:
    """Récupère les news directement depuis yfinance."""
    try:
        asset = yf.Ticker(ticker)
        news  = asset.news
        return [n.get("content", {}).get("title", "") for n in (news or [])
                if n.get("content", {}).get("title")][:10]
    except Exception:
        return []


def fetch_fear_greed() -> Optional[dict]:
    """Fear & Greed Index crypto (Alternative.me) — gratuit."""
    try:
        r    = requests.get("https://api.alternative.me/fng/?limit=7", timeout=5)
        data = r.json()["data"]
        actuel = data[0]
        hier   = data[1] if len(data) > 1 else actuel
        return {
            "value":     int(actuel["value"]),
            "label":     actuel["value_classification"],
            "hier":      int(hier["value"]),
            "tendance":  "HAUSSE" if int(actuel["value"]) > int(hier["value"]) else "BAISSE",
        }
    except Exception:
        return None


def collect_all_news(ticker: str, asset_info: dict) -> list[str]:
    """Agrège toutes les sources de news et déduplique."""
    headlines = []
    headlines += fetch_yfinance_news(ticker)
    headlines += fetch_newsapi(asset_info["keywords"])
    headlines += fetch_rss_news(asset_info["rss_keys"])
    # Déduplique (fuzzy: par début de titre)
    seen, unique = set(), []
    for h in headlines:
        key = h[:40].lower()
        if key not in seen and h.strip():
            seen.add(key)
            unique.append(h)
    return unique[:20]


# ─────────────────────────────────────────────
# MODULE 3 — ANALYSE SENTIMENT FINBERT
# ─────────────────────────────────────────────

def analyze_sentiment_finbert(headlines: list[str]) -> dict:
    if not headlines:
        return {"score": 0.0, "label": "NEUTRE", "positif": 0,
                "negatif": 0, "neutre": 0, "nb_sources": 0}
    try:
        from transformers import pipeline
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                           tokenizer="ProsusAI/finbert", device=-1)
        results = finbert(headlines, truncation=True, max_length=512)
        pos = sum(1 for r in results if r["label"] == "positive")
        neg = sum(1 for r in results if r["label"] == "negative")
        neu = sum(1 for r in results if r["label"] == "neutral")
        total = len(results)
        score = (pos - neg) / total if total > 0 else 0.0
        # Pondération par volume de sources
        confiance = min(total / 10, 1.0)  # max confiance à 10 sources
        return {
            "score":      round(score, 3),
            "label":      "POSITIF" if score > 0.15 else ("NÉGATIF" if score < -0.15 else "NEUTRE"),
            "positif":    pos, "negatif": neg, "neutre": neu,
            "nb_sources": total,
            "confiance":  round(confiance, 2),
        }
    except Exception as e:
        return _sentiment_keywords(headlines)


def _sentiment_keywords(headlines: list[str]) -> dict:
    """Fallback sentiment par mots-clés financiers."""
    pos_w = ["surge", "soar", "rally", "gain", "profit", "beat", "record",
             "bullish", "upgrade", "partnership", "launch", "growth", "strong",
             "innovation", "breakthrough", "buy", "rise", "jump", "all-time"]
    neg_w = ["crash", "fall", "drop", "loss", "miss", "lawsuit", "bearish",
             "downgrade", "layoff", "cut", "decline", "weak", "ban", "fine",
             "fraud", "sell", "plunge", "collapse", "warning", "risk"]
    pos = neg = 0
    for h in headlines:
        h_low = h.lower()
        pos += sum(1 for w in pos_w if w in h_low)
        neg += sum(1 for w in neg_w if w in h_low)
    total = (pos + neg) or 1
    score = (pos - neg) / total
    return {"score": round(score, 3),
            "label": "POSITIF" if score > 0.15 else ("NÉGATIF" if score < -0.15 else "NEUTRE"),
            "positif": pos, "negatif": neg, "neutre": 0,
            "nb_sources": len(headlines), "confiance": 0.5}


# ─────────────────────────────────────────────
# MODULE 4 — SCORING MULTI-HORIZON
# ─────────────────────────────────────────────

def compute_score_horizon(market: dict, sentiment: dict,
                          horizon: str, fear_greed: Optional[dict] = None) -> dict:
    """
    Calcule un score 0-100 adapté à l'horizon demandé.

    Court terme  → RSI + MACD + Volume + Momentum 1j/5j
    Moyen terme  → MA20/50 + MACD + Sentiment pondéré + Bollinger
    Long terme   → MA50/200 + Tendance + Sentiment confiance + Fear&Greed
    """
    score = 0
    detail = {}

    rsi   = market.get("rsi", 50)
    macd  = market.get("macd", {})
    boll  = market.get("bollinger", {})
    mas   = market.get("moyennes", {})
    is_crypto = "USD" in market.get("ticker", "")

    if horizon == "court":
        # RSI (25 pts) — zone 40-60 = idéal court terme
        if 40 < rsi < 60:   r = 25
        elif 30 < rsi < 70: r = 15
        elif rsi < 30:      r = 20  # survente = rebond possible
        else:               r = 5   # surachat = risque
        score += r; detail["rsi"] = r

        # MACD (25 pts)
        m = 20 if macd.get("tendance") == "HAUSSIER" else 5
        if macd.get("croisement") == "ACHAT": m = 25
        elif macd.get("croisement") == "VENTE": m = 0
        score += m; detail["macd"] = m

        # Volume (15 pts)
        vr = market.get("vol_ratio", 1.0)
        v  = min(int(vr * 10), 15)
        score += v; detail["volume"] = v

        # Momentum 1j (20 pts)
        ch = market.get("variation_1j", 0)
        mo = min(max(int((ch + 3) / 6 * 20), 0), 20)
        score += mo; detail["momentum_1j"] = mo

        # Bollinger (15 pts)
        bz = boll.get("zone", "NEUTRE")
        b  = 10 if bz == "NEUTRE" else (5 if bz == "SURVENTE" else 3)
        score += b; detail["bollinger"] = b

        signal_seuil_achat = 65
        signal_seuil_vente = 35

    elif horizon == "moyen":
        # MA20 vs MA50 (30 pts)
        ma_score = mas.get("score_ma", 0)
        m = ma_score * 8  # max 24
        score += m; detail["ma_score"] = m

        # MACD (20 pts)
        mc = 15 if macd.get("tendance") == "HAUSSIER" else 5
        if macd.get("croisement") == "ACHAT": mc = 20
        score += mc; detail["macd"] = mc

        # Sentiment pondéré (25 pts)
        conf = sentiment.get("confiance", 0.5)
        sent = sentiment.get("score", 0)
        s    = int((sent + 1) / 2 * 25 * conf)
        score += s; detail["sentiment"] = s

        # RSI zone (15 pts)
        r = 15 if 35 < rsi < 65 else (10 if 25 < rsi < 75 else 3)
        score += r; detail["rsi"] = r

        # Momentum 5j (10 pts)
        ch5 = market.get("variation_5j", 0)
        mo5 = min(max(int((ch5 + 5) / 10 * 10), 0), 10)
        score += mo5; detail["momentum_5j"] = mo5

        signal_seuil_achat = 60
        signal_seuil_vente = 35

    else:  # long
        # MA50 vs MA200 (35 pts) — Golden/Death cross
        ma50  = mas.get("ma50")
        ma200 = mas.get("ma200")
        if ma50 and ma200:
            if ma50 > ma200:   ma_l = 35  # Golden cross
            else:              ma_l = 10  # Death cross
        else:
            ma_l = 20
        score += ma_l; detail["golden_cross"] = ma_l

        # Tendance long terme (25 pts)
        tl = 25 if market.get("tendance_long") == "HAUSSIER" else 5
        score += tl; detail["tendance_long"] = tl

        # Sentiment confiance élevée (20 pts)
        conf = sentiment.get("confiance", 0.5)
        sent = sentiment.get("score", 0)
        s    = int((sent + 1) / 2 * 20 * conf)
        score += s; detail["sentiment"] = s

        # Fear & Greed crypto (20 pts) ou momentum 1an actions
        if is_crypto and fear_greed:
            fg = fear_greed["value"]
            if fg < 25:    fg_s = 18  # peur extrême = opportunité
            elif fg < 45:  fg_s = 14  # peur = plutôt bon
            elif fg < 60:  fg_s = 10  # neutre
            elif fg < 75:  fg_s = 6   # greed = attention
            else:          fg_s = 2   # greed extrême = danger
            score += fg_s; detail["fear_greed"] = fg_s
        else:
            ch1an = market.get("variation_1an", 0)
            mo1an = min(max(int((ch1an + 20) / 40 * 20), 0), 20)
            score += mo1an; detail["momentum_1an"] = mo1an

        signal_seuil_achat = 60
        signal_seuil_vente = 30

    score = min(score, 100)

    if score >= signal_seuil_achat:
        signal, couleur = "ACHETER", "green"
    elif score <= signal_seuil_vente:
        signal, couleur = "ÉVITER", "red"
    else:
        signal, couleur = "ATTENDRE", "yellow"

    return {
        "score":   round(score, 1),
        "signal":  signal,
        "couleur": couleur,
        "detail":  detail,
        "horizon": HORIZONS[horizon]["label"],
    }


def compute_score_global(scores: dict) -> dict:
    """Score synthèse pondéré des 3 horizons."""
    # Court 30% / Moyen 40% / Long 30%
    poids = {"court": 0.30, "moyen": 0.40, "long": 0.30}
    total = sum(scores[h]["score"] * poids[h] for h in poids)

    achat = sum(1 for h in scores if scores[h]["signal"] == "ACHETER")
    eviter = sum(1 for h in scores if scores[h]["signal"] == "ÉVITER")

    if achat >= 2:   signal, couleur = "ACHETER", "green"
    elif eviter >= 2: signal, couleur = "ÉVITER", "red"
    else:             signal, couleur = "ATTENDRE", "yellow"

    return {"score": round(total, 1), "signal": signal, "couleur": couleur}


# ─────────────────────────────────────────────
# MODULE 5 — RAPPORT IA MULTI-HORIZON (Gemma4)
# ─────────────────────────────────────────────

def generate_ai_report(ticker: str, asset_info: dict, market: dict,
                       sentiment: dict, scores: dict, headlines: list[str],
                       fear_greed: Optional[dict] = None) -> str:
    """Génère un rapport narratif multi-horizon via Ollama/Gemma4."""

    macd    = market.get("macd", {})
    boll    = market.get("bollinger", {})
    mas     = market.get("moyennes", {})
    news_3  = " | ".join(headlines[:3]) if headlines else "aucune news disponible"

    # Contexte Fear & Greed pour crypto
    fg_context = ""
    if fear_greed and "USD" in ticker:
        fg_context = f"Fear&Greed={fear_greed['value']} ({fear_greed['label']}) tendance={fear_greed['tendance']}"

    prompt = f"""Tu es analyste financier senior. Analyse {asset_info['name']} ({ticker}) en français.

DONNÉES TECHNIQUES:
- Prix: {market.get('prix_actuel', 0):.2f} | Var 1j: {market.get('variation_1j', 0):+.1f}% | 5j: {market.get('variation_5j', 0):+.1f}% | 30j: {market.get('variation_30j', 0):+.1f}%
- RSI: {market.get('rsi', 50):.0f} | MACD: {macd.get('tendance','?')} ({macd.get('croisement','?')}) | Bollinger: {boll.get('zone','?')}
- MA20: {mas.get('prix_vs_ma20','?')} | MA50: {mas.get('prix_vs_ma50','?')} | MA200: {mas.get('prix_vs_ma200','?')}
- Tendance court: {market.get('tendance_court','?')} | moyen: {market.get('tendance_moyen','?')} | long: {market.get('tendance_long','?')}
- Volume: {market.get('vol_ratio', 1):.1f}x moyenne {fg_context}

SENTIMENT: {sentiment['label']} ({sentiment['score']:+.2f}) sur {sentiment['nb_sources']} sources (confiance: {sentiment.get('confiance', 0):.0%})
NEWS: {news_3}

SCORES PAR HORIZON:
- Court terme (1-5j): {scores['court']['score']:.0f}/100 → {scores['court']['signal']}
- Moyen terme (2-8 sem): {scores['moyen']['score']:.0f}/100 → {scores['moyen']['signal']}
- Long terme (6m-2ans): {scores['long']['score']:.0f}/100 → {scores['long']['signal']}

Réponds EXACTEMENT dans ce format (une ligne par section, sois factuel et concis):
COURT TERME: [signal] — [1 phrase justification basée sur RSI/MACD/momentum]
MOYEN TERME: [signal] — [1 phrase justification basée sur MA/tendance/sentiment]
LONG TERME: [signal] — [1 phrase justification basée sur MA200/golden cross/fondamentaux]
SYNTHÈSE: [1 phrase de conclusion globale avec le signal dominant]"""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 300}},
            timeout=300,
        )
        data   = response.json()
        report = data.get("response", "").strip()
        if not report:
            console.print(f"[yellow][DEBUG] Ollama vide — clés: {list(data.keys())}[/yellow]")
            return "Rapport IA non disponible."
        return report
    except requests.exceptions.ConnectionError:
        return f"⚠ Ollama non accessible sur {OLLAMA_HOST}"
    except Exception as e:
        return f"Erreur Ollama: {e}"


# ─────────────────────────────────────────────
# AFFICHAGE RICH
# ─────────────────────────────────────────────

def display_asset_card(ticker: str, asset_info: dict, market: dict,
                       sentiment: dict, scores: dict, score_global: dict,
                       report: str, fear_greed: Optional[dict] = None):
    cat_colors = {"GPU": "cyan", "AI": "magenta", "Crypto": "yellow"}
    cat_color  = cat_colors.get(asset_info["category"], "white")

    ch   = market.get("variation_1j", 0)
    ch_s = f"[green]+{ch:.2f}%[/green]" if ch >= 0 else f"[red]{ch:.2f}%[/red]"
    prix = f"{market.get('prix_actuel', 0):.4f}" if "USD" in ticker else f"{market.get('prix_actuel', 0):.2f}"

    # Barre score global
    sv = score_global["score"]
    bar = "█" * int(sv / 5) + "░" * (20 - int(sv / 5))
    bc  = score_global["couleur"]

    # Ligne scores 3 horizons
    def fmt_score(h):
        s = scores[h]
        c = s["couleur"]
        return f"[{c}]{HORIZONS[h]['label'].split('(')[0].strip()}: {s['score']:.0f} → {s['signal']}[/{c}]"

    horizons_line = "  |  ".join(fmt_score(h) for h in ["court", "moyen", "long"])

    # Fear & Greed si crypto
    fg_line = ""
    if fear_greed and "USD" in ticker:
        fg_val = fear_greed["value"]
        fg_col = "green" if fg_val < 40 else ("red" if fg_val > 70 else "yellow")
        fg_line = f"\n[bold]Fear & Greed :[/bold] [{fg_col}]{fg_val} — {fear_greed['label']}[/{fg_col}] ({fear_greed['tendance']})"

    macd = market.get("macd", {})
    boll = market.get("bollinger", {})
    mas  = market.get("moyennes", {})

    content = f"""
[bold]Prix :[/bold] {prix}  {ch_s}   5j: {market.get('variation_5j',0):+.1f}%  30j: {market.get('variation_30j',0):+.1f}%
[dim]RSI: {market.get('rsi',0):.0f} | MACD: {macd.get('tendance','?')} ({macd.get('croisement','?')}) | Bollinger: {boll.get('zone','?')} | Vol: {market.get('vol_ratio',1):.1f}x[/dim]
[dim]MA20: {mas.get('prix_vs_ma20','?')} | MA50: {mas.get('prix_vs_ma50','?')} | MA200: {mas.get('prix_vs_ma200','?')}[/dim]{fg_line}

[bold]Sentiment :[/bold] {sentiment['label']} ({sentiment['score']:+.3f})  [dim]{sentiment['nb_sources']} sources — confiance {sentiment.get('confiance',0):.0%}[/dim]

[bold]Score global :[/bold] [{bc}]{bar}[/{bc}] [bold {bc}]{sv:.0f}/100 → {score_global['signal']}[/bold {bc}]
{horizons_line}

─────────────────────────────────────────────
[bold dim]ANALYSE IA — 3 HORIZONS[/bold dim]

{report}
"""
    title = f"[bold {cat_color}]{asset_info['name']}[/bold {cat_color}] [{ticker}]  [{cat_color}]{asset_info['category']}[/{cat_color}]"
    console.print(Panel(content, title=title, border_style=cat_color, padding=(0, 2)))


def display_summary_table(results: list[dict]):
    table = Table(
        title="📊 RÉCAPITULATIF MULTI-HORIZON — Market Intelligence v2",
        box=box.ROUNDED, show_header=True,
        header_style="bold white on dark_blue",
    )
    table.add_column("Ticker",   style="bold", width=9)
    table.add_column("Nom",      width=12)
    table.add_column("Cat",      width=7)
    table.add_column("Prix",     justify="right", width=11)
    table.add_column("1j",       justify="right", width=7)
    table.add_column("RSI",      justify="right", width=5)
    table.add_column("MACD",     width=9)
    table.add_column("Sent.",    width=9)
    table.add_column("Court",    justify="right", width=8)
    table.add_column("Moyen",    justify="right", width=8)
    table.add_column("Long",     justify="right", width=8)
    table.add_column("Global",   width=13)

    for r in results:
        if "error" in r:
            continue
        ch = r["market"].get("variation_1j", 0)
        sc = r["scores"]
        sg = r["score_global"]
        table.add_row(
            r["ticker"],
            r["asset_info"]["name"],
            r["asset_info"]["category"],
            f"{r['market'].get('prix_actuel', 0):.2f}",
            f"[green]+{ch:.1f}%[/green]" if ch >= 0 else f"[red]{ch:.1f}%[/red]",
            f"{r['market'].get('rsi', 0):.0f}",
            f"[{'green' if r['market'].get('macd',{}).get('tendance')=='HAUSSIER' else 'red'}]{r['market'].get('macd',{}).get('tendance','?')}[/]",
            f"[{'green' if r['sentiment']['label']=='POSITIF' else 'red' if r['sentiment']['label']=='NÉGATIF' else 'yellow'}]{r['sentiment']['label']}[/]",
            f"[{sc['court']['couleur']}]{sc['court']['score']:.0f}[/{sc['court']['couleur']}]",
            f"[{sc['moyen']['couleur']}]{sc['moyen']['score']:.0f}[/{sc['moyen']['couleur']}]",
            f"[{sc['long']['couleur']}]{sc['long']['score']:.0f}[/{sc['long']['couleur']}]",
            f"[{sg['couleur']}]{sg['score']:.0f} → {sg['signal']}[/{sg['couleur']}]",
        )
    console.print(table)


def save_results_json(results: list[dict], path: str = "results.json"):
    output = []
    for r in results:
        if "error" not in r:
            output.append({
                "ticker":       r["ticker"],
                "name":         r["asset_info"]["name"],
                "category":     r["asset_info"]["category"],
                "timestamp":    datetime.now().isoformat(),
                "market":       {k: v for k, v in r["market"].items() if k != "history_14j"},
                "sentiment":    r["sentiment"],
                "scores":       r["scores"],
                "score_global": r["score_global"],
                "report":       r["report"],
                "headlines":    r["headlines"][:5],
                "fear_greed":   r.get("fear_greed"),
            })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]💾 Résultats sauvegardés dans {path}[/dim]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_analysis(tickers: Optional[list[str]] = None, skip_ai: bool = False):
    targets = tickers or list(ASSETS.keys())

    console.print(Panel(
        f"[bold cyan]🚀 MARKET INTELLIGENCE v2[/bold cyan]\n"
        f"[dim]Actifs: {len(targets)} | Modèle: {OLLAMA_MODEL} | {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]\n"
        f"[dim]Horizons: Court (1-5j) · Moyen (2-8 sem) · Long (6m-2ans)[/dim]",
        border_style="cyan"
    ))

    # Fear & Greed une seule fois pour tous les crypto
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
            progress.update(task, description=f"[cyan]Analyse {asset_info['name']}...[/cyan]")

            # Données marché
            market = fetch_market_data(ticker)
            if "error" in market:
                console.print(f"[red]⚠ {ticker}: {market['error']}[/red]")
                progress.advance(task)
                continue
            market["ticker"] = ticker

            # News multi-sources
            headlines = collect_all_news(ticker, asset_info)

            # Sentiment
            sentiment = analyze_sentiment_finbert(headlines)

            # Scores 3 horizons
            fg = fear_greed if "USD" in ticker else None
            scores = {
                "court": compute_score_horizon(market, sentiment, "court", fg),
                "moyen": compute_score_horizon(market, sentiment, "moyen", fg),
                "long":  compute_score_horizon(market, sentiment, "long",  fg),
            }
            score_global = compute_score_global(scores)

            # Rapport IA
            if skip_ai:
                report = "⏭ Rapport IA désactivé (mode rapide)"
            else:
                progress.update(task, description=f"[magenta]IA {asset_info['name']}...[/magenta]")
                report = generate_ai_report(ticker, asset_info, market, sentiment,
                                           scores, headlines, fg)

            result = {
                "ticker": ticker, "asset_info": asset_info,
                "market": market, "headlines": headlines,
                "sentiment": sentiment, "scores": scores,
                "score_global": score_global, "report": report,
                "fear_greed": fg,
            }
            all_results.append(result)
            progress.advance(task)

    console.print()
    display_summary_table(all_results)
    console.print()
    for r in all_results:
        display_asset_card(r["ticker"], r["asset_info"], r["market"],
                           r["sentiment"], r["scores"], r["score_global"],
                           r["report"], r.get("fear_greed"))

    save_results_json(all_results)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Market Intelligence v2 — Multi-Horizon")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers à analyser. Défaut: tous")
    parser.add_argument("--no-ai", action="store_true",
                        help="Désactive Ollama (mode rapide)")
    parser.add_argument("--loop", type=int, default=0,
                        help="Relancer toutes les N minutes")
    args = parser.parse_args()

    if args.loop > 0:
        console.print(f"[cyan]Mode boucle: toutes les {args.loop} min[/cyan]")
        while True:
            run_analysis(tickers=args.tickers, skip_ai=args.no_ai)
            console.print(f"\n[dim]Prochaine analyse dans {args.loop} min...[/dim]\n")
            time.sleep(args.loop * 60)
    else:
        run_analysis(tickers=args.tickers, skip_ai=args.no_ai)
