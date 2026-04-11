"""
╔══════════════════════════════════════════════════════════════════════╗
║        MARKET INTELLIGENCE v2 — Multi-Horizon + Multi-Sources        ║
║        Stocks: NVDA, AMD, INTC, MSFT, GOOGL, META, TSLA             ║
║        Crypto: BTC, ETH, SOL                                         ║
║        Sources: yfinance + NewsAPI + RSS + Fear&Greed + FinBERT      ║
║        IA: Ollama — 3 horizons (court/moyen/long terme)              ║
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

RSS_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
]

HORIZONS = {
    "court": {"label": "Court terme (1-5 jours)",    "period": "5d",  "interval": "1h"},
    "moyen": {"label": "Moyen terme (2-8 semaines)", "period": "3mo", "interval": "1d"},
    "long":  {"label": "Long terme (6 mois-2 ans)",  "period": "2y",  "interval": "1wk"},
}

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

console = Console()


# ─────────────────────────────────────────────
# MODULE 1 — DONNÉES MARCHÉ
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
    pct_b = float((prix - l) / (u - l)) if u != l else 0.5
    if prix > u:   zone = "SURACHAT"
    elif prix < l: zone = "SURVENTE"
    else:          zone = "NEUTRE"
    return {"upper": u, "lower": l, "ma": float(ma.iloc[-1]),
            "pct_b": round(pct_b, 3), "zone": zone}


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
    return result


def fetch_market_data(ticker: str) -> dict:
    try:
        asset = yf.Ticker(ticker)
        hist  = asset.history(period="2y", interval="1d")
        if hist.empty:
            return {"error": f"Pas de données pour {ticker}"}

        close  = hist["Close"]
        volume = hist["Volume"]

        prix_actuel  = float(close.iloc[-1])
        prix_hier    = float(close.iloc[-2]) if len(close) > 1 else prix_actuel
        variation_1j = ((prix_actuel - prix_hier) / prix_hier) * 100

        rsi_14    = calcul_rsi(close, 14)
        macd_data = calcul_macd(close)
        bollinger = calcul_bollinger(close)
        mas       = calcul_moyennes_mobiles(close)

        high = hist["High"]
        low  = hist["Low"]
        tr   = pd.concat([high - low,
                          (high - close.shift()).abs(),
                          (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr  = float(tr.rolling(14).mean().iloc[-1])

        vol_moy    = float(volume.rolling(20).mean().iloc[-1])
        vol_actuel = float(volume.iloc[-1])
        vol_ratio  = vol_actuel / vol_moy if vol_moy > 0 else 1.0

        sma5  = close.rolling(5).mean()
        sma10 = close.rolling(10).mean()
        tendance_court = "HAUSSIER" if len(close) >= 10 and sma5.iloc[-1] > sma10.iloc[-1] else "BAISSIER"
        tendance_moyen = "HAUSSIER" if mas.get("ma20") and mas.get("ma50") and mas["ma20"] > mas["ma50"] else "BAISSIER"
        tendance_long  = "HAUSSIER" if mas.get("ma50") and mas.get("ma200") and mas["ma50"] > mas["ma200"] else "BAISSIER"

        def var_n(n):
            return ((prix_actuel - float(close.iloc[-n])) / float(close.iloc[-n])) * 100 if len(close) > n else 0.0

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
            "atr":            round(atr, 4),
            "vol_ratio":      round(vol_ratio, 2),
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
# MODULE 2 — SOURCES DE NEWS MULTIPLES
# ─────────────────────────────────────────────

def fetch_rss_news(rss_keys: list) -> list:
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
        return [a["title"] for a in data.get("articles", [])
                if a.get("title") and "[Removed]" not in a["title"]][:15]
    except Exception:
        return []


def fetch_yfinance_news(ticker: str) -> list:
    try:
        asset = yf.Ticker(ticker)
        news  = asset.news
        return [n.get("content", {}).get("title", "") for n in (news or [])
                if n.get("content", {}).get("title")][:10]
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
    headlines = []
    headlines += fetch_yfinance_news(ticker)
    headlines += fetch_newsapi(asset_info["keywords"])
    headlines += fetch_rss_news(asset_info["rss_keys"])
    seen, unique = set(), []
    for h in headlines:
        key = h[:40].lower()
        if key not in seen and h.strip():
            seen.add(key)
            unique.append(h)
    return unique[:20]


# ─────────────────────────────────────────────
# MODULE 3 — SENTIMENT FINBERT
# ─────────────────────────────────────────────

def analyze_sentiment_finbert(headlines: list) -> dict:
    if not headlines:
        return {"score": 0.0, "label": "NEUTRE", "positif": 0,
                "negatif": 0, "neutre": 0, "nb_sources": 0, "confiance": 0.0}
    try:
        from transformers import pipeline
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                           tokenizer="ProsusAI/finbert", device=-1)
        results   = finbert(headlines, truncation=True, max_length=512)
        pos       = sum(1 for r in results if r["label"] == "positive")
        neg       = sum(1 for r in results if r["label"] == "negative")
        neu       = sum(1 for r in results if r["label"] == "neutral")
        total     = len(results)
        score     = (pos - neg) / total if total > 0 else 0.0
        confiance = min(total / 10, 1.0)
        return {
            "score":      round(score, 3),
            "label":      "POSITIF" if score > 0.15 else ("NÉGATIF" if score < -0.15 else "NEUTRE"),
            "positif":    pos, "negatif": neg, "neutre": neu,
            "nb_sources": total,
            "confiance":  round(confiance, 2),
        }
    except Exception:
        return _sentiment_keywords(headlines)


def _sentiment_keywords(headlines: list) -> dict:
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
    return {
        "score":      round(score, 3),
        "label":      "POSITIF" if score > 0.15 else ("NÉGATIF" if score < -0.15 else "NEUTRE"),
        "positif":    pos, "negatif": neg, "neutre": 0,
        "nb_sources": len(headlines), "confiance": 0.5,
    }


# ─────────────────────────────────────────────
# MODULE 4 — SCORING MULTI-HORIZON
# ─────────────────────────────────────────────

def compute_score_horizon(market: dict, sentiment: dict,
                          horizon: str, fear_greed: Optional[dict] = None) -> dict:
    score     = 0
    detail    = {}
    rsi       = market.get("rsi", 50)
    macd      = market.get("macd", {})
    boll      = market.get("bollinger", {})
    mas       = market.get("moyennes", {})
    is_crypto = "USD" in market.get("ticker", "")

    if horizon == "court":
        if 40 < rsi < 60:   r = 25
        elif 30 < rsi < 70: r = 15
        elif rsi < 30:      r = 20
        else:               r = 5
        score += r; detail["rsi"] = r

        m = 20 if macd.get("tendance") == "HAUSSIER" else 5
        if macd.get("croisement") == "ACHAT":   m = 25
        elif macd.get("croisement") == "VENTE": m = 0
        score += m; detail["macd"] = m

        vr = market.get("vol_ratio", 1.0)
        v  = min(int(vr * 10), 15)
        score += v; detail["volume"] = v

        ch = market.get("variation_1j", 0)
        mo = min(max(int((ch + 3) / 6 * 20), 0), 20)
        score += mo; detail["momentum_1j"] = mo

        bz = boll.get("zone", "NEUTRE")
        b  = 10 if bz == "NEUTRE" else (15 if bz == "SURVENTE" else 3)
        score += b; detail["bollinger"] = b

        seuil_achat, seuil_vente = 65, 35

    elif horizon == "moyen":
        ma_score = mas.get("score_ma", 0)
        m = ma_score * 8
        score += m; detail["ma_score"] = m

        mc = 15 if macd.get("tendance") == "HAUSSIER" else 5
        if macd.get("croisement") == "ACHAT": mc = 20
        score += mc; detail["macd"] = mc

        conf = sentiment.get("confiance", 0.5)
        sent = sentiment.get("score", 0)
        s    = int((sent + 1) / 2 * 25 * conf)
        score += s; detail["sentiment"] = s

        r = 15 if 35 < rsi < 65 else (10 if 25 < rsi < 75 else 3)
        score += r; detail["rsi"] = r

        ch5 = market.get("variation_5j", 0)
        mo5 = min(max(int((ch5 + 5) / 10 * 10), 0), 10)
        score += mo5; detail["momentum_5j"] = mo5

        seuil_achat, seuil_vente = 60, 35

    else:  # long
        ma50  = mas.get("ma50")
        ma200 = mas.get("ma200")
        if ma50 and ma200:
            ma_l = 35 if ma50 > ma200 else 10
        else:
            ma_l = 20
        score += ma_l; detail["golden_cross"] = ma_l

        tl = 25 if market.get("tendance_long") == "HAUSSIER" else 5
        score += tl; detail["tendance_long"] = tl

        conf = sentiment.get("confiance", 0.5)
        sent = sentiment.get("score", 0)
        s    = int((sent + 1) / 2 * 20 * conf)
        score += s; detail["sentiment"] = s

        if is_crypto and fear_greed:
            fg = fear_greed["value"]
            if fg < 25:   fg_s = 18
            elif fg < 45: fg_s = 14
            elif fg < 60: fg_s = 10
            elif fg < 75: fg_s = 6
            else:         fg_s = 2
            score += fg_s; detail["fear_greed"] = fg_s
        else:
            ch1an = market.get("variation_1an", 0)
            mo1an = min(max(int((ch1an + 20) / 40 * 20), 0), 20)
            score += mo1an; detail["momentum_1an"] = mo1an

        seuil_achat, seuil_vente = 60, 30

    score = min(score, 100)

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
    poids = {"court": 0.30, "moyen": 0.40, "long": 0.30}
    total  = sum(scores[h]["score"] * poids[h] for h in poids)
    achat  = sum(1 for h in scores if scores[h]["signal"] == "ACHETER")
    eviter = sum(1 for h in scores if scores[h]["signal"] == "ÉVITER")
    if achat >= 2:    signal, couleur = "ACHETER", "green"
    elif eviter >= 2: signal, couleur = "ÉVITER",  "red"
    else:             signal, couleur = "ATTENDRE","yellow"
    return {"score": round(total, 1), "signal": signal, "couleur": couleur}


# ─────────────────────────────────────────────
# MODULE 5 — RAPPORT IA
#
# ROOT CAUSE IDENTIFIÉE :
#   Le modèle est un "thinking model" (gemma4, qwq, deepseek-r1, etc.)
#   Il met son raisonnement dans data["message"]["thinking"]
#   et retourne data["response"] = "" (vide).
#
# SOLUTION EN 3 NIVEAUX :
#   1. Extraction depuis TOUS les champs (response / content / thinking)
#   2. Pré-injection de "COURT:" pour forcer la continuation directe
#   3. Parsing du texte thinking pour en extraire les 4 lignes structurées
# ─────────────────────────────────────────────

def _extract_response(data: dict) -> str:
    """
    Extrait la réponse depuis n'importe quel format Ollama :
    - Standard   : data["response"]
    - Chat       : data["message"]["content"]
    - Thinking   : data["message"]["thinking"]  ← gemma4/qwq/deepseek-r1
    """
    # 1. /api/generate standard
    resp = (data.get("response") or "").strip()
    if resp:
        return resp

    msg = data.get("message", {})

    # 2. /api/chat contenu normal
    content = (msg.get("content") or "").strip()
    if content:
        return content

    # 3. Thinking mode — tout est dans "thinking", content est vide
    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        console.print("[yellow][DEBUG] Thinking mode détecté → extraction depuis 'thinking'[/yellow]")
        return thinking

    return ""


def _extract_4lines(text: str) -> str:
    """
    Cherche et extrait les 4 lignes structurées COURT/MOYEN/LONG/SYNTHESE
    depuis n'importe quel texte (réponse normale ou texte de raisonnement).
    """
    keywords = ("COURT", "MOYEN", "LONG", "SYNTH")
    kept = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if any(upper.startswith(k) for k in keywords):
            # Nettoie les préfixes parasites : "* COURT:" "- COURT:" "1. COURT:"
            line = line.lstrip("*-•123456789. ").strip()
            kept.append(line)
        if len(kept) == 4:
            break

    return "\n".join(kept)


def _clean_report(text: str) -> str:
    """Pipeline de nettoyage : cherche les 4 lignes, sinon retourne brut."""
    result = _extract_4lines(text)
    if result:
        return result
    # Fallback : texte brut tronqué
    return text[:400].strip()


def generate_ai_report(ticker: str, asset_info: dict, market: dict,
                       sentiment: dict, scores: dict, headlines: list,
                       fear_greed: Optional[dict] = None) -> str:

    # ── Données compactes ──
    news_1  = headlines[0][:60].replace("\n", " ") if headlines else "aucune"
    macd_t  = "H" if market.get("macd", {}).get("tendance") == "HAUSSIER" else "B"
    fg_str  = f" FG={fear_greed['value']}" if fear_greed else ""
    sent_sc = sentiment.get("score", 0)
    prix    = market.get("prix_actuel", 0)
    rsi     = market.get("rsi", 50)
    var1j   = market.get("variation_1j", 0)
    sc_c    = scores["court"]["score"]
    sc_m    = scores["moyen"]["score"]
    sc_l    = scores["long"]["score"]
    sig_c   = scores["court"]["signal"]
    sig_m   = scores["moyen"]["signal"]
    sig_l   = scores["long"]["signal"]

    # ── Prompt : on termine par "COURT:" pour forcer la continuation
    #    immédiate sans introduction ni raisonnement préalable visible ──
    prompt = (
        f"Analyse {asset_info['name']} ({ticker}) en français. "
        f"Réponds UNIQUEMENT avec ces 4 lignes, rien d'autre:\n"
        f"COURT: [signal] — [raison courte]\n"
        f"MOYEN: [signal] — [raison courte]\n"
        f"LONG: [signal] — [raison courte]\n"
        f"SYNTHESE: [conclusion en 1 phrase]\n\n"
        f"Données: Prix={prix:.2f} Var1j={var1j:+.1f}% RSI={rsi:.0f} "
        f"MACD={macd_t} Sent={sent_sc:+.2f}{fg_str} "
        f"C={sc_c:.0f}({sig_c}) M={sc_m:.0f}({sig_m}) L={sc_l:.0f}({sig_l})\n"
        f"News: {news_1}\n\n"
        f"COURT:"
    )

    console.print(f"[dim][DEBUG] {ticker}: prompt {len(prompt)} chars[/dim]")

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature":    0.1,
                    "num_predict":    300,
                    "num_ctx":        2048,
                    "top_p":          0.9,
                    "repeat_penalty": 1.1,
                    "stop": [
                        "\n\n\n",   # Arrête après triple saut de ligne
                        "Données:", # Évite une répétition du prompt
                        "Analyse ", # Évite une répétition du prompt
                        "Note:",
                        "Remarque:",
                    ],
                },
            },
            timeout=300,
        )

        data = response.json()
        ec   = data.get("eval_count", 0)
        dr   = data.get("done_reason", "?")

        # ── Extraction multi-champs (standard + thinking mode) ──
        raw = _extract_response(data)

        console.print(f"[dim][DEBUG] {ticker}: eval={ec} reason={dr} raw_chars={len(raw)}[/dim]")

        if not raw:
            # Affiche tous les champs disponibles pour le diagnostic
            all_keys = list(data.keys())
            msg_keys = list(data.get("message", {}).keys()) if "message" in data else []
            console.print(f"[red][DEBUG] Champs data: {all_keys}[/red]")
            console.print(f"[red][DEBUG] Champs message: {msg_keys}[/red]")
            console.print(f"[red][DEBUG] data[:400]: {str(data)[:400]}[/red]")
            return "Rapport IA non disponible."

        # Le prompt se termine par "COURT:" donc on le remet en tête
        full_text = "COURT:" + raw

        report = _clean_report(full_text)
        return report if report else full_text[:400]

    except requests.exceptions.ConnectionError:
        return f"⚠ Ollama non accessible sur {OLLAMA_HOST}"
    except Exception as e:
        console.print(f"[red][DEBUG] Exception generate_ai_report: {e}[/red]")
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
    prix = (f"{market.get('prix_actuel', 0):.4f}" if "USD" in ticker
            else f"{market.get('prix_actuel', 0):.2f}")

    sv  = score_global["score"]
    bar = "█" * int(sv / 5) + "░" * (20 - int(sv / 5))
    bc  = score_global["couleur"]

    def fmt_score(h):
        s     = scores[h]
        c     = s["couleur"]
        label = HORIZONS[h]["label"].split("(")[0].strip()
        return f"[{c}]{label}: {s['score']:.0f} -> {s['signal']}[/{c}]"

    horizons_line = "  |  ".join(fmt_score(h) for h in ["court", "moyen", "long"])

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

[bold]Score global :[/bold] [{bc}]{bar}[/{bc}] [bold {bc}]{sv:.0f}/100 -> {score_global['signal']}[/bold {bc}]
{horizons_line}

─────────────────────────────────────────────
[bold dim]ANALYSE IA — 3 HORIZONS[/bold dim]

{report}
"""
    title = (f"[bold {cat_color}]{asset_info['name']}[/bold {cat_color}]"
             f" [{ticker}]  [{cat_color}]{asset_info['category']}[/{cat_color}]")
    console.print(Panel(content, title=title, border_style=cat_color, padding=(0, 2)))


def display_summary_table(results: list):
    table = Table(
        title="📊 RECAPITULATIF MULTI-HORIZON — Market Intelligence v2",
        box=box.ROUNDED, show_header=True,
        header_style="bold white on dark_blue",
    )
    table.add_column("Ticker", style="bold", width=9)
    table.add_column("Nom",    width=12)
    table.add_column("Cat",    width=7)
    table.add_column("Prix",   justify="right", width=11)
    table.add_column("1j",     justify="right", width=7)
    table.add_column("RSI",    justify="right", width=5)
    table.add_column("MACD",   width=9)
    table.add_column("Sent.",  width=9)
    table.add_column("Court",  justify="right", width=8)
    table.add_column("Moyen",  justify="right", width=8)
    table.add_column("Long",   justify="right", width=8)
    table.add_column("Global", width=13)

    for r in results:
        if "error" in r:
            continue
        ch            = r["market"].get("variation_1j", 0)
        sc            = r["scores"]
        sg            = r["score_global"]
        macd_tendance = r["market"].get("macd", {}).get("tendance", "?")
        sent_label    = r["sentiment"]["label"]
        table.add_row(
            r["ticker"],
            r["asset_info"]["name"],
            r["asset_info"]["category"],
            f"{r['market'].get('prix_actuel', 0):.2f}",
            f"[green]+{ch:.1f}%[/green]" if ch >= 0 else f"[red]{ch:.1f}%[/red]",
            f"{r['market'].get('rsi', 0):.0f}",
            f"[{'green' if macd_tendance == 'HAUSSIER' else 'red'}]{macd_tendance}[/]",
            f"[{'green' if sent_label == 'POSITIF' else 'red' if sent_label == 'NÉGATIF' else 'yellow'}]{sent_label}[/]",
            f"[{sc['court']['couleur']}]{sc['court']['score']:.0f}[/{sc['court']['couleur']}]",
            f"[{sc['moyen']['couleur']}]{sc['moyen']['score']:.0f}[/{sc['moyen']['couleur']}]",
            f"[{sc['long']['couleur']}]{sc['long']['score']:.0f}[/{sc['long']['couleur']}]",
            f"[{sg['couleur']}]{sg['score']:.0f} -> {sg['signal']}[/{sg['couleur']}]",
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

        m  = r["market"]
        sg = r["score_global"]
        sc = r["scores"]
        se = r["sentiment"]

        output.append({
            "ticker":    r["ticker"],
            "name":      r["asset_info"]["name"],
            "category":  r["asset_info"]["category"],
            "timestamp": datetime.now().isoformat(),

            # ── Marché — clés attendues par index.html ──
            "market": {
                "current_price":  m.get("prix_actuel"),
                "change_1d":      m.get("variation_1j"),
                "change_5d":      m.get("variation_5j"),
                "change_30d":     m.get("variation_30j"),
                "rsi":            m.get("rsi"),
                "volume_ratio":   m.get("vol_ratio"),
                "trend":          m.get("tendance_court", ""),
                "macd_trend":     m.get("macd", {}).get("tendance", ""),
                "bollinger_zone": m.get("bollinger", {}).get("zone", ""),
                "ma20_status":    m.get("moyennes", {}).get("prix_vs_ma20", ""),
                "ma50_status":    m.get("moyennes", {}).get("prix_vs_ma50", ""),
                "ma200_status":   m.get("moyennes", {}).get("prix_vs_ma200", ""),
                "atr":            m.get("atr"),
                "high_52w":       m.get("prix_52w_haut"),
                "low_52w":        m.get("prix_52w_bas"),
            },

            # ── Sentiment — clés attendues par index.html ──
            "sentiment": {
                "label":          se.get("label", "NEUTRE"),
                "score":          se.get("score", 0),
                "headlines_used": se.get("nb_sources", 0),
                "confiance":      se.get("confiance", 0),
                "positif":        se.get("positif", 0),
                "negatif":        se.get("negatif", 0),
            },

            # ── Score — clés attendues par index.html ──
            "score": {
                "composite": round(sg["score"], 1),
                "signal":    sg["signal"],
                "court": {
                    "value":  sc["court"]["score"],
                    "signal": sc["court"]["signal"],
                },
                "moyen": {
                    "value":  sc["moyen"]["score"],
                    "signal": sc["moyen"]["signal"],
                },
                "long": {
                    "value":  sc["long"]["score"],
                    "signal": sc["long"]["signal"],
                },
            },

            "report":     r.get("report", ""),
            "headlines":  r.get("headlines", [])[:5],
            "fear_greed": r.get("fear_greed"),
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]💾 Résultats sauvegardés dans {path}[/dim]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_analysis(tickers: Optional[list] = None, skip_ai: bool = False):
    targets = tickers or list(ASSETS.keys())

    console.print(Panel(
        f"[bold cyan]🚀 MARKET INTELLIGENCE v2[/bold cyan]\n"
        f"[dim]Actifs: {len(targets)} | Modèle: {OLLAMA_MODEL} | {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]\n"
        f"[dim]Horizons: Court (1-5j) · Moyen (2-8 sem) · Long (6m-2ans)[/dim]",
        border_style="cyan"
    ))

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

            market = fetch_market_data(ticker)
            if "error" in market:
                console.print(f"[red]⚠ {ticker}: {market['error']}[/red]")
                progress.advance(task)
                continue
            market["ticker"] = ticker

            headlines = collect_all_news(ticker, asset_info)
            sentiment = analyze_sentiment_finbert(headlines)

            fg = fear_greed if "USD" in ticker else None
            scores = {
                "court": compute_score_horizon(market, sentiment, "court", fg),
                "moyen": compute_score_horizon(market, sentiment, "moyen", fg),
                "long":  compute_score_horizon(market, sentiment, "long",  fg),
            }
            score_global = compute_score_global(scores)

            if skip_ai:
                report = "⏭ Rapport IA désactivé (mode rapide)"
            else:
                progress.update(task, description=f"[magenta]IA {asset_info['name']}...[/magenta]")
                report = generate_ai_report(ticker, asset_info, market,
                                            sentiment, scores, headlines, fg)

            result = {
                "ticker":       ticker,
                "asset_info":   asset_info,
                "market":       market,
                "headlines":    headlines,
                "sentiment":    sentiment,
                "scores":       scores,
                "score_global": score_global,
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
            r["report"], r.get("fear_greed")
        )

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
