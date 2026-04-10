"""
╔══════════════════════════════════════════════════════════════╗
║        MARKET INTELLIGENCE — Sentiment + AI Engine           ║
║        Stocks: NVDA, AMD, INTC, MSFT, GOOGL, META, TSLA     ║
║        Crypto: BTC, ETH, SOL                                 ║
║        Powered by: yfinance + NewsAPI + Ollama/Gemma4        ║
╚══════════════════════════════════════════════════════════════╝

INSTALLATION:
    pip install yfinance requests transformers torch pandas numpy
    pip install newsapi-python python-dotenv rich colorama

CONFIG:
    Créer un fichier .env avec:
    NEWS_API_KEY=your_newsapi_key_here   (gratuit sur newsapi.org)
    OLLAMA_HOST=http://localhost:11434   (par défaut)
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
# Charge le .env depuis le meme dossier que le script
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ASSETS = {
    # GPU / Hardware
    "NVDA":  {"name": "NVIDIA",       "category": "GPU",    "keywords": "NVIDIA GPU graphics RTX AI chip"},
    "AMD":   {"name": "AMD",          "category": "GPU",    "keywords": "AMD Radeon RX graphics chip processor"},
    "INTC":  {"name": "Intel",        "category": "GPU",    "keywords": "Intel Arc GPU Arc graphics chip"},

    # IA / Tech
    "MSFT":  {"name": "Microsoft",    "category": "AI",     "keywords": "Microsoft AI Copilot Azure OpenAI"},
    "GOOGL": {"name": "Alphabet",     "category": "AI",     "keywords": "Google Gemini AI DeepMind Alphabet"},
    "META":  {"name": "Meta",         "category": "AI",     "keywords": "Meta AI LLaMA Facebook Instagram"},
    "TSLA":  {"name": "Tesla",        "category": "AI",     "keywords": "Tesla FSD Autopilot Dojo AI robotics"},

    # Crypto
    "BTC-USD":  {"name": "Bitcoin",   "category": "Crypto", "keywords": "Bitcoin BTC crypto halving ETF"},
    "ETH-USD":  {"name": "Ethereum",  "category": "Crypto", "keywords": "Ethereum ETH DeFi staking Layer2"},
    "SOL-USD":  {"name": "Solana",    "category": "Crypto", "keywords": "Solana SOL blockchain DeFi meme"},
}

NEWS_API_KEY  = os.getenv("NEWS_API_KEY", "4be80a8b3bd9476fa8dfc3b8fb31fba4")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "gemma4:latest")

console = Console()


# ─────────────────────────────────────────────
# MODULE 1 — DONNÉES FINANCIÈRES (yfinance)
# ─────────────────────────────────────────────

def fetch_market_data(ticker: str) -> dict:
    """Récupère prix actuel, historique, indicateurs techniques."""
    try:
        asset = yf.Ticker(ticker)
        hist  = asset.history(period="30d", interval="1d")

        if hist.empty:
            return {"error": f"Pas de données pour {ticker}"}

        close    = hist["Close"]
        volume   = hist["Volume"]
        current  = float(close.iloc[-1])
        prev_day = float(close.iloc[-2]) if len(close) > 1 else current
        change_1d = ((current - prev_day) / prev_day) * 100

        # Calcul indicateurs techniques simples
        sma_7  = float(close.rolling(7).mean().iloc[-1])
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        
        # RSI (14)
        delta  = close.diff()
        gain   = delta.where(delta > 0, 0).rolling(14).mean()
        loss   = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs     = gain / loss
        rsi    = float(100 - (100 / (1 + rs)).iloc[-1])

        # Volatilité (std sur 14j)
        volatility = float(close.pct_change().rolling(14).std().iloc[-1] * 100)

        # Tendance
        trend = "HAUSSIER" if sma_7 > sma_20 else "BAISSIER"

        # Volume moyen
        avg_volume    = float(volume.rolling(7).mean().iloc[-1])
        current_volume = float(volume.iloc[-1])
        volume_ratio  = current_volume / avg_volume if avg_volume > 0 else 1.0

        return {
            "ticker":          ticker,
            "current_price":   current,
            "change_1d":       change_1d,
            "sma_7":           sma_7,
            "sma_20":          sma_20,
            "rsi":             rsi,
            "volatility":      volatility,
            "trend":           trend,
            "volume_ratio":    volume_ratio,
            "price_30d_high":  float(close.max()),
            "price_30d_low":   float(close.min()),
            "history_closes":  close.tail(14).tolist(),
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# MODULE 2 — SENTIMENT (NewsAPI + FinBERT)
# ─────────────────────────────────────────────

def fetch_news(keywords: str, days_back: int = 3) -> list[str]:
    """Récupère les titres de news récentes via NewsAPI."""
    if not NEWS_API_KEY:
        console.print("[yellow]⚠ NEWS_API_KEY manquant — sentiment simulé[/yellow]")
        return []

    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        keywords,
            "from":     from_date,
            "sortBy":   "publishedAt",
            "language": "en",
            "pageSize": 20,
            "apiKey":   NEWS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "ok":
            return []

        headlines = [
            art["title"] for art in data.get("articles", [])
            if art.get("title") and "[Removed]" not in art["title"]
        ]
        return headlines[:15]

    except Exception as e:
        console.print(f"[red]NewsAPI erreur: {e}[/red]")
        return []


def analyze_sentiment_finbert(headlines: list[str]) -> dict:
    """Analyse de sentiment avec FinBERT (modèle NLP financier)."""
    if not headlines:
        # Retourne un score neutre si pas de news
        return {"score": 0.0, "label": "NEUTRE", "positive": 0, "negative": 0, "neutral": 0, "headlines_used": 0}

    try:
        from transformers import pipeline
        # FinBERT est entraîné spécifiquement sur des textes financiers
        finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1  # CPU
        )

        results   = finbert(headlines, truncation=True, max_length=512)
        pos_count = sum(1 for r in results if r["label"] == "positive")
        neg_count = sum(1 for r in results if r["label"] == "negative")
        neu_count = sum(1 for r in results if r["label"] == "neutral")
        total     = len(results)

        # Score entre -1 et +1
        score = (pos_count - neg_count) / total if total > 0 else 0.0

        if score > 0.2:
            label = "POSITIF"
        elif score < -0.2:
            label = "NÉGATIF"
        else:
            label = "NEUTRE"

        return {
            "score":         round(score, 3),
            "label":         label,
            "positive":      pos_count,
            "negative":      neg_count,
            "neutral":       neu_count,
            "headlines_used": total,
        }

    except ImportError:
        console.print("[yellow]⚠ transformers non installé — utilisation de sentiment simplifié[/yellow]")
        return _simple_sentiment(headlines)

    except Exception as e:
        console.print(f"[red]FinBERT erreur: {e}[/red]")
        return _simple_sentiment(headlines)


def _simple_sentiment(headlines: list[str]) -> dict:
    """Fallback : analyse de sentiment basée sur mots-clés financiers."""
    positive_words = [
        "surge", "soar", "rally", "gain", "profit", "beat", "record",
        "bullish", "upgrade", "partnership", "launch", "growth", "strong",
        "innovation", "breakthrough", "invest", "buy", "rise", "jump"
    ]
    negative_words = [
        "crash", "fall", "drop", "loss", "miss", "lawsuit", "scandal",
        "bearish", "downgrade", "layoff", "cut", "decline", "weak",
        "ban", "fine", "fraud", "sell", "plunge", "sink", "collapse"
    ]

    pos = neg = 0
    for h in headlines:
        h_lower = h.lower()
        pos += sum(1 for w in positive_words if w in h_lower)
        neg += sum(1 for w in negative_words if w in h_lower)

    total = pos + neg if (pos + neg) > 0 else 1
    score = (pos - neg) / total

    return {
        "score":          round(score, 3),
        "label":          "POSITIF" if score > 0.2 else ("NÉGATIF" if score < -0.2 else "NEUTRE"),
        "positive":       pos,
        "negative":       neg,
        "neutral":        len(headlines) - pos - neg,
        "headlines_used": len(headlines),
    }


# ─────────────────────────────────────────────
# MODULE 3 — SCORING COMPOSITE
# ─────────────────────────────────────────────

def compute_composite_score(market: dict, sentiment: dict) -> dict:
    """
    Calcule un score composite de signal (0 à 100).
    Pondération:
      - Sentiment news    : 35%
      - Momentum prix     : 25%
      - RSI               : 20%
      - Volume            : 10%
      - Tendance SMA      : 10%
    """
    scores = {}

    # 1. Score sentiment (-1 → +1) → normalisé 0-100
    sent_norm = (sentiment["score"] + 1) / 2 * 100
    scores["sentiment"] = sent_norm

    # 2. Momentum prix (variation sur 1j)
    change = market.get("change_1d", 0)
    momentum = min(max((change + 5) / 10 * 100, 0), 100)
    scores["momentum"] = momentum

    # 3. RSI : 30-70 = zone saine, <30 = survendu (opportunité), >70 = suracheté
    rsi = market.get("rsi", 50)
    if rsi < 30:
        rsi_score = 80  # Survendu = potentiel rebond
    elif rsi > 70:
        rsi_score = 30  # Suracheté = risque correction
    else:
        rsi_score = 50 + (rsi - 50)  # Linéaire dans la zone neutre
    scores["rsi"] = rsi_score

    # 4. Volume
    vol_ratio = market.get("volume_ratio", 1.0)
    vol_score = min(vol_ratio * 50, 100)
    scores["volume"] = vol_score

    # 5. Tendance SMA
    scores["trend"] = 70 if market.get("trend") == "HAUSSIER" else 30

    # Score composite pondéré
    composite = (
        scores["sentiment"] * 0.35 +
        scores["momentum"]  * 0.25 +
        scores["rsi"]       * 0.20 +
        scores["volume"]    * 0.10 +
        scores["trend"]     * 0.10
    )

    # Signal décisionnel
    if composite >= 65:
        signal = "ACHETER"
        signal_color = "green"
    elif composite <= 35:
        signal = "VENDRE/ÉVITER"
        signal_color = "red"
    else:
        signal = "ATTENDRE"
        signal_color = "yellow"

    return {
        "composite":    round(composite, 1),
        "signal":       signal,
        "signal_color": signal_color,
        "breakdown":    scores,
    }


# ─────────────────────────────────────────────
# MODULE 4 — RAPPORT IA (Ollama/Gemma4)
# ─────────────────────────────────────────────

def generate_ai_report(
    ticker: str,
    asset_info: dict,
    market: dict,
    sentiment: dict,
    score: dict,
    headlines: list[str],
) -> str:
    """Génère un rapport d'analyse narratif via Ollama/Gemma4."""

    headlines_text = "\n".join(f"  - {h}" for h in headlines[:8]) if headlines else "  - Aucune news récente disponible"

    # Prompt court pour eviter depassement de contexte
    headlines_short = " | ".join(headlines[:3]) if headlines else "aucune news"
    prompt = f"""Analyste financier. Rapport BREF en francais pour {asset_info['name']} ({ticker}).

Donnees: prix={market.get('current_price',0):.2f} var24h={market.get('change_1d',0):+.1f}% RSI={market.get('rsi',50):.0f} tendance={market.get('trend','')} vol={market.get('volume_ratio',1):.1f}x sentiment={sentiment['label']}({sentiment['score']:+.2f}) score={score['composite']:.0f}/100
News: {headlines_short}

Reponds avec exactement ce format (3 lignes):
CONSTAT: [1 phrase factuelle sur les donnees]
RISQUES: [1 phrase sur risques et catalyseurs]
VERDICT {score['signal']}: [1 phrase de justification]"""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Bas pour plus de cohérence factuelle
                    "top_p": 0.9,
                    "num_predict": 1024,
                }
            },
            timeout=300,
        )
        data = response.json()
        report = data.get("response", "").strip()
        if not report:
            # Debug: affiche la reponse brute si vide
            print(f"[DEBUG] Reponse Ollama vide. Cles disponibles: {list(data.keys())}")
            print(f"[DEBUG] Reponse brute: {str(data)[:300]}")
            return "Rapport non disponible (reponse vide d Ollama)."
        return report

    except requests.exceptions.ConnectionError:
        return f"⚠ Ollama non accessible sur {OLLAMA_HOST}. Lance Ollama avec: ollama serve"
    except Exception as e:
        return f"Erreur Ollama: {str(e)}"


# ─────────────────────────────────────────────
# AFFICHAGE RICH — DASHBOARD TERMINAL
# ─────────────────────────────────────────────

def display_asset_card(ticker: str, asset_info: dict, market: dict, sentiment: dict, score: dict, report: str):
    """Affiche une carte complète pour un actif dans le terminal."""

    cat_colors = {"GPU": "cyan", "AI": "magenta", "Crypto": "yellow"}
    cat_color  = cat_colors.get(asset_info["category"], "white")

    # Prix et variation
    change = market.get("change_1d", 0)
    change_str = f"[green]+{change:.2f}%[/green]" if change >= 0 else f"[red]{change:.2f}%[/red]"
    price_str  = f"{market.get('current_price', 0):.4f}" if "USD" in ticker else f"{market.get('current_price', 0):.2f}"

    # Score bar
    score_val = score["composite"]
    filled    = int(score_val / 5)
    bar       = "█" * filled + "░" * (20 - filled)
    bar_color = "green" if score_val >= 65 else ("red" if score_val <= 35 else "yellow")

    title = f"[bold {cat_color}]{asset_info['name']}[/bold {cat_color}] [{ticker}]  [{cat_color}]{asset_info['category']}[/{cat_color}]"

    content = f"""
[bold]Prix :[/bold] {price_str}  {change_str}   [dim]RSI: {market.get('rsi', 0):.0f} | Vol: {market.get('volume_ratio', 1):.1f}x | {market.get('trend', '')}[/dim]

[bold]Sentiment :[/bold] {sentiment['label']} ({sentiment['score']:+.3f})   [dim]{sentiment['headlines_used']} news analysées[/dim]

[bold]Score :[/bold] [{bar_color}]{bar}[/{bar_color}] [bold {bar_color}]{score_val:.0f}/100[/bold {bar_color}]  ➤ [bold {score["signal_color"]}]{score["signal"]}[/bold {score["signal_color"]}]

─────────────────────────────────────────────
[bold dim]ANALYSE IA (Gemma4)[/bold dim]

{report}
"""
    console.print(Panel(content, title=title, border_style=cat_color, padding=(0, 2)))


def display_summary_table(results: list[dict]):
    """Affiche un tableau récapitulatif de tous les actifs."""

    table = Table(
        title="📊 RÉCAPITULATIF — Market Intelligence",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
    )
    table.add_column("Ticker",    style="bold", width=10)
    table.add_column("Nom",       width=14)
    table.add_column("Catégorie", width=10)
    table.add_column("Prix",      justify="right", width=12)
    table.add_column("24h",       justify="right", width=8)
    table.add_column("RSI",       justify="right", width=6)
    table.add_column("Sentiment", width=10)
    table.add_column("Score",     justify="right", width=7)
    table.add_column("Signal",    width=14)

    for r in results:
        if "error" in r:
            continue
        change = r["market"].get("change_1d", 0)
        table.add_row(
            r["ticker"],
            r["asset_info"]["name"],
            r["asset_info"]["category"],
            f"{r['market'].get('current_price', 0):.2f}",
            f"[green]+{change:.1f}%[/green]" if change >= 0 else f"[red]{change:.1f}%[/red]",
            f"{r['market'].get('rsi', 0):.0f}",
            f"[{'green' if r['sentiment']['label']=='POSITIF' else 'red' if r['sentiment']['label']=='NÉGATIF' else 'yellow'}]{r['sentiment']['label']}[/]",
            f"{r['score']['composite']:.0f}",
            f"[{r['score']['signal_color']}]{r['score']['signal']}[/{r['score']['signal_color']}]",
        )
    console.print(table)


def save_results_json(results: list[dict], path: str = "results.json"):
    """Sauvegarde les résultats en JSON pour usage externe (dashboard web, etc.)."""
    output = []
    for r in results:
        if "error" not in r:
            output.append({
                "ticker":    r["ticker"],
                "name":      r["asset_info"]["name"],
                "category":  r["asset_info"]["category"],
                "timestamp": datetime.now().isoformat(),
                "market":    {k: v for k, v in r["market"].items() if k != "history_closes"},
                "sentiment": r["sentiment"],
                "score":     r["score"],
                "report":    r["report"],
                "headlines": r["headlines"][:5],
            })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]💾 Résultats sauvegardés dans {path}[/dim]")


# ─────────────────────────────────────────────
# MAIN — ORCHESTRATION
# ─────────────────────────────────────────────

def run_analysis(tickers: Optional[list[str]] = None, skip_ai_report: bool = False):
    """Lance l'analyse complète sur tous les actifs (ou une sélection)."""

    targets = tickers if tickers else list(ASSETS.keys())

    console.print(Panel(
        f"[bold cyan]🚀 MARKET INTELLIGENCE ENGINE[/bold cyan]\n"
        f"[dim]Actifs: {len(targets)} | Modèle IA: {OLLAMA_MODEL} | {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]",
        border_style="cyan"
    ))

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task("Analyse en cours...", total=len(targets))

        for ticker in targets:
            if ticker not in ASSETS:
                console.print(f"[red]Ticker inconnu: {ticker}[/red]")
                continue

            asset_info = ASSETS[ticker]
            progress.update(task, description=f"[cyan]Analyse {asset_info['name']}...[/cyan]")

            # Module 1 : Données marché
            market = fetch_market_data(ticker)
            if "error" in market:
                console.print(f"[red]⚠ {ticker}: {market['error']}[/red]")
                progress.advance(task)
                continue

            # Module 2 : Sentiment
            headlines = fetch_news(asset_info["keywords"])
            sentiment = analyze_sentiment_finbert(headlines)

            # Module 3 : Score composite
            score = compute_composite_score(market, sentiment)

            # Module 4 : Rapport IA
            if skip_ai_report:
                report = "⏭ Rapport IA désactivé (mode rapide)"
            else:
                progress.update(task, description=f"[magenta]Génération rapport IA {asset_info['name']}...[/magenta]")
                report = generate_ai_report(ticker, asset_info, market, sentiment, score, headlines)

            result = {
                "ticker":     ticker,
                "asset_info": asset_info,
                "market":     market,
                "headlines":  headlines,
                "sentiment":  sentiment,
                "score":      score,
                "report":     report,
            }
            all_results.append(result)
            progress.advance(task)

    # ─── Affichage des résultats ───
    console.print()
    display_summary_table(all_results)
    console.print()

    for r in all_results:
        display_asset_card(
            r["ticker"], r["asset_info"], r["market"],
            r["sentiment"], r["score"], r["report"]
        )

    save_results_json(all_results)
    return all_results


# ─────────────────────────────────────────────
# ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Intelligence — Sentiment + IA")
    parser.add_argument(
        "--tickers", nargs="+",
        help="Liste de tickers à analyser (ex: NVDA AMD BTC-USD). Défaut: tous",
        default=None
    )
    parser.add_argument(
        "--no-ai", action="store_true",
        help="Désactive les rapports Ollama (mode rapide)"
    )
    parser.add_argument(
        "--loop", type=int, default=0,
        help="Relancer l'analyse toutes les N minutes (0 = une seule fois)"
    )
    args = parser.parse_args()

    if args.loop > 0:
        console.print(f"[cyan]Mode boucle : analyse toutes les {args.loop} minutes[/cyan]")
        while True:
            run_analysis(tickers=args.tickers, skip_ai_report=args.no_ai)
            console.print(f"\n[dim]Prochaine analyse dans {args.loop} minutes...[/dim]\n")
            time.sleep(args.loop * 60)
    else:
        run_analysis(tickers=args.tickers, skip_ai_report=args.no_ai)
