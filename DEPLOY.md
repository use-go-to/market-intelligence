# 🚀 Déploiement GitHub Pages — Market Intelligence

## Structure finale du repo

```
market-intelligence/
├── analyzer.py           ← Script principal (local + Ollama)
├── analyzer_cloud.py     ← Version GitHub Actions (sans Ollama)
├── requirements.txt
├── .env                  ← NE PAS committer (ajoute au .gitignore)
├── .gitignore
├── .github/
│   └── workflows/
│       └── analyze.yml   ← Tâche automatique toutes les heures
└── docs/                 ← Servi par GitHub Pages
    ├── index.html        ← Dashboard web
    └── results.json      ← Données mises à jour par Actions
```

---

## Étape 1 — Créer le repo GitHub

1. Va sur https://github.com/new
2. Nom : `market-intelligence` (ou ce que tu veux)
3. **Public** (requis pour GitHub Pages gratuit)
4. Clique "Create repository"

---

## Étape 2 — Pousser ton code

Dans PowerShell, dans ton dossier `files (3)` :

```powershell
cd "C:\Users\user\Downloads\files (3)"

git init
git add .
git commit -m "feat: market intelligence engine"

git remote add origin https://github.com/TON_USERNAME/market-intelligence.git
git branch -M main
git push -u origin main
```

---

## Étape 3 — Activer GitHub Pages

1. Dans ton repo GitHub → **Settings** → **Pages**
2. Source : **Deploy from a branch**
3. Branch : `main` / Folder : `/docs`
4. Clique **Save**

⏳ Attends 1-2 minutes → ton dashboard sera accessible sur :
```
https://TON_USERNAME.github.io/market-intelligence/
```

---

## Étape 4 — Ajouter ta clé NewsAPI comme secret

1. Dans ton repo → **Settings** → **Secrets and variables** → **Actions**
2. Clique **New repository secret**
3. Name : `NEWS_API_KEY`
4. Value : `4be80a8b3bd9476fa8dfc3b8fb31fba4`
5. Clique **Add secret**

---

## Étape 5 — Activer GitHub Actions

Le fichier `.github/workflows/analyze.yml` est déjà configuré.
Il se lancera automatiquement toutes les heures.

Pour forcer un premier lancement :
1. Va dans **Actions** → **Market Intelligence — Analyse Horaire**
2. Clique **Run workflow** → **Run workflow**

---

## Étape 6 — Créer le .gitignore

```powershell
@"
.env
__pycache__/
*.pyc
.DS_Store
results.json
"@ | Out-File -FilePath ".gitignore" -Encoding utf8
```

**Important** : Le `results.json` à la racine n'est pas commité (données locales).
Seul `docs/results.json` est dans le repo (mis à jour par Actions).

---

## Résultat final

| Quand | Quoi |
|-------|------|
| Toutes les heures | GitHub Actions lance `analyzer_cloud.py` |
| Après l'analyse | `results.json` est poussé dans `docs/` |
| Sur ton téléphone | Tu ouvres `username.github.io/market-intelligence` |
| La page | Se rafraîchit toutes les 5 minutes automatiquement |

---

## ⚠️ Limites GitHub Actions gratuit

- **2000 minutes/mois** gratuites → 1 analyse/heure = ~720 min/mois ✅ (OK)
- Pas d'accès à Ollama → les rapports IA ne seront pas générés en cloud
- Solution pour les rapports IA : lancer `analyzer.py` localement quand tu veux

---

## 💡 Astuce — Fréquence personnalisable

Dans `.github/workflows/analyze.yml`, change le cron :
```yaml
- cron: '0 * * * *'     # Toutes les heures
- cron: '0 */2 * * *'   # Toutes les 2h
- cron: '0 9,15,21 * * *'  # 3x/jour (9h, 15h, 21h UTC)
```
