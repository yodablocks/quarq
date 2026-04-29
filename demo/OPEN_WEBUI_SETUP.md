# Open WebUI Setup for quarq Demo

## 1. Install and Start Open WebUI

### Method: pip (no Docker required)

```bash
pip install open-webui
open-webui serve
```

Open WebUI starts at **http://localhost:8080** (pip install default port).

If pip install fails due to dependency conflicts, fall back to Docker:

```bash
docker run -d -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

**Verified method:** pip install v0.9.2 (no Docker needed on this machine). Default port: 8080.

---

## 2. Connect Open WebUI to LM Studio

LM Studio is running at `http://192.168.1.101:1234` (from `~/.quarq/config.toml`).

In Open WebUI:

1. Go to **Settings > Connections > OpenAI API**
2. Set **API Base URL** to:
   - `http://192.168.1.101:1234/v1` (pip install method)
   - `http://host.docker.internal:1234/v1` (Docker method)
3. Set **API Key** to `lm-studio` (any non-empty string — LM Studio ignores it)
4. Click **Save**

---

## 3. Install the quarq Tools

1. In Open WebUI, go to **Admin > Tools**
2. Click **+** to add a new tool
3. Upload or paste the contents of `demo/tools/quarq_rag_tool.py`
4. Save. Repeat for `demo/tools/quarq_portfolio_tool.py`

The tools will appear in the chat interface tool selector once installed.

---

## 4. Start quarq

In a terminal:

```bash
quarq serve
```

quarq API will be available at **http://127.0.0.1:8000**.
Swagger docs: **http://127.0.0.1:8000/docs**

Or use the demo launcher (starts quarq and opens the browser):

```bash
./demo/start_demo.sh
```

---

## 5. Test the Tools

### RAG demo prompt

```
What does the ECB Financial Stability Review say about concentration risk
in European equity markets?
```

Expected: a grounded answer with source citations (document name + page number).

### Portfolio demo prompt

```
Analyse this portfolio: MC.PA, TTE.PA, AIR.PA, SAN.PA with weights
0.30, 0.25, 0.25, 0.20. Use the CAC 40 as benchmark.
```

Expected: formatted metrics table showing CAGR, Sharpe, Max Drawdown, VaR, Volatility, Beta, Alpha.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "quarq server is not running" in chat | Run `quarq serve` in a terminal |
| No models in Open WebUI | Check LM Studio is running and the API URL is correct |
| Tool not appearing | Re-upload the `.py` file in Admin > Tools |
| Port 8080 already in use | Kill existing process: `lsof -ti:8080 \| xargs kill` |
