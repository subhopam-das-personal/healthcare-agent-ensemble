# Deploy A2A Agent to Railway

The MCP server is already live. These steps deploy the A2A agent as a second Railway service from the same repo.

## Step 1 — Open your Railway project dashboard

Go to the Railway project that already has the MCP server running.

## Step 2 — Add a new service

1. Click **"+ New"** → **"GitHub Repo"**
2. Select the same repo (`healthcare-agent-ensemble`)
3. Railway will detect it and start a deployment — **pause it before it runs**, because you need to set the start command first

## Step 3 — Set the start command

In the new service's settings → **Deploy** tab → **Start Command**:

```
python src/a2a_agent/server.py
```

> The `Procfile` only has `web: python src/mcp_server/server.py`. The A2A service needs an explicit override or it will start the MCP server instead.

## Step 4 — Set environment variables

In the new service → **Variables** tab, add:

| Variable | Value |
|---|---|
| `MCP_SERVER_URL` | `https://healthcare-agent-ensemble-production.up.railway.app/mcp` |
| `MCP_API_KEY` | `clin-intel-2026` |
| `ANTHROPIC_API_KEY` | `<your Anthropic key>` |
| `A2A_PUBLIC_URL` | Leave blank for now — fill in after Railway assigns the URL |

> Do **not** set `PORT` — Railway injects it automatically. `server.py` already reads it via `os.environ.get("PORT", ...)`.

## Step 5 — Deploy and get the URL

Click **Deploy**. Once live, Railway assigns a URL like:

```
https://healthcare-agent-ensemble-a2a-production.up.railway.app
```

## Step 6 — Set `A2A_PUBLIC_URL`

Go back to the new service → **Variables** → add:

```
A2A_PUBLIC_URL = https://<your-new-a2a-service>.up.railway.app
```

Railway auto-redeploys on variable changes.

## Step 7 — Verify it's live

```bash
curl https://<your-a2a-url>/.well-known/agent.json
```

Expected response: the agent card JSON with `"name": "Clinical Decision Support Orchestrator"`.

## Step 8 — Test the full chain

Run a quick drug check first (faster than comprehensive review) to confirm A2A → MCP → FHIR is wired correctly:

```bash
curl -X POST https://<your-a2a-url>/message/send \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "{\"patient_id\": \"fa064acf-b7f1-4279-83d3-7a94686da7ba\", \"skill\": \"quick-drug-check\"}"}]
    }
  }'
```

Once that works, run the full synthesis:

```bash
curl -X POST https://<your-a2a-url>/message/send \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "{\"patient_id\": \"fa064acf-b7f1-4279-83d3-7a94686da7ba\", \"skill\": \"comprehensive-clinical-review\"}"}]
    }
  }'
```

Demo patient: **Jeffery Daniel** (`fa064acf-b7f1-4279-83d3-7a94686da7ba`) on `https://r4.smarthealthit.org` — 16 conditions, 9 medications.
