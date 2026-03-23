#!/usr/bin/env bash
# Deploy the A2A agent as a second Railway service.
# Prerequisites: railway CLI installed, `railway login` done, repo linked to the project.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-...
#   export MCP_API_KEY=clin-intel-2026
#   bash scripts/deploy_a2a.sh

set -euo pipefail

MCP_SERVER_URL="${MCP_SERVER_URL:-https://healthcare-agent-ensemble-production.up.railway.app/mcp}"
MCP_API_KEY="${MCP_API_KEY:?Set MCP_API_KEY}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
SERVICE_NAME="a2a-agent"

echo "==> Linking to Railway project..."
railway link

echo "==> Creating service: $SERVICE_NAME"
railway service create "$SERVICE_NAME" || echo "(service may already exist, continuing)"

echo "==> Setting environment variables..."
railway variables \
  --service "$SERVICE_NAME" \
  --set "MCP_SERVER_URL=$MCP_SERVER_URL" \
  --set "MCP_API_KEY=$MCP_API_KEY" \
  --set "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"

echo "==> Deploying..."
railway up --service "$SERVICE_NAME" --detach

echo ""
echo "Once deployed, get the URL from the Railway dashboard and run:"
echo "  railway variables --service $SERVICE_NAME --set 'A2A_PUBLIC_URL=https://<your-url>.up.railway.app'"
echo ""
echo "Then verify:"
echo "  curl https://<your-url>.up.railway.app/.well-known/agent.json"
