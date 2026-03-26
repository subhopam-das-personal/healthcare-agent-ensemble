#!/usr/bin/env bash
# Deploy the UI server as a Railway service.
# Prerequisites: railway CLI installed, `railway login` done, repo linked to the project.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-...
#   export A2A_AGENT_URL=https://<your-a2a-service>.up.railway.app
#   bash scripts/deploy_ui.sh

set -euo pipefail

ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
A2A_AGENT_URL="${A2A_AGENT_URL:-}"
SERVICE_NAME="ui-server"

echo "==> Linking to Railway project..."
railway link

echo "==> Creating service: $SERVICE_NAME"
railway service create "$SERVICE_NAME" || echo "(service may already exist, continuing)"

echo "==> Setting environment variables..."
railway variables \
  --service "$SERVICE_NAME" \
  --set "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  --set "RAILWAY_SERVICE_NAME=$SERVICE_NAME"

if [ -n "$A2A_AGENT_URL" ]; then
  railway variables \
    --service "$SERVICE_NAME" \
    --set "A2A_AGENT_URL=$A2A_AGENT_URL"
fi

echo "==> Deploying..."
railway up --service "$SERVICE_NAME" --detach

echo ""
echo "Once deployed, get the URL from the Railway dashboard."
echo "Then verify:"
echo "  curl https://<your-ui-url>.up.railway.app/health"
echo ""
echo "Open the UI at: https://<your-ui-url>.up.railway.app/"
