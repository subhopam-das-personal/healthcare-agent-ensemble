#!/usr/bin/env bash
# Deploy the UI server as a Railway service.
# Prerequisites: railway CLI installed, `railway login` done, repo linked to the project.
#
# Usage:
#   bash scripts/deploy_ui.sh
#
# ANTHROPIC_API_KEY is expected to already be set in the Railway project.

set -euo pipefail

SERVICE_NAME="ui-server"

echo "==> Linking to Railway project..."
railway link

echo "==> Creating service: $SERVICE_NAME"
railway service create "$SERVICE_NAME" || echo "(service may already exist, continuing)"

echo "==> Setting environment variables..."
railway variables \
  --service "$SERVICE_NAME" \
  --set "RAILWAY_SERVICE_NAME=$SERVICE_NAME"

echo "==> Deploying..."
railway up --service "$SERVICE_NAME" --detach

echo ""
echo "Once deployed, get the URL from the Railway dashboard."
echo "Then verify:"
echo "  curl https://<your-ui-url>.up.railway.app/health"
echo ""
echo "Open the UI at: https://<your-ui-url>.up.railway.app/"
