#!/usr/bin/env bash
# Railway monorepo dispatcher — routes to the correct service process.
# Railway injects RAILWAY_SERVICE_NAME automatically for each service.
set -euo pipefail

SERVICE="${RAILWAY_SERVICE_NAME:-}"

if [ "$SERVICE" = "ui-server" ]; then
  echo "Starting UI server (service: $SERVICE)"
  exec streamlit run src/ui_server/app.py \
    --server.port "${PORT:-7000}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
elif [ "$SERVICE" = "a2a-agent" ]; then
  echo "Starting A2A agent (service: $SERVICE)"
  exec python src/a2a_agent/server.py
elif [ "$SERVICE" = "demo-server" ]; then
  echo "Starting demo script server (service: $SERVICE)"
  exec python demo/serve.py
else
  echo "Starting MCP server (service: ${SERVICE:-unknown})"
  exec python src/mcp_server/server.py
fi
