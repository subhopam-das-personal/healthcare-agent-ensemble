#!/usr/bin/env bash
# Railway monorepo dispatcher — routes to the correct service process.
# Railway injects RAILWAY_SERVICE_NAME automatically for each service.
set -euo pipefail

SERVICE="${RAILWAY_SERVICE_NAME:-}"

if [ "$SERVICE" = "a2a-agent" ]; then
  echo "Starting A2A agent (service: $SERVICE)"
  exec python src/a2a_agent/server.py
else
  echo "Starting MCP server (service: ${SERVICE:-unknown})"
  exec python src/mcp_server/server.py
fi
