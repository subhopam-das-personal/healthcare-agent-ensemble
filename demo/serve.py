"""
Demo script server — serves demo/script.md as rendered HTML.
Any edits to script.md are live on next page refresh.

Usage:
    python demo/serve.py          # default port 7777
    python demo/serve.py 9090     # custom port

Access from office laptop:
    http://<this-machine-ip>:7777
"""

import http.server
import os
import socketserver
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 7777))
SCRIPT_MD = Path(__file__).parent / "script.md"

HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Demo Script — Healthcare Agent Ensemble</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e6edf3;
    padding: 32px 24px;
    max-width: 860px;
    margin: 0 auto;
    line-height: 1.65;
  }}
  #toolbar {{
    position: fixed;
    top: 0; left: 0; right: 0;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    z-index: 100;
  }}
  #toolbar h1 {{
    font-size: 14px;
    font-weight: 600;
    color: #58a6ff;
    flex: 1;
  }}
  #toolbar .ts {{
    font-size: 12px;
    color: #8b949e;
  }}
  #refresh-btn {{
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
    cursor: pointer;
    font-weight: 600;
  }}
  #refresh-btn:hover {{ background: #2ea043; }}
  #content {{ margin-top: 56px; }}
  h1, h2, h3 {{
    color: #e6edf3;
    margin: 28px 0 12px;
    font-weight: 600;
  }}
  h1 {{ font-size: 22px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
  h2 {{ font-size: 18px; color: #58a6ff; }}
  h3 {{ font-size: 15px; color: #79c0ff; }}
  p {{ margin: 10px 0; }}
  blockquote {{
    border-left: 3px solid #388bfd;
    background: #161b22;
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    color: #c9d1d9;
    font-style: italic;
  }}
  code {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 13px;
    color: #f0883e;
  }}
  pre {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    margin: 12px 0;
  }}
  pre code {{
    background: none;
    border: none;
    padding: 0;
    color: #e6edf3;
    font-size: 13px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
  }}
  th {{
    background: #161b22;
    color: #58a6ff;
    padding: 10px 14px;
    text-align: left;
    border: 1px solid #30363d;
    font-weight: 600;
  }}
  td {{
    padding: 9px 14px;
    border: 1px solid #30363d;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{ background: #0d1117; }}
  hr {{ border: none; border-top: 1px solid #30363d; margin: 24px 0; }}
  a {{ color: #58a6ff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  ul, ol {{ margin: 10px 0 10px 24px; }}
  li {{ margin: 4px 0; }}
  strong {{ color: #f0f6fc; }}
  .badge-checkpoint {{
    display: inline-block;
    background: #1f6feb33;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
  }}
</style>
</head>
<body>
<div id="toolbar">
  <h1>Healthcare Agent Ensemble — Demo Script</h1>
  <span class="ts" id="ts"></span>
  <button id="refresh-btn" onclick="loadScript()">↻ Refresh</button>
</div>
<div id="content">Loading...</div>
<script>
async function loadScript() {{
  try {{
    const r = await fetch('/raw?t=' + Date.now());
    const md = await r.text();
    document.getElementById('content').innerHTML = marked.parse(md);
    document.getElementById('ts').textContent = 'Updated ' + new Date().toLocaleTimeString();
  }} catch(e) {{
    document.getElementById('content').innerHTML = '<p style="color:#f85149">Failed to load script.md — ' + e + '</p>';
  }}
}}
loadScript();
// Auto-refresh every 30 seconds
setInterval(loadScript, 30000);
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._serve_html()
        elif self.path.startswith("/raw"):
            self._serve_raw()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        body = HTML_SHELL.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_raw(self):
        try:
            body = SCRIPT_MD.read_bytes()
        except FileNotFoundError:
            body = b"# script.md not found"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress request noise


def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    ip = get_local_ip()
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"\n  Demo script server running")
        print(f"\n  Local:   http://localhost:{PORT}")
        print(f"  Network: http://{ip}:{PORT}  ← open this on your office laptop\n")
        print(f"  Serving: {SCRIPT_MD}")
        print(f"  Auto-refresh: every 30 seconds\n")
        print("  Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")
