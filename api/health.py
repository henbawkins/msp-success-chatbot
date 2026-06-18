"""Diagnostic: reports which env vars are present (booleans only, never values)."""
import os, json
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        status = {k: bool(os.environ.get(k)) for k in (
            "GEMINI_API_KEY", "SUPABASE_URL", "SUPABASE_SECRET_KEY",
            "ANTHROPIC_API_KEY", "APP_PASSWORD")}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"env_present": status}).encode())
