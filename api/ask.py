"""Vercel Python serverless function: RAG over Digital Marketing Hour webinars.
Flow: embed question (Gemini) -> match_chunks (Supabase pgvector) -> answer (Claude), grounded + cited.
"""
import os, json, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

EMBED_MODEL = "models/gemini-embedding-001"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MATCH_COUNT = 8

SYSTEM_PROMPT = """You are the internal knowledge assistant for a digital marketing agency team. \
Your knowledge comes ONLY from two internal sources:
1. Transcripts of the team's "Digital Marketing Hour" webinars (recurring Q&A on SEO, local search / Google \
Business Profile, PPC, content, and social media for MSP/IT clients).
2. The "MSP Success Q&A" support dashboard — answered support/process/technical questions from the team and clients.

Rules:
- Answer ONLY using the provided excerpts. Do NOT use outside knowledge or general marketing advice.
- If the excerpts do not contain the answer, say so plainly: "I don't have anything on that in the webinars or the Q&A dashboard." Do not guess.
- Cite the source label given in brackets above each excerpt, e.g. (Digital Marketing Hour - 06.05.25) or (MSP Success Q&A - Q012).
- Be concise and practical, in the team's voice. Prefer specifics the team actually said over generic statements.
- Some excerpts are spoken-transcript text, so tolerate minor transcription noise and infer intent reasonably."""


def _post_json(url, payload, headers, timeout=60):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def embed_question(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:embedContent?key={GEMINI_KEY}"
    payload = {"content": {"parts": [{"text": text}]}, "outputDimensionality": 768,
               "taskType": "RETRIEVAL_QUERY"}
    d = _post_json(url, payload, {"Content-Type": "application/json"}, timeout=30)
    return d["embedding"]["values"]


def retrieve(vec):
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_chunks"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json"}
    return _post_json(url, {"query_embedding": vec, "match_count": MATCH_COUNT}, headers, timeout=30)


def log_event(fields):
    """Fire-and-forget usage log to Supabase. Never raises into the request path."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/chat_logs"
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                   "Content-Type": "application/json", "Prefer": "return=minimal"}
        req = urllib.request.Request(url, data=json.dumps(fields).encode(), headers=headers)
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def build_context(chunks):
    parts = []
    for c in chunks:
        label = c.get("webinar_title") or "Digital Marketing Hour"
        parts.append(f"[{label}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def ask_claude(question, context, history=None):
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    user_msg = (f"Here are relevant excerpts from our webinars and Q&A dashboard:\n\n{context}\n\n"
                f"Question: {question}\n\nAnswer using only these excerpts, and cite the source label(s).")
    messages = []
    for turn in (history or []):
        role = turn.get("role")
        text = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_msg})
    payload = {"model": CLAUDE_MODEL, "max_tokens": 1024, "system": SYSTEM_PROMPT,
               "messages": messages}
    d = _post_json(url, payload, headers, timeout=60)
    return "".join(b.get("text", "") for b in d.get("content", []))


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad request"})

        if APP_PASSWORD and req.get("password") != APP_PASSWORD:
            return self._send(401, {"error": "unauthorized"})

        question = (req.get("question") or "").strip()
        if not question:
            return self._send(400, {"error": "no question"})

        history = req.get("history") or []
        if not isinstance(history, list):
            history = []
        history = history[-6:]  # cap context to the last few turns

        # For retrieval, blend the previous user turn into the query so vague
        # follow-ups ("what about Bing?") still pull relevant excerpts.
        prev_user = next((t.get("content", "") for t in reversed(history)
                          if t.get("role") == "user"), "")
        retrieval_query = (prev_user + " " + question).strip() if prev_user else question

        try:
            vec = embed_question(retrieval_query)
            chunks = retrieve(vec)
            if not chunks:
                log_event({"question": question, "no_results": True, "num_sources": 0})
                return self._send(200, {"answer": "The webinars don't cover that directly.", "sources": []})
            answer = ask_claude(question, build_context(chunks), history)
            seen, sources = set(), []
            for c in chunks:
                label = c.get("webinar_title") or c.get("webinar_date")
                if label and label not in seen:
                    seen.add(label)
                    sources.append({"label": label, "similarity": round(c["similarity"], 3)})
            log_event({"question": question, "answer_len": len(answer),
                       "top_source": sources[0]["label"] if sources else None,
                       "top_sim": sources[0]["similarity"] if sources else None,
                       "num_sources": len(sources)})
            return self._send(200, {"answer": answer, "sources": sources})
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            log_event({"question": question, "error": f"upstream {e.code}: {detail[:150]}"})
            return self._send(502, {"error": f"upstream {e.code}", "detail": detail})
        except Exception as e:
            log_event({"question": question, "error": str(e)[:200]})
            return self._send(500, {"error": str(e)})
