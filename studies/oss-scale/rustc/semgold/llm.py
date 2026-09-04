"""Two-family LLM caller with a disk cache. Backends: openai (gpt-5.6-terra, flex, no
reasoning: the paper's convention) and anthropic (Claude via API). Cache key = model+prompt."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import CACHE

_lock = threading.Lock()
_clients: dict = {}


def _key(model: str, prompt: str, salt: str = "") -> str:
    tail = ("\n\x00" + salt) if salt else ""
    return hashlib.sha256((model + tail + "\n\x00" + prompt).encode()).hexdigest()


def _openai(model: str, prompt: str, timeout: int) -> tuple[str, dict]:
    import openai
    with _lock:
        if "openai" not in _clients:
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OAI_KEY")
            _clients["openai"] = openai.OpenAI(api_key=key, timeout=timeout)
    client = _clients["openai"]
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}],
                  max_completion_tokens=4096, service_tier=os.environ.get("OPENAI_SERVICE_TIER", "flex"))
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "none")
    if effort and effort != "none":
        kwargs["reasoning_effort"] = effort
    resp = client.chat.completions.create(**kwargs)
    usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
    return resp.choices[0].message.content or "", usage


def _anthropic(model: str, prompt: str, timeout: int) -> tuple[str, dict]:
    import anthropic
    with _lock:
        if "anthropic" not in _clients:
            _clients["anthropic"] = anthropic.Anthropic(timeout=timeout)
    client = _clients["anthropic"]
    resp = client.messages.create(model=model, max_tokens=4096,
                                  messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    usage = {"prompt_tokens": resp.usage.input_tokens, "completion_tokens": resp.usage.output_tokens}
    return text, usage


BACKENDS = {"openai": _openai, "anthropic": _anthropic}


def call(backend: str, model: str, prompt: str, timeout: int = 300, retries: int = 4, salt: str = "") -> dict:
    """Returns {text, usage, cached, model}. Cached on disk under semgold/cache/<backend>/."""
    cdir = CACHE / backend
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / (_key(model, prompt, salt) + ".json")
    if path.exists():
        data = json.loads(path.read_text())
        data["cached"] = True
        return data
    delay = 5.0
    last = None
    for attempt in range(retries):
        try:
            text, usage = BACKENDS[backend](model, prompt, timeout)
            data = {"model": model, "text": text, "usage": usage, "prompt_sha": _key(model, prompt), "salt": salt}
            path.write_text(json.dumps(data))
            data["cached"] = False
            return data
        except Exception as exc:  # rate limits, timeouts, 5xx
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 90)
    raise RuntimeError(f"{backend}/{model} failed after {retries} tries: {last}")


def call_many(backend: str, model: str, prompts: list[str], workers: int = 8, progress: str = "", salt: str = "") -> list[dict]:
    out: list = [None] * len(prompts)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(call, backend, model, p, 300, 4, salt): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
            done += 1
            if progress and done % 25 == 0:
                print(f"[{progress}] {done}/{len(prompts)}", flush=True)
    return out


def extract_json(text: str):
    """First balanced JSON object/array in text (tolerates ``` fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    return None


def _claude_cli(model: str, prompt: str, timeout: int) -> tuple[str, dict]:
    """Second model family via the local Claude Code CLI (same route approach/ uses)."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = ["claude", "-p", "--output-format", "json", "--model", model, prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    data = json.loads(res.stdout.strip() or "{}")
    if data.get("type") != "result" or data.get("is_error"):
        raise RuntimeError(res.stderr[:300] or data.get("result", "")[:300] or "claude cli: no result")
    u = data.get("usage", {})
    usage = {"prompt_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
             "completion_tokens": u.get("output_tokens", 0)}
    return data["result"], usage


BACKENDS["claude_cli"] = _claude_cli


def extract_entries(text: str) -> dict:
    """Tolerant fallback for truncated per-sentence JSON: pull every complete
    "<number>": {...} entry."""
    import re
    out = {}
    for m in re.finditer(r'"(\d+)"\s*:\s*(\{)', text):
        start = m.start(2)
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        out[m.group(1)] = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
    return out
