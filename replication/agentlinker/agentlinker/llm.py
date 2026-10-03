import json
import os
import time
from collections import defaultdict, deque

SYSTEM_PROMPT = ("You are a helpful assistant that analyzes software architecture documents "
                 "and extracts trace links between documentation and architecture models. "
                 "Always respond with valid JSON when asked.")


class OpenAIChat:
    def __init__(self, model, attempts=5):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.attempts = attempts
        self.tier = os.environ.get("OPENAI_SERVICE_TIER", "flex")
        self.calls = []

    def ask(self, phase, prompt, timeout):
        for attempt in range(self.attempts):
            started = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    seed=42,
                    max_completion_tokens=4096,
                    reasoning_effort="none",
                    service_tier=self.tier,
                    timeout=timeout * 2 if self.tier == "flex" else timeout,
                )
                break
            except Exception as error:
                if attempt == self.attempts - 1:
                    raise
                print(f"    {phase}: {error}; retrying")
                time.sleep(2 ** attempt * 2 + 1)
        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise RuntimeError(f"{phase}: empty response")
        usage = response.usage
        self.calls.append({
            "phase": phase,
            "prompt": prompt,
            "response": text,
            "model": response.model,
            "usage": {"prompt_tokens": usage.prompt_tokens,
                      "completion_tokens": usage.completion_tokens,
                      "total_tokens": usage.total_tokens} if usage else None,
            "latency_ms": int((time.time() - started) * 1000),
        })
        return text


class ReplayChat:
    def __init__(self, path):
        with open(path, encoding="utf-8") as handle:
            self.calls = json.load(handle)
        self.pending = defaultdict(deque)
        for call in self.calls:
            self.pending[call["prompt"]].append(call["response"])

    def ask(self, phase, prompt, timeout):
        if not self.pending[prompt]:
            raise RuntimeError(f"{phase}: prompt not in the recorded calls")
        return self.pending[prompt].popleft()

    def unused(self):
        return sum(len(responses) for responses in self.pending.values())


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:end + 1])
                except ValueError:
                    break
    return None


def ask_json(chat, phase, prompt, timeout, require=None, require_present=None, attempts=2):
    data = {}
    for _ in range(attempts):
        data = parse_json(chat.ask(phase, prompt, timeout)) or {}
        if not data:
            continue
        if require_present is not None:
            if require_present in data:
                return data
        elif require is not None:
            if data.get(require):
                return data
        else:
            return data
    return data
