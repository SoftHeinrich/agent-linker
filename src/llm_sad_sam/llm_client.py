"""LLM Client abstraction for supporting multiple backends.

Supported backends:
- codex: OpenAI Codex CLI (codex exec)
- claude: Local Claude Code CLI (claude -p)
- openai: OpenAI API with GPT-5.2 (requires OPENAI_API_KEY)
"""

import subprocess
import json
import re
import os
import logging
import diskcache
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class LLMBackend(Enum):
    CODEX = "codex"
    CLAUDE = "claude"
    OPENAI = "openai"
    CHECKPOINT = "checkpoint"


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    success: bool
    error: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None


@dataclass
class ConversationMessage:
    """A message in a conversation."""
    role: str  # "user" or "assistant" or "system"
    content: str


class LLMClient:
    """Unified LLM client supporting multiple backends with logging and token tracking.

    Supports two modes:
    1. Stateless queries: Each query() call is independent
    2. Conversation mode: Use start_conversation(), query_conversation(), end_conversation()
       to maintain context across multiple queries
    """

    # Default backend - can be overridden via environment variable
    _default_backend: LLMBackend = LLMBackend.CLAUDE

    # Cumulative token usage across all instances (class-level)
    _cumulative_usage: TokenUsage = TokenUsage()
    _total_requests: int = 0
    _total_errors: int = 0

    def __init__(self, backend: Optional[LLMBackend] = None, model: Optional[str] = None,
                 log_dir: Optional[str] = None, enable_logging: bool = True,
                 temperature: Optional[float] = None,
                 checkpoint_fallback: Optional[LLMBackend | str] = None,
                 checkpoint_fallback_model: Optional[str] = None):
        """Initialize LLM client.

        Args:
            backend: LLM backend to use. If None, uses LLM_BACKEND env var or default.
            model: Model name. For OpenAI: defaults to OPENAI_MODEL_NAME env var or "gpt-5.2".
                   For Claude CLI: passed as --model flag. Defaults to CLAUDE_MODEL env var (unset = CLI default).
            log_dir: Directory to save logs. Defaults to LLM_LOG_DIR env var or "./results/llm_logs".
            enable_logging: Whether to enable file logging. Defaults to True.
            temperature: Temperature for generation (0.0-1.0). Only works with OpenAI backend.
                        Lower = more deterministic. Default 0.1 for OpenAI.
            checkpoint_fallback: Backend used for checkpoint cache misses.
                                 Defaults to CHECKPOINT_FALLBACK env var or "claude".
            checkpoint_fallback_model: Model used on checkpoint cache misses.
                                       Examples: "sonnet", "gpt", "gpt-5.2".
        """
        if backend is not None:
            self.backend = backend
        else:
            env_backend = os.environ.get("LLM_BACKEND", "").lower()
            if env_backend == "claude":
                self.backend = LLMBackend.CLAUDE
            elif env_backend == "codex":
                self.backend = LLMBackend.CODEX
            elif env_backend == "openai":
                self.backend = LLMBackend.OPENAI
            elif env_backend == "checkpoint":
                self.backend = LLMBackend.CHECKPOINT
            else:
                self.backend = self._default_backend

        # Initialize these early so partially constructed instances can close cleanly.
        self._logger = None
        self._log_file_handle = None

        # Model configuration
        self.openai_model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
        self.claude_model = os.environ.get("CLAUDE_MODEL", "sonnet")
        if model is not None:
            if self.backend == LLMBackend.OPENAI:
                self.openai_model = model
            elif self.backend == LLMBackend.CLAUDE:
                self.claude_model = model
            elif self.backend == LLMBackend.CHECKPOINT and checkpoint_fallback_model is None:
                checkpoint_fallback_model = model
        self.temperature = temperature if temperature is not None else 0.1
        self._openai_client = None

        # Checkpoint backend configuration
        self._checkpoint_dir: Optional[Path] = None
        self._checkpoint_fallback: Optional[LLMBackend] = None
        self._checkpoint_fallback_model: Optional[str] = None
        self._cache: Optional[diskcache.Cache] = None
        if self.backend == LLMBackend.CHECKPOINT:
            self._checkpoint_dir = Path(
                os.environ.get("CHECKPOINT_DIR", "./results/llm_checkpoint")
            )
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(self._checkpoint_dir))
            fallback = checkpoint_fallback
            if fallback is None:
                fallback = os.environ.get("CHECKPOINT_FALLBACK")
            fallback_model = checkpoint_fallback_model or os.environ.get("CHECKPOINT_FALLBACK_MODEL")
            self._checkpoint_fallback, self._checkpoint_fallback_model = (
                self._resolve_checkpoint_target(fallback, fallback_model)
            )
            if self._checkpoint_fallback == LLMBackend.OPENAI and self._checkpoint_fallback_model:
                self.openai_model = self._checkpoint_fallback_model
            elif self._checkpoint_fallback == LLMBackend.CLAUDE and self._checkpoint_fallback_model:
                self.claude_model = self._checkpoint_fallback_model

        # Logging configuration
        self.enable_logging = enable_logging
        self.log_dir = Path(log_dir or os.environ.get("LLM_LOG_DIR", "./results/llm_logs"))
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._request_count = 0
        self._session_usage = TokenUsage()
        self._log_file_handle = None  # Keep open to avoid fd churn

        # Dedicated working directory for CLI subprocesses (avoids cwd side-effects)
        self._subprocess_cwd = Path(
            os.environ.get("LLM_SESSION_DIR", Path.home() / ".llm-sad-sam" / "sessions")
        ) / self._session_id
        self._subprocess_cwd.mkdir(parents=True, exist_ok=True)

        # Conversation state
        self._conversation_active: bool = False
        self._conversation_history: list[ConversationMessage] = []
        self._conversation_system_prompt: Optional[str] = None
        self._claude_resume_id: Optional[str] = None  # For Claude CLI --resume
        self._conversation_token_count: int = 0
        self._max_conversation_tokens: int = 50000  # Trim history if exceeded

        if self.enable_logging:
            self._setup_logging()

    @staticmethod
    def _parse_backend(value: Optional[LLMBackend | str]) -> Optional[LLMBackend]:
        """Parse a backend enum or name."""
        if value is None:
            return None
        if isinstance(value, LLMBackend):
            return value

        normalized = value.strip().lower()
        mapping = {
            "codex": LLMBackend.CODEX,
            "claude": LLMBackend.CLAUDE,
            "openai": LLMBackend.OPENAI,
            "checkpoint": LLMBackend.CHECKPOINT,
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown LLM backend: {value}")
        return mapping[normalized]

    @staticmethod
    def _infer_backend_from_model(model_name: Optional[str]) -> tuple[Optional[LLMBackend], Optional[str]]:
        """Infer fallback backend from a human-friendly model selector."""
        if not model_name:
            return None, None

        normalized = model_name.strip()
        lowered = normalized.lower()
        if lowered == "gpt":
            return LLMBackend.OPENAI, os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
        if lowered == "openai":
            return LLMBackend.OPENAI, os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
        if lowered.startswith("gpt"):
            return LLMBackend.OPENAI, normalized

        if lowered == "sonnet":
            return LLMBackend.CLAUDE, "sonnet"
        if lowered == "claude":
            return LLMBackend.CLAUDE, os.environ.get("CLAUDE_MODEL", "sonnet")
        if lowered.startswith("claude"):
            return LLMBackend.CLAUDE, normalized

        return None, normalized

    def _resolve_checkpoint_target(
        self,
        fallback: Optional[LLMBackend | str],
        fallback_model: Optional[str],
    ) -> tuple[LLMBackend, Optional[str]]:
        """Resolve checkpoint cache-miss backend + model."""
        requested_fallback = self._parse_backend(fallback)
        inferred_backend, normalized_model = self._infer_backend_from_model(fallback_model)

        if inferred_backend is not None and requested_fallback is not None and requested_fallback != inferred_backend:
            raise ValueError(
                f"Checkpoint fallback backend/model conflict: backend={requested_fallback.value}, "
                f"model={fallback_model}"
            )
        resolved_fallback = inferred_backend or requested_fallback or LLMBackend.CLAUDE

        if resolved_fallback == LLMBackend.OPENAI:
            return resolved_fallback, normalized_model or self.openai_model
        if resolved_fallback == LLMBackend.CLAUDE:
            return resolved_fallback, normalized_model or self.claude_model
        return resolved_fallback, None

    def get_active_model(self, backend: Optional[LLMBackend] = None) -> Optional[str]:
        """Return the model configured for a backend."""
        backend = backend or self.backend
        if backend == LLMBackend.OPENAI:
            return self.openai_model
        if backend == LLMBackend.CLAUDE:
            return self.claude_model
        if backend == LLMBackend.CHECKPOINT:
            return self.get_active_model(self._checkpoint_fallback)
        return None

    def describe_backend(self) -> str:
        """Human-readable backend/model summary."""
        if self.backend == LLMBackend.CHECKPOINT:
            fallback = self._checkpoint_fallback.value if self._checkpoint_fallback else "unknown"
            model = self.get_active_model(self._checkpoint_fallback) or "default"
            return f"checkpoint -> {fallback} ({model})"
        model = self.get_active_model() or "default"
        return f"{self.backend.value} ({model})"

    def close(self):
        """Close open file handles."""
        if self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.close()
            self._log_file_handle = None

    def __del__(self):
        self.close()

    # ==================== Conversation Mode ====================

    def start_conversation(self, system_prompt: Optional[str] = None) -> None:
        """Start a new conversation with optional system prompt.

        In conversation mode, context is maintained across queries.

        Args:
            system_prompt: Optional system prompt to set context
        """
        self._conversation_active = True
        self._conversation_history = []
        self._conversation_system_prompt = system_prompt or (
            "You are a helpful assistant that analyzes software architecture documents "
            "and extracts trace links between documentation and architecture models. "
            "Remember context from our previous exchanges in this conversation."
        )
        self._claude_resume_id = None
        self._conversation_token_count = 0

        if self._logger:
            self._logger.info(f"Conversation started | system_prompt_len={len(self._conversation_system_prompt)}")

    def query_conversation(self, prompt: str, timeout: int = 180) -> LLMResponse:
        """Query within an active conversation, maintaining context.

        Args:
            prompt: The user's message
            timeout: Timeout in seconds

        Returns:
            LLMResponse with the result
        """
        if not self._conversation_active:
            # Auto-start conversation if not active
            self.start_conversation()

        import time
        start_time = time.time()

        # Add user message to history
        self._conversation_history.append(ConversationMessage(role="user", content=prompt))
        self._conversation_token_count += len(prompt) // 4  # Rough token estimate

        # Query based on backend
        if self.backend == LLMBackend.OPENAI:
            response = self._query_openai_conversation(timeout)
        elif self.backend == LLMBackend.CLAUDE:
            response = self._query_claude_conversation(prompt, timeout)
        else:
            # Fallback to stateless for unsupported backends
            response = self._query_codex(prompt, timeout)

        # Add assistant response to history
        if response.success and response.text:
            self._conversation_history.append(ConversationMessage(role="assistant", content=response.text))
            self._conversation_token_count += len(response.text) // 4

        # Trim history if too long
        self._maybe_trim_history()

        latency_ms = int((time.time() - start_time) * 1000)
        response.latency_ms = latency_ms

        if self.enable_logging:
            self._log_request(prompt, response, latency_ms)

        return response

    def end_conversation(self) -> list[ConversationMessage]:
        """End the current conversation and return the history.

        Returns:
            List of all messages in the conversation
        """
        history = self._conversation_history.copy()
        self._conversation_active = False
        self._conversation_history = []
        self._conversation_system_prompt = None
        self._claude_resume_id = None
        self._conversation_token_count = 0

        if self._logger:
            self._logger.info(f"Conversation ended | messages={len(history)}")

        return history

    def get_conversation_context(self) -> str:
        """Get a summary of the current conversation context.

        Returns:
            String summary of conversation history
        """
        if not self._conversation_history:
            return ""

        summary_parts = []
        for msg in self._conversation_history[-10:]:  # Last 10 messages
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"[{msg.role}]: {preview}")

        return "\n".join(summary_parts)

    def is_conversation_active(self) -> bool:
        """Check if a conversation is currently active."""
        return self._conversation_active

    def _maybe_trim_history(self) -> None:
        """Trim conversation history if it's too long."""
        if self._conversation_token_count > self._max_conversation_tokens:
            # Keep system prompt context and last N messages
            if len(self._conversation_history) > 10:
                # Summarize early messages
                trimmed = self._conversation_history[-8:]
                summary = f"[Earlier context: {len(self._conversation_history) - 8} messages about document analysis]"
                self._conversation_history = [
                    ConversationMessage(role="system", content=summary)
                ] + trimmed
                self._conversation_token_count = sum(len(m.content) // 4 for m in self._conversation_history)

    def _query_openai_conversation(self, timeout: int) -> LLMResponse:
        """Query OpenAI with full conversation history."""
        try:
            client = self._get_openai_client()
        except (ImportError, ValueError) as e:
            return LLMResponse(text="", success=False, error=str(e))

        # Build messages array
        messages = []
        if self._conversation_system_prompt:
            messages.append({"role": "system", "content": self._conversation_system_prompt})

        for msg in self._conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        try:
            create_kwargs = dict(
                model=self.openai_model,
                messages=messages,
                temperature=self.temperature,
                seed=42,
                max_completion_tokens=4096,
                timeout=timeout,
            )
            create_kwargs["service_tier"] = os.environ.get("OPENAI_SERVICE_TIER", "flex")
            response = client.chat.completions.create(**create_kwargs)

            token_usage = None
            if response.usage:
                token_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0
                )

            if response.choices and len(response.choices) > 0:
                text = response.choices[0].message.content
                if text:
                    return LLMResponse(
                        text=text, success=True, token_usage=token_usage, model=response.model
                    )

            return LLMResponse(text="", success=False, error="No response", token_usage=token_usage)

        except Exception as e:
            return LLMResponse(text="", success=False, error=str(e))

    def _query_claude_conversation(self, prompt: str, timeout: int) -> LLMResponse:
        """Query Claude with conversation resume support."""
        try:
            cmd = ["claude", "-p", "--output-format", "json", "--dangerously-skip-permissions"]
            if self.claude_model:
                cmd.extend(["--model", self.claude_model])

            # Use --resume if we have a previous conversation ID
            if self._claude_resume_id:
                cmd.extend(["--resume", self._claude_resume_id])

            cmd.append(prompt)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                    cwd=self._subprocess_cwd)

            response_text = ""
            conversation_id = None

            # Try parsing response
            try:
                data = json.loads(result.stdout.strip())
                if data.get('type') == 'result':
                    response_text = data.get('result', '')
                    conversation_id = data.get('session_id') or data.get('conversation_id')
            except json.JSONDecodeError:
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'result':
                            response_text = event.get('result', '')
                            conversation_id = event.get('session_id') or event.get('conversation_id')
                            break
                    except json.JSONDecodeError:
                        continue

            if not response_text and result.stdout.strip():
                response_text = result.stdout.strip()

            # Store conversation ID for resume
            if conversation_id:
                self._claude_resume_id = conversation_id

            if response_text:
                return LLMResponse(text=response_text, success=True)
            else:
                error_msg = result.stderr.strip() if result.stderr else "No response"
                return LLMResponse(text="", success=False, error=error_msg)

        except subprocess.TimeoutExpired:
            return LLMResponse(text="", success=False, error="Claude request timed out")
        except Exception as e:
            return LLMResponse(text="", success=False, error=str(e))

    def _setup_logging(self):
        """Setup file logging for this session."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create session log file
        log_file = self.log_dir / f"llm_session_{self._session_id}.log"

        self._logger = logging.getLogger(f"llm_client_{self._session_id}")
        self._logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if not self._logger.handlers:
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._logger.info(f"Session started | target={self.describe_backend()}")

    def _log_request(self, prompt: str, response: LLMResponse, latency_ms: int):
        """Log a request/response pair."""
        if not self._logger:
            return

        self._request_count += 1
        LLMClient._total_requests += 1

        # Update token usage
        if response.token_usage:
            self._session_usage.prompt_tokens += response.token_usage.prompt_tokens
            self._session_usage.completion_tokens += response.token_usage.completion_tokens
            self._session_usage.total_tokens += response.token_usage.total_tokens
            LLMClient._cumulative_usage.prompt_tokens += response.token_usage.prompt_tokens
            LLMClient._cumulative_usage.completion_tokens += response.token_usage.completion_tokens
            LLMClient._cumulative_usage.total_tokens += response.token_usage.total_tokens

        if not response.success:
            LLMClient._total_errors += 1

        # Log summary
        status = "OK" if response.success else f"ERROR: {response.error}"
        tokens = f"tokens={response.token_usage.total_tokens}" if response.token_usage else "tokens=N/A"
        self._logger.info(f"REQ#{self._request_count} | {latency_ms}ms | {tokens} | {status}")

        # Log full request/response to separate file for debugging
        detail_file = self.log_dir / f"llm_requests_{self._session_id}.jsonl"
        log_entry = {
            "request_id": self._request_count,
            "timestamp": datetime.now().isoformat(),
            "backend": self.backend.value,
            "model": response.model or self.get_active_model(),
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_length": len(response.text) if response.text else 0,
            "response_preview": (response.text[:200] + "...") if response.text and len(response.text) > 200 else response.text,
            "success": response.success,
            "error": response.error,
            "latency_ms": latency_ms,
            "token_usage": {
                "prompt_tokens": response.token_usage.prompt_tokens if response.token_usage else None,
                "completion_tokens": response.token_usage.completion_tokens if response.token_usage else None,
                "total_tokens": response.token_usage.total_tokens if response.token_usage else None,
            } if response.token_usage else None
        }
        if self._log_file_handle is None or self._log_file_handle.closed:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file_handle = open(detail_file, 'a', encoding='utf-8')
        self._log_file_handle.write(json.dumps(log_entry) + "\n")
        self._log_file_handle.flush()

    def get_session_usage(self) -> dict:
        """Get token usage for this session."""
        return {
            "session_id": self._session_id,
            "request_count": self._request_count,
            "prompt_tokens": self._session_usage.prompt_tokens,
            "completion_tokens": self._session_usage.completion_tokens,
            "total_tokens": self._session_usage.total_tokens,
        }

    @classmethod
    def get_cumulative_usage(cls) -> dict:
        """Get cumulative token usage across all sessions."""
        return {
            "total_requests": cls._total_requests,
            "total_errors": cls._total_errors,
            "prompt_tokens": cls._cumulative_usage.prompt_tokens,
            "completion_tokens": cls._cumulative_usage.completion_tokens,
            "total_tokens": cls._cumulative_usage.total_tokens,
        }

    @classmethod
    def reset_cumulative_usage(cls):
        """Reset cumulative usage statistics."""
        cls._cumulative_usage = TokenUsage()
        cls._total_requests = 0
        cls._total_errors = 0

    @classmethod
    def set_default_backend(cls, backend: LLMBackend):
        """Set the default backend for all new instances."""
        cls._default_backend = backend

    def enable_checkpoint(self, checkpoint_dir: str,
                          fallback: Optional[LLMBackend | str] = None,
                          fallback_model: Optional[str] = None) -> None:
        """Switch this client to checkpoint mode.

        Existing responses in checkpoint_dir are loaded on cache hit;
        cache misses delegate to the fallback backend and save the result.

        Args:
            checkpoint_dir: Directory to store/load cached LLM responses.
            fallback: Backend for cache misses. Defaults to current backend.
            fallback_model: Model used on cache misses.
        """
        if fallback is None:
            if self.backend == LLMBackend.CHECKPOINT:
                fallback = self._checkpoint_fallback or LLMBackend.CLAUDE
            else:
                fallback = self.backend
        self._checkpoint_fallback, self._checkpoint_fallback_model = (
            self._resolve_checkpoint_target(fallback, fallback_model)
        )
        if self._checkpoint_fallback == LLMBackend.OPENAI and self._checkpoint_fallback_model:
            self.openai_model = self._checkpoint_fallback_model
        elif self._checkpoint_fallback == LLMBackend.CLAUDE and self._checkpoint_fallback_model:
            self.claude_model = self._checkpoint_fallback_model
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self._cache is not None:
            self._cache.close()
        self._cache = diskcache.Cache(str(self._checkpoint_dir))
        self.backend = LLMBackend.CHECKPOINT

    def save_usage_summary(self, output_path: Optional[str] = None):
        """Save usage summary to a JSON file."""
        if output_path is None:
            output_path = self.log_dir / f"usage_summary_{self._session_id}.json"

        summary = {
            "session": self.get_session_usage(),
            "cumulative": self.get_cumulative_usage(),
            "backend": self.backend.value,
            "model": self.get_active_model(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self._logger:
            self._logger.info(f"Usage summary saved to {output_path}")

        return summary

    def query(self, prompt: str, timeout: int = 180, max_retries: int = 3) -> LLMResponse:
        """Send a prompt to the LLM and get a response.

        Retries on transient failures (timeout, connection errors) with
        exponential backoff. Raises RuntimeError after all retries exhausted.

        For the checkpoint backend, responses are cached by prompt hash:
        cache hit returns instantly; cache miss delegates to the fallback
        backend (with retries), then saves the response.

        Args:
            prompt: The prompt to send
            timeout: Timeout in seconds
            max_retries: Max retry attempts for transient errors (default 3)

        Returns:
            LLMResponse with the result

        Raises:
            RuntimeError: If all retries exhausted on transient errors
        """
        import time

        # Checkpoint backend: check cache first, delegate on miss
        if self.backend == LLMBackend.CHECKPOINT:
            return self._query_checkpoint(prompt, timeout, max_retries)

        last_response = None
        for attempt in range(max_retries):
            start_time = time.time()

            if self.backend == LLMBackend.CODEX:
                response = self._query_codex(prompt, timeout)
            elif self.backend == LLMBackend.CLAUDE:
                response = self._query_claude(prompt, timeout)
            elif self.backend == LLMBackend.OPENAI:
                response = self._query_openai(prompt, timeout)
            else:
                response = LLMResponse(text="", success=False, error=f"Unknown backend: {self.backend}")

            latency_ms = int((time.time() - start_time) * 1000)
            response.latency_ms = latency_ms

            # Log the request
            if self.enable_logging:
                self._log_request(prompt, response, latency_ms)

            # Success — return immediately
            if response.success:
                return response

            last_response = response

            # Check if error is transient (retryable)
            error_lower = (response.error or "").lower()
            is_transient = any(keyword in error_lower for keyword in [
                "timed out", "timeout", "connection", "503", "502", "504",
                "overloaded", "rate_limit", "rate limit", "server_error",
            ])

            if not is_transient:
                # Non-transient error (e.g., bad prompt, auth failure) — don't retry
                return response

            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2 + 1  # 3s, 5s, 9s
                print(f"    LLM error: {response.error} — retrying in {wait_time}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        # All retries exhausted on transient error — raise
        raise RuntimeError(
            f"LLM query failed after {max_retries} retries: {last_response.error}"
        )

    # ==================== Checkpoint Backend ====================

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        """Deterministic SHA-256 hash of a prompt string, used as cache key."""
        import hashlib
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def _llm_response_to_dict(response: LLMResponse) -> dict:
        """Serialize an LLMResponse to a plain dict for cache storage."""
        return {
            "text": response.text,
            "success": response.success,
            "error": response.error,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "token_usage": {
                "prompt_tokens": response.token_usage.prompt_tokens,
                "completion_tokens": response.token_usage.completion_tokens,
                "total_tokens": response.token_usage.total_tokens,
            } if response.token_usage else None,
        }

    @staticmethod
    def _llm_response_from_dict(data: dict) -> LLMResponse:
        """Reconstruct an LLMResponse from a cached dict."""
        token_usage = None
        if data.get("token_usage"):
            tu = data["token_usage"]
            token_usage = TokenUsage(
                prompt_tokens=tu.get("prompt_tokens", 0),
                completion_tokens=tu.get("completion_tokens", 0),
                total_tokens=tu.get("total_tokens", 0),
            )
        return LLMResponse(
            text=data["text"],
            success=data["success"],
            error=data.get("error"),
            token_usage=token_usage,
            model=data.get("model"),
            latency_ms=data.get("latency_ms"),
        )

    def _query_checkpoint(self, prompt: str, timeout: int, max_retries: int) -> LLMResponse:
        """Checkpoint backend: load cached response or call fallback and save.

        Cache key is SHA-256 of the prompt text. On hit, returns instantly
        with latency_ms=0. On miss, delegates to the fallback backend
        (with full retry logic), saves the successful response, and returns it.
        """
        key = self._prompt_hash(prompt)

        # Cache hit
        cached_dict = self._cache.get(key) if self._cache is not None else None
        if cached_dict is not None:
            cached = self._llm_response_from_dict(cached_dict)
            cached.latency_ms = 0
            if self.enable_logging:
                self._log_request(prompt, cached, 0)
            return cached

        # Cache miss — delegate to fallback backend
        original_backend = self.backend
        self.backend = self._checkpoint_fallback
        try:
            response = self.query(prompt, timeout=timeout, max_retries=max_retries)
        finally:
            self.backend = original_backend

        # Save successful responses
        if response.success and self._cache is not None:
            self._cache[key] = self._llm_response_to_dict(response)

        return response

    def _query_codex(self, prompt: str, timeout: int) -> LLMResponse:
        """Query using Codex CLI."""
        try:
            result = subprocess.run(
                ["codex", "exec", "--skip-git-repo-check", "--json", prompt],
                capture_output=True, text=True, timeout=timeout,
                cwd=self._subprocess_cwd,
            )

            response = ""
            for line in result.stdout.strip().split('\n'):
                try:
                    event = json.loads(line)
                    if event.get('type') == 'item.completed':
                        item = event.get('item', {})
                        if item.get('type') == 'agent_message':
                            response = item.get('text', '')
                            break
                except json.JSONDecodeError:
                    continue

            if response:
                return LLMResponse(text=response, success=True)
            else:
                return LLMResponse(text="", success=False, error="No response from Codex")

        except subprocess.TimeoutExpired:
            return LLMResponse(text="", success=False, error="Codex request timed out")
        except Exception as e:
            return LLMResponse(text="", success=False, error=str(e))

    def _query_claude(self, prompt: str, timeout: int) -> LLMResponse:
        """Query using Claude Code CLI."""
        try:
            cmd = ["claude", "-p", "--output-format", "json", "--dangerously-skip-permissions"]
            if self.claude_model:
                cmd.extend(["--model", self.claude_model])
            cmd.append(prompt)
            # Strip CLAUDECODE env var so nested CLI calls work
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=timeout,
                cwd=self._subprocess_cwd,
                env=env,
            )

            response = ""

            # Try parsing as JSON result format
            try:
                data = json.loads(result.stdout.strip())
                if data.get('type') == 'result':
                    response = data.get('result', '')
            except json.JSONDecodeError:
                # Try stream-json format (multiple lines)
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'result':
                            response = event.get('result', '')
                            break
                        elif event.get('type') == 'assistant':
                            msg = event.get('message', {})
                            if isinstance(msg, dict):
                                content = msg.get('content', [])
                                for block in content:
                                    if isinstance(block, dict) and block.get('type') == 'text':
                                        response += block.get('text', '')
                            elif isinstance(msg, str):
                                response += msg
                    except json.JSONDecodeError:
                        continue

            # Fallback to raw output
            if not response and result.stdout.strip():
                response = result.stdout.strip()

            if response:
                return LLMResponse(text=response, success=True)
            else:
                error_msg = result.stderr.strip() if result.stderr else "No response from Claude"
                return LLMResponse(text="", success=False, error=error_msg)

        except subprocess.TimeoutExpired:
            return LLMResponse(text="", success=False, error="Claude request timed out")
        except FileNotFoundError:
            return LLMResponse(text="", success=False, error="Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        except Exception as e:
            return LLMResponse(text="", success=False, error=str(e))

    def _get_openai_client(self):
        """Lazily initialize OpenAI client with connection pool management."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                import httpx
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            # Limit connection pool to prevent fd exhaustion on high-volume runs
            http_client = httpx.Client(
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
            )
            self._openai_client = OpenAI(api_key=api_key, http_client=http_client)
        return self._openai_client

    def _query_openai(self, prompt: str, timeout: int, max_retries: int = 3) -> LLMResponse:
        """Query using OpenAI API (GPT-5.2) with retry logic and token tracking."""
        import time

        try:
            client = self._get_openai_client()
        except ImportError as e:
            return LLMResponse(text="", success=False, error=str(e))
        except ValueError as e:
            return LLMResponse(text="", success=False, error=str(e))

        last_error = None
        for attempt in range(max_retries):
            try:
                create_kwargs = dict(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that analyzes software architecture documents and extracts trace links between documentation and architecture models. Always respond with valid JSON when asked."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    seed=42,
                    max_completion_tokens=4096,
                    timeout=timeout,
                )
                create_kwargs["service_tier"] = os.environ.get("OPENAI_SERVICE_TIER", "flex")
                response = client.chat.completions.create(**create_kwargs)

                # Extract token usage
                token_usage = None
                if response.usage:
                    token_usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                        total_tokens=response.usage.total_tokens or 0
                    )

                if response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content
                    if text:
                        return LLMResponse(
                            text=text,
                            success=True,
                            token_usage=token_usage,
                            model=response.model
                        )

                return LLMResponse(text="", success=False, error="No response from OpenAI", token_usage=token_usage)

            except Exception as e:
                last_error = str(e)
                is_retryable = any(err in last_error.lower() for err in [
                    "timeout", "rate_limit", "rate limit", "connection",
                    "server_error", "503", "502", "504", "overloaded"
                ])

                if is_retryable and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 3, 5 seconds
                    print(f"    OpenAI retry {attempt + 1}/{max_retries} after {wait_time}s: {last_error[:50]}...")
                    time.sleep(wait_time)
                    continue
                break

        if last_error and "timeout" in last_error.lower():
            return LLMResponse(text="", success=False, error=f"OpenAI request timed out after {max_retries} retries")
        return LLMResponse(text="", success=False, error=f"OpenAI error after {max_retries} retries: {last_error}")

    def extract_json(self, response: LLMResponse) -> Optional[dict]:
        """Extract JSON from LLM response.

        Args:
            response: LLMResponse object

        Returns:
            Parsed JSON dict or None if extraction fails
        """
        if not response.success or not response.text:
            return None

        text = response.text.strip()

        # Fast path: entire response is valid JSON
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Find JSON by balanced-brace matching (handles stray { before real JSON)
        for i, ch in enumerate(text):
            if ch == '{':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[i:j+1])
                            except json.JSONDecodeError:
                                break  # This { wasn't the start, try next one
        return None


# Singleton instance for reuse
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the default LLM client instance.

    Returns a singleton instance based on LLM_BACKEND environment variable.
    """
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


# Convenience function for quick queries
def llm_query(prompt: str, backend: Optional[LLMBackend] = None, timeout: int = 180) -> LLMResponse:
    """Quick query function.

    Args:
        prompt: The prompt to send
        backend: Optional backend override
        timeout: Timeout in seconds

    Returns:
        LLMResponse
    """
    client = LLMClient(backend=backend)
    return client.query(prompt, timeout)


def llm_query_json(prompt: str, backend: Optional[LLMBackend] = None, timeout: int = 180) -> Optional[dict]:
    """Query and extract JSON response.

    Args:
        prompt: The prompt to send
        backend: Optional backend override
        timeout: Timeout in seconds

    Returns:
        Parsed JSON dict or None
    """
    client = LLMClient(backend=backend)
    response = client.query(prompt, timeout)
    return client.extract_json(response)


def print_usage_summary():
    """Print cumulative usage summary to console."""
    usage = LLMClient.get_cumulative_usage()
    print("\n" + "=" * 60)
    print("LLM USAGE SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {usage['total_requests']}")
    print(f"Total Errors:   {usage['total_errors']}")
    print(f"Prompt Tokens:  {usage['prompt_tokens']:,}")
    print(f"Completion Tokens: {usage['completion_tokens']:,}")
    print(f"Total Tokens:   {usage['total_tokens']:,}")

    # Estimate cost (approximate for GPT-4 class models)
    # GPT-4: ~$0.03/1K prompt, ~$0.06/1K completion
    if usage['total_tokens'] > 0:
        est_cost = (usage['prompt_tokens'] * 0.00003) + (usage['completion_tokens'] * 0.00006)
        print(f"Estimated Cost: ${est_cost:.4f} (GPT-4 pricing)")
    print("=" * 60)
