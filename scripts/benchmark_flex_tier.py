"""
Benchmark: standard vs flex service_tier latency on OpenAI API.

Runs N calls on each tier using a realistic linker-sized prompt,
reports per-call latency stats and whether flex tier is even supported
for the configured model.

Usage:
    python scripts/benchmark_flex_tier.py [--n 10] [--model gpt-5.4]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Load .env manually (no dotenv dependency required)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _openai_call(client, model: str, prompt: str, service_tier: str | None, timeout: int = 120):
    """Single synchronous chat completion call. Returns (latency_ms, response_text, error, actual_tier)."""
    create_kwargs = dict(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze software architecture documentation and identify "
                    "trace links between text and model elements. Respond with JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        seed=42,
        max_completion_tokens=512,
        timeout=timeout,
    )
    if service_tier is not None:
        create_kwargs["service_tier"] = service_tier

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**create_kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = resp.choices[0].message.content if resp.choices else ""
        actual_tier = getattr(resp, "service_tier", None)
        return latency_ms, text, None, actual_tier
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return latency_ms, "", str(e), None


def run_tier(client, model: str, prompt: str, service_tier: str | None, n: int, label: str):
    """Run N calls and return latency list + any errors."""
    print(f"\n{'='*60}")
    print(f"Tier: {label}  (service_tier={service_tier!r})  n={n}")
    print(f"{'='*60}")

    latencies = []
    errors = []
    actual_tiers = []

    for i in range(1, n + 1):
        lat, text, err, actual = _openai_call(client, model, prompt, service_tier)
        if err:
            errors.append(err)
            print(f"  call {i:2d}: ERROR  {err[:80]}")
        else:
            latencies.append(lat)
            actual_tiers.append(actual)
            print(f"  call {i:2d}: {lat:6d} ms  actual_tier={actual!r}  resp_len={len(text)}")

    return latencies, errors, actual_tiers


def print_stats(label: str, latencies: list[int]):
    if not latencies:
        print(f"\n{label}: no successful calls")
        return
    s = sorted(latencies)
    n = len(s)
    print(f"\n{label} latency stats (n={n}):")
    print(f"  min    = {min(s):7.0f} ms")
    print(f"  median = {statistics.median(s):7.0f} ms")
    print(f"  mean   = {statistics.mean(s):7.0f} ms")
    print(f"  p75    = {s[int(n*0.75)]:7.0f} ms")
    print(f"  p90    = {s[int(n*0.90)]:7.0f} ms")
    print(f"  max    = {max(s):7.0f} ms")
    print(f"  stdev  = {statistics.stdev(s) if n > 1 else 0:7.0f} ms")


REPRESENTATIVE_PROMPT = """Analyze this architecture documentation excerpt and identify which
model component names are mentioned. Return JSON with key "components": [list of component names].

Documentation excerpt:
The system uses a dedicated Database Service to persist all user records and session data.
The API Gateway routes incoming requests to downstream microservices including the
Authentication Service and the Notification Service. A Message Queue decouples the
Notification Service from the core request path. The Cache Layer sits in front of the
Database Service to reduce read latency. All services communicate over an internal
Service Mesh.

Model elements (possible matches):
- DatabaseService
- APIGateway
- AuthenticationService
- NotificationService
- MessageQueue
- CacheLayer
- ServiceMesh
- UserManager
- SessionStore
- LoadBalancer

Return only JSON."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="calls per tier (default 10)")
    parser.add_argument("--model", default=None, help="model override (default: OPENAI_MODEL_NAME env)")
    parser.add_argument("--output", default="results/flex_tier_benchmark.json", help="JSON output path")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    model = args.model or os.environ.get("OPENAI_MODEL_NAME", "gpt-5.4")

    try:
        from openai import OpenAI
        import httpx
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    http_client = httpx.Client(limits=httpx.Limits(max_connections=5, max_keepalive_connections=2))
    client = OpenAI(api_key=api_key, http_client=http_client)

    print(f"Model: {model}")
    print(f"N per tier: {args.n}")
    print(f"Prompt length: {len(REPRESENTATIVE_PROMPT)} chars")

    # Run standard tier first
    std_latencies, std_errors, std_tiers = run_tier(
        client, model, REPRESENTATIVE_PROMPT,
        service_tier=None, n=args.n, label="STANDARD (no service_tier)"
    )

    # Run flex tier
    flex_latencies, flex_errors, flex_tiers = run_tier(
        client, model, REPRESENTATIVE_PROMPT,
        service_tier="flex", n=args.n, label="FLEX (service_tier=flex)"
    )

    # Print comparison
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print_stats("Standard", std_latencies)
    print_stats("Flex    ", flex_latencies)

    if std_latencies and flex_latencies:
        ratio = statistics.median(flex_latencies) / statistics.median(std_latencies)
        print(f"\nFlex/Standard median ratio: {ratio:.2f}x")

    if flex_errors:
        unique_errors = list(set(flex_errors))
        print(f"\nFlex tier errors ({len(flex_errors)} calls):")
        for e in unique_errors:
            print(f"  {e[:120]}")

    # Save JSON results
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model,
        "n_per_tier": args.n,
        "prompt_length": len(REPRESENTATIVE_PROMPT),
        "standard": {
            "latencies_ms": std_latencies,
            "errors": std_errors,
            "actual_tiers": [t for t in std_tiers if t],
            "n_success": len(std_latencies),
            "median_ms": statistics.median(std_latencies) if std_latencies else None,
            "mean_ms": statistics.mean(std_latencies) if std_latencies else None,
        },
        "flex": {
            "latencies_ms": flex_latencies,
            "errors": flex_errors,
            "actual_tiers": [t for t in flex_tiers if t],
            "n_success": len(flex_latencies),
            "median_ms": statistics.median(flex_latencies) if flex_latencies else None,
            "mean_ms": statistics.mean(flex_latencies) if flex_latencies else None,
        },
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"\nJSON saved to: {out}")


if __name__ == "__main__":
    main()
