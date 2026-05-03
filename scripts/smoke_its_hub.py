"""Smoke test for its_hub.

Two tiers:
  Tier 1 (always run): import the library, list available algorithms.
  Tier 2 (run if --live): run Self-Consistency and Best-of-N on one question.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv


def tier1_imports() -> None:
    print("[tier1] importing its_hub ...")
    import its_hub

    print(f"[tier1] its_hub version: {getattr(its_hub, '__version__', '?')}")
    symbols = sorted(s for s in dir(its_hub) if not s.startswith("_"))[:20]
    print(f"[tier1] top-level symbols: {symbols}")

    try:
        from its_hub import (  # noqa: F401
            BestOfN,
            LLMJudge,
            OpenAICompatibleLanguageModel,
            SelfConsistency,
        )
    except ImportError as e:
        raise RuntimeError(
            "its_hub optional LM integration is unavailable. "
            "Install the package with the 'lm' extra: uv add 'its-hub[lm]>=1.0.0'."
        ) from e


async def tier2_live_call() -> None:
    print("[tier2] live SC + Best-of-N calls against gpt-4o-mini ...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[tier2] SKIP: OPENAI_API_KEY not set")
        return

    from its_hub import (
        BestOfN,
        LLMJudge,
        OpenAICompatibleLanguageModel,
        SelfConsistency,
    )

    lm = OpenAICompatibleLanguageModel(
        endpoint="https://api.openai.com/v1",
        api_key=api_key,
        model_name="gpt-4o-mini",
        max_tokens=24,
        temperature=0.0,
        max_concurrency=2,
    )
    judge_lm = OpenAICompatibleLanguageModel(
        endpoint="https://api.openai.com/v1",
        api_key=api_key,
        model_name="gpt-4o-mini",
        max_tokens=8,
        temperature=0.0,
        max_concurrency=2,
    )

    prompt = "What is 7 + 5? Reply with only the number."
    try:
        sc = SelfConsistency()
        sc_result = await sc.ainfer(lm, prompt, budget=2)
        print(f"[tier2] SelfConsistency selected: {sc_result.get('content')!r}")

        bon = BestOfN(LLMJudge(judge_lm))
        bon_result = await bon.ainfer(lm, prompt, budget=2)
        print(f"[tier2] BestOfN selected: {bon_result.get('content')!r}")
    finally:
        await lm.close()
        await judge_lm.close()


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run tier 2 (calls OpenAI)")
    args = parser.parse_args()

    try:
        tier1_imports()
    except Exception as e:
        print(f"[FAIL] its_hub tier1: {type(e).__name__}: {e}")
        return 1

    if args.live:
        try:
            asyncio.run(tier2_live_call())
        except Exception as e:
            print(f"[FAIL] its_hub tier2: {type(e).__name__}: {e}")
            return 1

    print("[OK] its_hub smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
