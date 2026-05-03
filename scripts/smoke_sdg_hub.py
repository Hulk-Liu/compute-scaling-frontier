"""Smoke test for sdg_hub.

Two tiers:
  Tier 1 (always run): import the library, list built-in flows.
  Tier 2 (run if --live): actually execute a tiny generation against gpt-4o-mini.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


SMOKE_FLOW_YAML = """
metadata:
  id: smoke-openai-ping
  name: OpenAI Ping Smoke Flow
  description: Minimal two-block sdg_hub flow used by scripts/smoke_sdg_hub.py.
  version: "0.1.0"
  author: Local Smoke Test
  license: Apache-2.0
  dataset_requirements:
    required_columns:
      - messages
    description: Input dataset contains OpenAI-style chat messages.
  output_columns:
    - extract_reply_content
blocks:
  - block_type: LLMChatBlock
    block_config:
      block_name: call_teacher
      input_cols: messages
      output_cols: raw_reply
      temperature: 0.0
      max_completion_tokens: 8
      async_mode: false
  - block_type: LLMResponseExtractorBlock
    block_config:
      block_name: extract_reply
      input_cols: raw_reply
      output_cols: extract_reply_content
      extract_content: true
      expand_lists: true
"""


def tier1_imports() -> None:
    print("[tier1] importing sdg_hub ...")
    import sdg_hub
    from sdg_hub import FlowRegistry  # noqa: F401

    print(f"[tier1] sdg_hub version: {getattr(sdg_hub, '__version__', '?')}")
    print("[tier1] discovering built-in flows ...")
    FlowRegistry.discover_flows()
    flows = FlowRegistry.list_flows() if hasattr(FlowRegistry, "list_flows") else []
    print(f"[tier1] found {len(flows)} built-in flow(s)")
    for name in flows[:10]:
        print(f"  - {name}")


def tier2_live_call() -> None:
    print("[tier2] live sdg_hub flow call against gpt-4o-mini ...")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[tier2] SKIP: OPENAI_API_KEY not set")
        return

    import pandas as pd
    from sdg_hub import Flow

    dataset = pd.DataFrame(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly one word: pong",
                    }
                ]
            }
        ]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        flow_path = Path(tmp_dir) / "smoke_flow.yaml"
        flow_path.write_text(SMOKE_FLOW_YAML, encoding="utf-8")

        flow = Flow.from_yaml(str(flow_path))
        flow.set_model_config(model="openai/gpt-4o-mini")
        result = flow.generate(dataset)

    reply = result["extract_reply_content"].iloc[0]
    print(f"[tier2] sdg_hub flow reply: {reply!r}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run tier 2 (calls OpenAI)")
    args = parser.parse_args()

    try:
        tier1_imports()
    except Exception as e:
        print(f"[FAIL] sdg_hub tier1: {type(e).__name__}: {e}")
        return 1

    if args.live:
        try:
            tier2_live_call()
        except Exception as e:
            print(f"[FAIL] sdg_hub tier2: {type(e).__name__}: {e}")
            return 1

    print("[OK] sdg_hub smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
