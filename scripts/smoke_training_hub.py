"""Smoke test for training_hub.

We expect this one to surface the Apple Silicon limitation: training_hub.lora_sft
is backed by Unsloth, which currently does not support training on macOS / MPS
(MLX backend "coming soon" per Unsloth README). The plan accommodates this by
running training in Colab Pro; this smoke test just confirms the failure mode
locally so we can document it.
"""

from __future__ import annotations

import sys


def tier1_imports() -> None:
    print("[tier1] importing training_hub ...")
    import training_hub

    print(f"[tier1] training_hub version: {getattr(training_hub, '__version__', '?')}")
    print(f"[tier1] top-level symbols: {sorted(s for s in dir(training_hub) if not s.startswith('_'))[:20]}")


def tier1b_lora_backend_check() -> None:
    """Try to import the Unsloth backend that training_hub.lora_sft uses.

    A clean import here suggests Mac is supported (would be a surprise — Unsloth
    upstream has Triton/CUDA-only kernels). An ImportError or platform error is
    the expected outcome and is what we want to capture for the README write-up.
    """
    print("[tier1b] checking Unsloth backend availability ...")
    try:
        import unsloth  # noqa: F401

        print("[tier1b] UNEXPECTED: unsloth imported cleanly on this platform")
    except Exception as e:
        print(f"[tier1b] EXPECTED: unsloth unavailable — {type(e).__name__}: {str(e)[:200]}")
        print("[tier1b] -> training_hub.lora_sft cannot run locally on Mac.")
        print("[tier1b] -> Plan: train in Colab Pro instead. Will document in README 'What Didn't Work'.")


def main() -> int:
    try:
        tier1_imports()
    except Exception as e:
        print(f"[FAIL] training_hub tier1: {type(e).__name__}: {e}")
        return 1

    # tier1b is informational — never fails the smoke test, the failure is the finding.
    tier1b_lora_backend_check()

    print("[OK] training_hub smoke test completed (lora backend status logged above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
