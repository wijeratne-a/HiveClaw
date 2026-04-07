#!/usr/bin/env python3
"""
Estimate monthly / annual savings running HiveClaw locally vs a cloud chat model.

Example:
  python tools/roi_calculator.py \\
    --prompt-tokens-per-day 500000 \\
    --completion-tokens-per-day 200000 \\
    --price-per-1m-input 2.50 \\
    --price-per-1m-output 10.00
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="Monthly cloud cost vs ~$0 local inference (HiveClaw)."
    )
    p.add_argument(
        "--prompt-tokens-per-day",
        type=float,
        default=0.0,
        help="Average prompt (input) tokens per day",
    )
    p.add_argument(
        "--completion-tokens-per-day",
        type=float,
        default=0.0,
        help="Average completion (output) tokens per day",
    )
    p.add_argument(
        "--price-per-1m-input",
        type=float,
        default=5.0,
        help="USD per 1M input tokens (e.g. GPT-4o list price)",
    )
    p.add_argument(
        "--price-per-1m-output",
        type=float,
        default=15.0,
        help="USD per 1M output tokens",
    )
    p.add_argument(
        "--days-per-month",
        type=float,
        default=30.0,
        help="Billing month length (default 30)",
    )
    args = p.parse_args()

    pin = max(0.0, float(args.prompt_tokens_per_day))
    cout = max(0.0, float(args.completion_tokens_per_day))
    days = max(1.0, float(args.days_per_month))

    monthly_in = pin * days
    monthly_out = cout * days

    cost_in = (monthly_in / 1_000_000.0) * float(args.price_per_1m_input)
    cost_out = (monthly_out / 1_000_000.0) * float(args.price_per_1m_output)
    monthly_cloud = cost_in + cost_out
    annual_cloud = monthly_cloud * 12.0

    print("HiveClaw ROI (illustrative; local power/hardware not included)")
    print("-" * 56)
    print(f"  Prompt tokens / day:        {pin:,.0f}")
    print(f"  Completion tokens / day:    {cout:,.0f}")
    print(f"  Days / month:               {days:,.0f}")
    print(f"  Cloud $/1M input:           ${args.price_per_1m_input:.4f}")
    print(f"  Cloud $/1M output:          ${args.price_per_1m_output:.4f}")
    print("-" * 56)
    print(f"  Estimated monthly (cloud):  ${monthly_cloud:,.2f}")
    print(f"  Estimated annual (cloud):   ${annual_cloud:,.2f}")
    print(f"  Local (HiveClaw) tokens:    ~$0 API spend")
    print(f"  Monthly savings vs cloud:   ${monthly_cloud:,.2f}")
    print(f"  Annualized savings:         ${annual_cloud:,.2f}")


if __name__ == "__main__":
    main()
