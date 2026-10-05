#!/usr/bin/env python3
"""
Crude stock screener: tickers in -> latest 10-K / 10-Q from SEC EDGAR -> Claude
judges financial strength -> ranked picks with 2-3 bullets of reasoning.

Setup:
    pip install anthropic requests
    Windows PowerShell:
        $env:ANTHROPIC_API_KEY = "sk-ant-..."
        $env:SEC_USER_AGENT = "Your Name you@example.com"   # SEC requires a contact
    Mac / Linux:
        export ANTHROPIC_API_KEY=sk-ant-...
        export SEC_USER_AGENT="Your Name you@example.com"

Usage:
    python screener.py AAPL MSFT F
    python screener.py --file tickers.txt        # one ticker per line
Output:
    Ranked summary printed to the terminal, full detail in results.json
"""
import argparse
import json
import os
import sys
import time
from datetime import date

import requests

MODEL = "claude-sonnet-5-5"
N_ANNUAL = 3  # fiscal years of history to show the model

# Label -> candidate XBRL tags (companies differ; the freshest one wins).
CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "stockholders_equity": ["StockholdersEquity"],
}

ASSESSMENT_TOOL = {
    "name": "record_assessment",
    "description": "Record the financial-strength assessment for one company.",
    "input_schema": {
        "type": "object",
        "properties": {
            "financial_strength": {"type": "string", "enum": ["strong", "adequate", "weak"]},
            "score": {"type": "integer", "description": "1 (distressed) to 10 (fortress)"},
            "recommendation": {"type": "string", "enum": ["buy", "watch", "avoid"]},
            "reasoning": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 short bullets, each citing specific figures from the data.",
            },
            "data_gaps": {"type": "string",
                          "description": "Anything missing that limits confidence, or 'none'."},
        },
        "required": ["financial_strength", "score", "recommendation", "reasoning", "data_gaps"],
    },
}

SYSTEM_PROMPT = """You are a fundamentals analyst screening companies for financial strength.
You are given figures taken from a company's latest 10-K (annual) and 10-Q (quarterly) filings.

Judge strength on: revenue and earnings trend, margins, cash generation
(operating cash flow minus capex), liquidity (current ratio, cash), and
leverage (debt vs equity, interest coverage).

Rules:
- Use only the numbers provided. Never invent or recall figures.
- Every reasoning bullet must cite at least one specific number.
- You have no price or valuation data. "buy" means the business is financially
  strong and improving, "watch" means mixed, "avoid" means weak or deteriorating.
- If key data is missing, say so in data_gaps and be more conservative."""


# ---------------------------------------------------------------- SEC EDGAR

def sec_get(url):
    resp = requests.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=30)
    resp.raise_for_status()
    time.sleep(0.15)  # stay well under SEC's 10 requests/second limit
    return resp.json()


def load_cik_map():
    data = sec_get("https://www.sec.gov/files/company_tickers.json")
    return {row["ticker"].upper(): (row["cik_str"], row["title"]) for row in data.values()}


def latest_filings(cik):
    """Most recent 10-K and 10-Q: accession number, dates, and a link."""
    recent = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")["filings"]["recent"]
    found = {}
    for i, form in enumerate(recent["form"]):  # newest first
        if form in ("10-K", "10-Q") and form not in found:
            accn = recent["accessionNumber"][i]
            found[form] = {
                "accession": accn,
                "filed": recent["filingDate"][i],
                "period_end": recent["reportDate"][i],
                "url": (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                        f"{accn.replace('-', '')}/{recent['primaryDocument'][i]}"),
            }
        if len(found) == 2:
            break
    return found


def _days(entry):
    if "start" not in entry:
        return None  # balance-sheet item (point in time)
    return (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days


def _annual_series(entries):
    """Full-year values from 10-Ks, one per fiscal year end, newest last."""
    by_end = {}
    for e in entries:
        if e.get("form") != "10-K":
            continue
        d = _days(e)
        if d is not None and not 340 <= d <= 380:
            continue
        if e["end"] not in by_end or e["filed"] > by_end[e["end"]]["filed"]:
            by_end[e["end"]] = e
    return [{"period_end": k, "value": by_end[k]["val"]} for k in sorted(by_end)][-N_ANNUAL:]


def extract_financials(companyfacts, q_accession):
    """Pull the CONCEPTS out of SEC's XBRL 'companyfacts' blob."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    out = {}
    for label, tags in CONCEPTS.items():
        best = None
        for tag in tags:
            if tag not in gaap:
                continue
            entries = next(iter(gaap[tag]["units"].values()))
            annual = _annual_series(entries)
            quarterly = [
                {"period_end": e["end"], "days_covered": _days(e), "value": e["val"]}
                for e in entries if e.get("accn") == q_accession
            ]
            if not annual and not quarterly:
                continue
            freshness = max([a["period_end"] for a in annual] +
                            [q["period_end"] for q in quarterly])
            if best is None or freshness > best[0]:
                best = (freshness, {"annual_10K": annual, "latest_10Q": quarterly})
        if best:
            out[label] = best[1]
    return out


# ------------------------------------------------------------------- Claude

def assess(client, ticker, name, filings, financials):
    payload = {"ticker": ticker, "company": name, "filings": filings,
               "financials_usd": financials}
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[ASSESSMENT_TOOL],
        messages=[{
            "role": "user",
            "content": ("Assess this company and report the result by calling the "
                        "record_assessment tool exactly once. In latest_10Q, days_covered of ~90 is the "
                        "quarter alone, larger is year-to-date, null is a balance-sheet "
                        "value; earlier period_end dates are prior-year comparisons.\n\n"
                        + json.dumps(payload, indent=1)),
        }],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("model replied without a structured assessment; try again")


# --------------------------------------------------------------------- main

def screen(tickers):
    import anthropic
    client = anthropic.Anthropic()
    cik_map = load_cik_map()
    results = []
    for ticker in tickers:
        print(f"... {ticker}", file=sys.stderr)
        row = {"ticker": ticker}
        try:
            if ticker not in cik_map:
                raise ValueError("ticker not found on SEC EDGAR")
            cik, name = cik_map[ticker]
            filings = latest_filings(cik)
            if "10-K" not in filings:
                raise ValueError("no 10-K on file (foreign filers use 20-F / 40-F)")
            facts = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
            financials = extract_financials(facts, filings.get("10-Q", {}).get("accession"))
            row.update(company=name, filings=filings,
                       **assess(client, ticker, name, filings, financials))
        except Exception as exc:  # one bad ticker shouldn't sink the run
            row["error"] = str(exc)
        results.append(row)
    return results


def print_report(results):
    ok = sorted((r for r in results if "error" not in r), key=lambda r: -r["score"])
    for heading, rec in (("BUY", "buy"), ("WATCH", "watch"), ("AVOID", "avoid")):
        group = [r for r in ok if r["recommendation"] == rec]
        if not group:
            continue
        print(f"\n=== {heading} ===")
        for r in group:
            print(f"\n{r['ticker']}  {r['company']}  "
                  f"[{r['financial_strength']}, {r['score']}/10]")
            for bullet in r["reasoning"]:
                print(f"  - {bullet}")
            if r["data_gaps"].strip().lower() != "none":
                print(f"  (gaps: {r['data_gaps']})")
            print(f"  10-K: {r['filings']['10-K']['url']}")
            if "10-Q" in r["filings"]:
                print(f"  10-Q: {r['filings']['10-Q']['url']}")
    errors = [r for r in results if "error" in r]
    if errors:
        print("\n=== SKIPPED ===")
        for r in errors:
            print(f"{r['ticker']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="Screen stocks on 10-K / 10-Q fundamentals.")
    parser.add_argument("tickers", nargs="*", help="e.g. AAPL MSFT F")
    parser.add_argument("--file", help="text file with one ticker per line")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.file:
        with open(args.file) as fh:
            tickers += [line.strip() for line in fh if line.strip()]
    tickers = list(dict.fromkeys(t.upper() for t in tickers))  # de-dupe, keep order
    if not tickers:
        parser.error("give at least one ticker or --file")
    for var in ("ANTHROPIC_API_KEY", "SEC_USER_AGENT"):
        if not os.environ.get(var):
            sys.exit(f"Set the {var} environment variable first (see top of this file).")

    results = screen(tickers)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print_report(results)
    print(f"\nFull detail saved to {args.out}")


if __name__ == "__main__":
    main()