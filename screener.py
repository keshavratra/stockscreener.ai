#!/usr/bin/env python3
"""
Crude stock screener: tickers in -> latest 10-K / 10-Q from SEC EDGAR -> Claude
judges financial strength -> ranked picks with 2-3 bullets of reasoning.

Setup:
    pip install anthropic requests fpdf2
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
    Ranked summary printed to the terminal, full detail in results.json,
    and a readable report in results.pdf (change with --pdf, skip with --no-pdf)
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


# ---------------------------------------------------------------------- PDF

_PDF_SWAPS = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
              "\u2014": "-", "\u2212": "-", "\u2022": "-", "\u2026": "...", "\u00a0": " ",
              "\u2010": "-", "\u2011": "-",
              "\u2248": "~", "\u2265": ">=", "\u2264": "<=", "\u2192": "->", "\u00d7": "x"}
_PDF_COLORS = {"buy": (22, 130, 70), "watch": (190, 130, 0), "avoid": (190, 45, 45)}


def _pdf_text(text):
    """The PDF's built-in fonts only cover Latin-1; swap or drop anything else."""
    text = str(text)
    for bad, good in _PDF_SWAPS.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(results):
    """Render the screening results as a PDF report and return it as bytes."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("PDF output needs the fpdf2 package: pip install fpdf2")

    class Report(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", size=8)
            self.set_text_color(130)
            self.cell(0, 5, f"Page {self.page_no()}", align="C")

    pdf = Report(format="letter")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()
    width = pdf.epw

    def line(text, size=10, style="", color=(30, 30, 30), height=5, indent=0, link=""):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(width - indent, height, _pdf_text(text), link=link,
                       new_x="LMARGIN", new_y="NEXT")

    ok = sorted((r for r in results if "error" not in r), key=lambda r: -r["score"])
    errors = [r for r in results if "error" in r]

    line("Stock Screener Report", size=18, style="B", height=9)
    line(f"{date.today():%B %d, %Y}  |  {len(ok)} screened"
         + (f", {len(errors)} skipped" if errors else ""), size=9, color=(110, 110, 110))
    pdf.ln(4)

    if ok:  # summary table
        cols = [("Ticker", 0.12), ("Company", 0.44), ("Recommendation", 0.18),
                ("Strength", 0.14), ("Score", 0.12)]
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(235, 238, 242)
        pdf.set_text_color(30, 30, 30)
        for title, frac in cols:
            pdf.cell(width * frac, 7, title, border="B", fill=True)
        pdf.ln()
        for r in ok:
            cells = [r["ticker"], r["company"], r["recommendation"].capitalize(),
                     r["financial_strength"].capitalize(), f"{r['score']} / 10"]
            for i, ((_, frac), text) in enumerate(zip(cols, cells)):
                text = _pdf_text(text)
                pdf.set_font("Helvetica", "B" if i in (0, 2) else "", 9)
                while len(text) > 4 and pdf.get_string_width(text) > width * frac - 3:
                    text = text[:-4] + "..."   # trim long company names to fit
                pdf.set_text_color(*(_PDF_COLORS[r["recommendation"]] if i == 2
                                     else (30, 30, 30)))
                pdf.cell(width * frac, 6.5, text, border="B")
            pdf.ln()
        pdf.ln(5)

    for heading, rec in (("Buy", "buy"), ("Watch", "watch"), ("Avoid", "avoid")):
        group = [r for r in ok if r["recommendation"] == rec]
        if not group:
            continue
        if pdf.will_page_break(40):
            pdf.add_page()
        line(heading, size=13, style="B", color=_PDF_COLORS[rec], height=7)
        pdf.set_draw_color(*_PDF_COLORS[rec])
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
        pdf.ln(2)
        for r in group:
            if pdf.will_page_break(30):   # keep a company's heading with its bullets
                pdf.add_page()
            line(f"{r['ticker']}  -  {r['company']}", size=11, style="B", height=6)
            line(f"{r['financial_strength'].capitalize()}  |  {r['score']}/10",
                 size=9, color=(110, 110, 110))
            pdf.ln(1)
            for bullet in r["reasoning"]:
                top = pdf.get_y()
                line("-", indent=2)
                pdf.set_y(top)
                line(bullet, indent=6)
            if r["data_gaps"].strip().lower() != "none":
                line(f"Data gaps: {r['data_gaps']}", size=9, style="I",
                     color=(110, 110, 110), indent=6)
            for form in ("10-K", "10-Q"):
                if form in r["filings"]:
                    f = r["filings"][form]
                    line(f"{form}, filed {f['filed']}", size=9, style="U",
                         color=(30, 90, 170), indent=6, link=f["url"])
            pdf.ln(4)

    if errors:
        line("Skipped", size=13, style="B", height=7)
        for r in errors:
            line(f"{r['ticker']}: {r['error']}", size=9)
        pdf.ln(3)

    line('A "buy" here means the business is financially strong. No price or valuation '
         "data is used, and this is not financial advice.", size=8, style="I",
         color=(110, 110, 110), height=4)
    return bytes(pdf.output())


def save_pdf(results, path):
    with open(path, "wb") as fh:
        fh.write(build_pdf(results))


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
    parser.add_argument("--pdf", default="results.pdf", help="where to save the PDF report")
    parser.add_argument("--no-pdf", action="store_true", help="skip the PDF report")
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
    if not args.no_pdf:
        try:
            save_pdf(results, args.pdf)
            print(f"PDF report saved to {args.pdf}")
        except Exception as exc:  # e.g. fpdf2 missing, or the PDF is open in a viewer
            print(f"Could not write the PDF: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()