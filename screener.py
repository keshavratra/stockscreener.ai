#!/usr/bin/env python3
"""
Crude stock screener: tickers in -> latest 10-K / 10-Q from SEC EDGAR -> Claude
judges financial strength -> ranked picks with 2-3 bullets of reasoning.
Give it an ETF and it screens the companies the fund holds instead, and
reports what percent of the fund each one makes up.

Setup:
    pip install anthropic requests fpdf2
    Windows PowerShell:
        $env:ANTHROPIC_API_KEY = "sk-ant-..."
        $env:SEC_USER_AGENT = "Your Name you@example.com"   # SEC requires a contact
        $env:ALPHAVANTAGE_API_KEY = "..."                   # only needed for ETFs
    Mac / Linux:
        export ANTHROPIC_API_KEY=sk-ant-...
        export SEC_USER_AGENT="Your Name you@example.com"
        export ALPHAVANTAGE_API_KEY=...                     # only needed for ETFs
    The Alpha Vantage key supplies ETF holdings; get one free at
    https://www.alphavantage.co/support/#api-key

Usage:
    python screener.py AAPL MSFT F
    python screener.py SPY                       # top 25 holdings of the ETF
    python screener.py SPY --etf-top 100         # top 100 (0 = every holding)
    python screener.py --file tickers.txt        # one ticker per line
Output:
    Ranked summary printed to the terminal, full detail in results.json,
    and a readable report in results.pdf (change with --pdf, skip with --no-pdf)
"""
import argparse
import json
import os
import re
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


# --------------------------------------------------------------------- ETFs

ETF_TOP_DEFAULT = 25  # holdings screened per ETF, biggest first (0 = all of them)
TICKER_OK = re.compile(r"^[A-Z][A-Z0-9]{0,5}(-[A-Z0-9]{1,2})?$")


def etf_holdings(ticker, api_key):
    """Every holding of an ETF, biggest first: [{"ticker", "name", "weight_pct"}].

    Returns [] when the symbol is not an ETF. Uses Alpha Vantage's ETF_PROFILE
    endpoint (one request per ETF). To use another data source, replace this
    function and keep the return shape.
    """
    resp = requests.get("https://www.alphavantage.co/query", timeout=30,
                        params={"function": "ETF_PROFILE", "symbol": ticker,
                                "apikey": api_key})
    resp.raise_for_status()
    data = resp.json()
    for key in ("Information", "Note", "Error Message"):  # rate limit, bad key, bad symbol
        if key in data:
            raise ValueError(f"Alpha Vantage says: {data[key]}")
    rows = []
    for h in data.get("holdings") or []:
        try:
            weight = float(h.get("weight"))
        except (TypeError, ValueError):
            continue
        symbol = str(h.get("symbol") or "").strip().upper().replace(".", "-")
        rows.append({"ticker": symbol, "name": str(h.get("description") or "").strip(),
                     "weight_pct": weight})
    if rows and sum(r["weight_pct"] for r in rows) <= 1.5:  # sent as fractions (0.07 = 7%)
        for r in rows:
            r["weight_pct"] *= 100
    for r in rows:
        r["weight_pct"] = round(r["weight_pct"], 4)
    return sorted(rows, key=lambda r: -r["weight_pct"])


def expand(tickers, cik_map, av_key="", etf_top=ETF_TOP_DEFAULT):
    """Turn what the user typed into the stocks to screen, opening up any ETFs.

    Returns (plan, etfs, errors):
      plan   - {ticker: {"filings": dict or None, "etfs": {etf: weight_pct}}}
      etfs   - one summary per ETF, with the holdings chosen for screening
      errors - rows for symbols that are neither a screenable stock nor an ETF
    """
    plan, etfs, errors = {}, [], []
    for ticker in tickers:
        try:
            filings = latest_filings(cik_map[ticker][0]) if ticker in cik_map else {}
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            continue
        if "10-K" in filings:  # an ordinary company
            plan.setdefault(ticker, {"filings": None, "etfs": {}})["filings"] = filings
            continue
        problem = ("no 10-K on file (foreign filers use 20-F / 40-F)" if ticker in cik_map
                   else "ticker not found on SEC EDGAR")
        if not av_key:
            errors.append({"ticker": ticker, "error":
                           f"{problem}. If this is an ETF, add an Alpha Vantage key to "
                           "screen its holdings."})
            continue
        try:
            holdings = etf_holdings(ticker, av_key)
        except Exception as exc:
            errors.append({"ticker": ticker,
                           "error": f"{problem}; ETF holdings lookup failed: {exc}"})
            continue
        usable = [h for h in holdings if TICKER_OK.match(h["ticker"])]  # drops cash, futures
        if not usable:
            errors.append({"ticker": ticker,
                           "error": f"{problem}, and no ETF holdings were found for it"})
            continue
        chosen = usable[:etf_top] if etf_top else usable
        etfs.append({"ticker": ticker,
                     "name": cik_map[ticker][1] if ticker in cik_map else "",
                     "holdings_total": len(usable),  # stocks only; cash etc. left out
                     "weight_screened_pct": round(sum(h["weight_pct"] for h in chosen), 2),
                     "holdings": chosen})
        for h in chosen:
            plan.setdefault(h["ticker"], {"filings": None, "etfs": {}})["etfs"][ticker] = \
                h["weight_pct"]
    return plan, etfs, errors


def etf_scope(etf):
    """One sentence saying how much of the ETF was screened."""
    n = len(etf["holdings"])
    part = "all" if n >= etf["holdings_total"] else f"the top {n} of"
    return (f"Screened {part} {etf['holdings_total']} stock holdings, "
            f"{etf['weight_screened_pct']:.1f}% of the fund by weight.")


def etf_breakdown(etf, results):
    """Join an ETF's holdings to their assessments.

    Returns {"rows": [...], "weights": {"buy", "watch", "avoid", "skipped"},
    "weighted_score": float or None}. Weights are percent of the whole fund.
    """
    by_ticker = {r["ticker"]: r for r in results}
    weights = {"buy": 0.0, "watch": 0.0, "avoid": 0.0, "skipped": 0.0}
    rows, score_sum = [], 0.0
    for h in etf["holdings"]:
        r = by_ticker.get(h["ticker"], {"error": "not screened"})
        rated = "error" not in r
        weights[r["recommendation"] if rated else "skipped"] += h["weight_pct"]
        if rated:
            score_sum += h["weight_pct"] * r["score"]
        rows.append({"ticker": h["ticker"], "weight_pct": h["weight_pct"],
                     "company": r.get("company") or h["name"],
                     "recommendation": r["recommendation"] if rated else None,
                     "financial_strength": r["financial_strength"] if rated else None,
                     "score": r["score"] if rated else None})
    rated_weight = weights["buy"] + weights["watch"] + weights["avoid"]
    return {"rows": rows, "weights": weights,
            "weighted_score": score_sum / rated_weight if rated_weight else None}


def etf_tag(row):
    """'SPY 7.12%, QQQ 8.90%' for a stock that was screened as part of an ETF."""
    return ", ".join(f"{etf} {w:.2f}%" for etf, w in row.get("etfs", {}).items())


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


def build_pdf(results, etfs=()):
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
    dark, grey = (30, 30, 30), (110, 110, 110)

    def line(text, size=10, style="", color=dark, height=5, indent=0, link=""):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(width - indent, height, _pdf_text(text), link=link,
                       new_x="LMARGIN", new_y="NEXT")

    def table(cols, rows):
        """cols: [(title, fraction of page width)]; rows: lists of (text, style, color)."""
        def header():
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(235, 238, 242)
            pdf.set_text_color(*dark)
            for title, frac in cols:
                pdf.cell(width * frac, 7, title, border="B", fill=True)
            pdf.ln()
        header()
        for cells in rows:
            if pdf.will_page_break(6.5):
                pdf.add_page()
                header()
            for (_, frac), (text, style, color) in zip(cols, cells):
                text = _pdf_text(text)
                pdf.set_font("Helvetica", style, 9)
                while len(text) > 4 and pdf.get_string_width(text) > width * frac - 3:
                    text = text[:-4] + "..."   # trim long company names to fit
                pdf.set_text_color(*color)
                pdf.cell(width * frac, 6.5, text, border="B")
            pdf.ln()
        pdf.ln(5)

    def rec_cell(rec):
        if rec is None:
            return ("Skipped", "", grey)
        return (rec.capitalize(), "B", _PDF_COLORS[rec])

    ok = sorted((r for r in results if "error" not in r), key=lambda r: -r["score"])
    errors = [r for r in results if "error" in r]

    line("Stock Screener Report", size=18, style="B", height=9)
    line(f"{date.today():%B %d, %Y}  |  {len(ok)} screened"
         + (f", {len(errors)} skipped" if errors else ""), size=9, color=grey)
    pdf.ln(4)

    for etf in etfs:  # one holdings table per ETF, biggest holding first
        info = etf_breakdown(etf, results)
        if pdf.will_page_break(45):
            pdf.add_page()
        line(f"ETF: {etf['ticker']}" + (f"  -  {etf['name']}" if etf["name"] else ""),
             size=13, style="B", height=7)
        line(etf_scope(etf), size=9, color=grey)
        w = info["weights"]
        mix = (f"Share of the fund rated  Buy {w['buy']:.1f}%  |  Watch {w['watch']:.1f}%"
               f"  |  Avoid {w['avoid']:.1f}%")
        if w["skipped"]:
            mix += f"  |  Skipped {w['skipped']:.1f}%"
        line(mix, size=9)
        if info["weighted_score"] is not None:
            line(f"Weight-averaged score of the rated holdings: "
                 f"{info['weighted_score']:.1f} / 10", size=9)
        pdf.ln(2)
        table([("Weight", 0.11), ("Ticker", 0.11), ("Company", 0.38),
               ("Recommendation", 0.17), ("Strength", 0.13), ("Score", 0.10)],
              [[(f"{h['weight_pct']:.2f}%", "", dark), (h["ticker"], "B", dark),
                (h["company"], "", dark), rec_cell(h["recommendation"]),
                ((h["financial_strength"] or "-").capitalize(), "", dark),
                (f"{h['score']} / 10" if h["score"] is not None else "-", "", dark)]
               for h in info["rows"]])

    listed = [r for r in ok if not r.get("etfs")] if etfs else ok
    if listed:  # summary table (stocks entered directly, when ETFs are also present)
        if etfs:
            line("Individual stocks", size=13, style="B", height=7)
            pdf.ln(1)
        table([("Ticker", 0.12), ("Company", 0.44), ("Recommendation", 0.18),
               ("Strength", 0.14), ("Score", 0.12)],
              [[(r["ticker"], "B", dark), (r["company"], "", dark),
                rec_cell(r["recommendation"]),
                (r["financial_strength"].capitalize(), "", dark),
                (f"{r['score']} / 10", "", dark)] for r in listed])

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
            sub = f"{r['financial_strength'].capitalize()}  |  {r['score']}/10"
            if r.get("etfs"):
                sub += f"  |  {etf_tag(r)}"
            line(sub, size=9, color=grey)
            pdf.ln(1)
            for bullet in r["reasoning"]:
                top, page = pdf.get_y(), pdf.page
                line("-", indent=2)
                if pdf.page == page:
                    pdf.set_y(top)
                line(bullet, indent=6)
            if r["data_gaps"].strip().lower() != "none":
                line(f"Data gaps: {r['data_gaps']}", size=9, style="I", color=grey, indent=6)
            for form in ("10-K", "10-Q"):
                if form in r["filings"]:
                    f = r["filings"][form]
                    line(f"{form}, filed {f['filed']}", size=9, style="U",
                         color=(30, 90, 170), indent=6, link=f["url"])
            pdf.ln(4)

    if errors:
        if pdf.will_page_break(20):
            pdf.add_page()
        line("Skipped", size=13, style="B", height=7)
        for r in errors:
            tag = f" ({etf_tag(r)})" if r.get("etfs") else ""
            line(f"{r['ticker']}{tag}: {r['error']}", size=9)
        pdf.ln(3)

    line('A "buy" here means the business is financially strong. No price or valuation '
         "data is used, and this is not financial advice.", size=8, style="I",
         color=grey, height=4)
    return bytes(pdf.output())


def save_pdf(results, path, etfs=()):
    with open(path, "wb") as fh:
        fh.write(build_pdf(results, etfs))


# --------------------------------------------------------------------- main

def screen_one(client, cik_map, ticker, filings=None):
    """Screen one company. Failures come back as row['error'] instead of raising."""
    row = {"ticker": ticker}
    try:
        if ticker not in cik_map:
            raise ValueError("ticker not found on SEC EDGAR")
        cik, name = cik_map[ticker]
        filings = filings or latest_filings(cik)
        if "10-K" not in filings:
            raise ValueError("no 10-K on file (foreign filers use 20-F / 40-F)")
        facts = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        financials = extract_financials(facts, filings.get("10-Q", {}).get("accession"))
        row.update(company=name, filings=filings,
                   **assess(client, ticker, name, filings, financials))
    except Exception as exc:  # one bad ticker shouldn't sink the run
        row["error"] = str(exc)
    return row


def screen(tickers, client=None, cik_map=None, av_key="", etf_top=ETF_TOP_DEFAULT,
           progress=None):
    """Screen stocks and ETFs. Returns (results, etfs).

    An ETF is replaced by its holdings; each of those rows carries
    row["etfs"] = {etf ticker: percent of that fund}. A stock that shows up
    more than once (in two ETFs, or typed in as well) is only screened once.
    progress(i, n, ticker) is called before each company, if given.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    if cik_map is None:
        cik_map = load_cik_map()
    plan, etfs, errors = expand(tickers, cik_map, av_key, etf_top)
    results = [e for e in errors if e["ticker"] not in plan]  # else it is listed twice
    for i, (ticker, info) in enumerate(plan.items()):
        if progress:
            progress(i, len(plan), ticker)
        row = screen_one(client, cik_map, ticker, info["filings"])
        if info["etfs"]:
            row["etfs"] = info["etfs"]
        results.append(row)
    return results, etfs


def to_json(results, etfs=()):
    """results.json content: a plain list, or {"etfs", "results"} when ETFs were screened."""
    return json.dumps({"etfs": list(etfs), "results": results} if etfs else results, indent=2)


def print_report(results, etfs=()):
    ok = sorted((r for r in results if "error" not in r), key=lambda r: -r["score"])
    for etf in etfs:
        info = etf_breakdown(etf, results)
        w = info["weights"]
        print(f"\n=== ETF: {etf['ticker']}  {etf['name']} ===")
        print(etf_scope(etf))
        print(f"Share of the fund rated  buy {w['buy']:.1f}%  |  watch {w['watch']:.1f}%  |  "
              f"avoid {w['avoid']:.1f}%  |  skipped {w['skipped']:.1f}%")
        if info["weighted_score"] is not None:
            print(f"Weight-averaged score of the rated holdings: "
                  f"{info['weighted_score']:.1f}/10")
        for h in info["rows"]:
            verdict = (f"{h['recommendation']:<6}{h['score']:>2}/10" if h["recommendation"]
                       else "skipped")
            print(f"  {h['weight_pct']:6.2f}%  {h['ticker']:<7} {verdict:<12} {h['company']}")
    for heading, rec in (("BUY", "buy"), ("WATCH", "watch"), ("AVOID", "avoid")):
        group = [r for r in ok if r["recommendation"] == rec]
        if not group:
            continue
        print(f"\n=== {heading} ===")
        for r in group:
            print(f"\n{r['ticker']}  {r['company']}  "
                  f"[{r['financial_strength']}, {r['score']}/10]"
                  + (f"  ({etf_tag(r)})" if r.get("etfs") else ""))
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
            tag = f" ({etf_tag(r)})" if r.get("etfs") else ""
            print(f"{r['ticker']}{tag}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Screen stocks, and the holdings of ETFs, on 10-K / 10-Q fundamentals.")
    parser.add_argument("tickers", nargs="*", help="stocks or ETFs, e.g. AAPL MSFT SPY")
    parser.add_argument("--file", help="text file with one ticker per line")
    parser.add_argument("--etf-top", type=int, default=ETF_TOP_DEFAULT, metavar="N",
                        help="for an ETF, screen its N biggest holdings "
                             f"(default {ETF_TOP_DEFAULT}; 0 = every holding)")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--pdf", default="results.pdf", help="where to save the PDF report")
    parser.add_argument("--no-pdf", action="store_true", help="skip the PDF report")
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.file:
        with open(args.file) as fh:
            tickers += [line.strip() for line in fh if line.strip()]
    tickers = list(dict.fromkeys(t.upper().replace(".", "-") for t in tickers))  # de-dupe
    if not tickers:
        parser.error("give at least one ticker or --file")
    for var in ("ANTHROPIC_API_KEY", "SEC_USER_AGENT"):
        if not os.environ.get(var):
            sys.exit(f"Set the {var} environment variable first (see top of this file).")

    results, etfs = screen(
        tickers, av_key=os.environ.get("ALPHAVANTAGE_API_KEY", ""), etf_top=args.etf_top,
        progress=lambda i, n, t: print(f"... {t} ({i + 1} of {n})", file=sys.stderr))
    with open(args.out, "w") as fh:
        fh.write(to_json(results, etfs))
    print_report(results, etfs)
    print(f"\nFull detail saved to {args.out}")
    if not args.no_pdf:
        try:
            save_pdf(results, args.pdf, etfs)
            print(f"PDF report saved to {args.pdf}")
        except Exception as exc:  # e.g. fpdf2 missing, or the PDF is open in a viewer
            print(f"Could not write the PDF: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()