"""
Simple web UI for screener.py. Keep this file in the same folder as screener.py.

    pip install streamlit anthropic requests fpdf2
    streamlit run app.py

It opens in your browser at http://localhost:8501 and runs only on your computer.
"""
import csv
import io
import os
import re
from datetime import date

import streamlit as st

import screener

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="centered")

GROUPS = [("buy", "Buy", "🟢"), ("watch", "Watch", "🟡"), ("avoid", "Avoid", "🔴")]
WEB_LIMIT = 10  # most stocks shown per section on the page; the PDF always has them all


def more_note(total, what):
    """Caption saying how many rows were left off the page, if any."""
    if total > WEB_LIMIT:
        st.caption(f"Showing the first {WEB_LIMIT} of {total} {what}. "
                   "The full list is in the PDF report.")

EMAIL_RE = re.compile(r"[^@\s<>()]+@[^@\s<>()]+\.[^@\s<>()]+")
HEADER_WORDS = {"TICKER", "TICKERS", "SYMBOL", "SYMBOLS", "STOCK", "STOCKS"}


def clean_tickers(tokens):
    """Uppercase, turn BRK.B into BRK-B (the SEC's spelling), drop junk and repeats."""
    out = []
    for tok in tokens:
        t = tok.strip().strip("\"'$").upper().replace(".", "-").replace("/", "-")
        if t and t not in HEADER_WORDS and screener.TICKER_OK.match(t) and t not in out:
            out.append(t)
    return out


def tickers_from_file(name, data):
    """Read tickers from an uploaded .txt or .csv file."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if not name.lower().endswith(".csv"):
        return clean_tickers(re.split(r"[\s,;]+", text))
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        return []
    header = [c.strip().upper() for c in rows[0]]
    col = next((i for i, h in enumerate(header) if h in HEADER_WORDS), None)
    if col is not None:                      # a column labeled Ticker / Symbol
        cells = [r[col] for r in rows[1:] if len(r) > col]
    elif len(rows) == 1 or all(len(r) == 1 for r in rows):
        cells = [c for r in rows for c in r]  # one row or one column of tickers
    else:                                     # a table with no label: first column
        cells = [r[0] for r in rows]
    return clean_tickers(cells)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cik_map():
    return screener.load_cik_map()


# ----------------------------------------------------------------- settings
with st.sidebar:
    st.header("Settings")
    _saved = EMAIL_RE.search(os.environ.get("SEC_USER_AGENT", ""))
    email = st.text_input("Your email", value=_saved.group(0) if _saved else "",
                          placeholder="you@example.com",
                          help="The SEC requires a contact email on every data request. "
                               "It is sent only to the SEC.").strip()
    st.subheader("ETFs")
    av_key = st.text_input("Alpha Vantage key", type="password",
                           value=os.environ.get("ALPHAVANTAGE_API_KEY", ""),
                           help="Only needed for ETFs: it supplies each fund's list of "
                                "holdings. Free at alphavantage.co/support/#api-key").strip()
    etf_top = st.number_input("Holdings to screen per ETF", min_value=0, max_value=5000,
                              value=screener.ETF_TOP_DEFAULT, step=5,
                              help="Biggest holdings first. 0 screens every holding, "
                                   "which for a fund like SPY is about 500 companies.")

# The Claude API key is never shown here; it is read from the environment.
api_key = os.environ.get("ANTHROPIC_API_KEY", "")

# -------------------------------------------------------------------- input
st.title("📈 Stock Screener")
st.caption("Rates financial strength from each company's latest 10-K and 10-Q. "
           "Enter an ETF to screen the companies it holds.")

raw = st.text_area("Stocks or ETFs to screen", placeholder="AAPL, MSFT, SPY",
                   help="Ticker symbols separated by commas, spaces, or new lines.")
upload = st.file_uploader("Or upload a list (.csv or .txt)", type=["csv", "txt"],
                          help="A plain list of tickers, or a spreadsheet saved as CSV "
                               "with a column named Ticker or Symbol.")
tickers = clean_tickers(re.split(r"[\s,;]+", raw))
if upload is not None:
    from_file = tickers_from_file(upload.name, upload.getvalue())
    if not from_file:
        st.warning(f"No ticker symbols found in {upload.name}.")
    tickers = list(dict.fromkeys(tickers + from_file))
if tickers:
    shown = ", ".join(tickers[:15]) + (" ..." if len(tickers) > 15 else "")
    st.caption(f"{len(tickers)} symbol{'s' if len(tickers) != 1 else ''} to screen: {shown}")
run = st.button("Screen stocks", type="primary", disabled=not tickers)

if run:
    if not api_key:
        st.error("No Claude API key found. In PowerShell run "
                 '`setx ANTHROPIC_API_KEY "sk-ant-..."`, then close the terminal, '
                 "open a new one, and start the app again.")
        st.stop()
    if not EMAIL_RE.fullmatch(email):
        st.error("Enter a valid email in the sidebar first. The SEC requires one.")
        st.stop()
    os.environ["SEC_USER_AGENT"] = f"StockScreener {email}"
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        lookup = cik_map()
    except Exception as exc:
        st.error(f"Could not reach the SEC: {exc}")
        st.stop()
    bar = st.progress(0.0, text="Looking up tickers and ETF holdings")
    results, etfs = screener.screen(
        tickers, client=client, cik_map=lookup, av_key=av_key, etf_top=int(etf_top),
        progress=lambda i, n, t: bar.progress(i / n, text=f"Analyzing {t} ({i + 1} of {n})"))
    bar.empty()
    st.session_state["results"] = results
    st.session_state["etfs"] = etfs

# ------------------------------------------------------------------ results
results = st.session_state.get("results")
etfs = st.session_state.get("etfs") or []
if results:
    ok = sorted((r for r in results if "error" not in r), key=lambda r: -r["score"])
    errors = [r for r in results if "error" in r]
    score_col = st.column_config.ProgressColumn("Score", min_value=0, max_value=10,
                                                format="%d / 10")

    for etf in etfs:
        info = screener.etf_breakdown(etf, results)
        st.subheader(f"ETF: {etf['ticker']}")
        st.caption((f"{etf['name']}  \n" if etf["name"] else "") + screener.etf_scope(etf))
        w = info["weights"]
        cols = st.columns(4)
        cols[0].metric("🟢 Buy", f"{w['buy']:.1f}%")
        cols[1].metric("🟡 Watch", f"{w['watch']:.1f}%")
        cols[2].metric("🔴 Avoid", f"{w['avoid']:.1f}%")
        cols[3].metric("Weighted score", "n/a" if info["weighted_score"] is None
                       else f"{info['weighted_score']:.1f} / 10")
        note = "Percentages are each rating's share of the whole fund."
        if w["skipped"]:
            note += f" Another {w['skipped']:.1f}% could not be rated (see Skipped)."
        st.caption(note)
        st.dataframe(
            [{"Weight": h["weight_pct"], "Ticker": h["ticker"], "Company": h["company"],
              "Recommendation": (h["recommendation"] or "skipped").capitalize(),
              "Strength": (h["financial_strength"] or "").capitalize(),
              "Score": h["score"]} for h in info["rows"][:WEB_LIMIT]],
            hide_index=True, width="stretch",
            column_config={"Score": score_col,
                           "Weight": st.column_config.NumberColumn(
                               "Weight", format="%.2f%%",
                               help="Percent of the fund held in this company")},
        )
        more_note(len(info["rows"]), "holdings (biggest first)")

    listed = [r for r in ok if not r.get("etfs")] if etfs else ok
    if listed:
        st.subheader("Individual stocks" if etfs else "Summary")
        st.dataframe(
            [{"Ticker": r["ticker"], "Company": r["company"],
              "Recommendation": r["recommendation"].capitalize(),
              "Strength": r["financial_strength"].capitalize(),
              "Score": r["score"]} for r in listed[:WEB_LIMIT]],
            hide_index=True, width="stretch", column_config={"Score": score_col},
        )
        more_note(len(listed), "stocks (highest score first)")

    ordered = [r for key, _, _ in GROUPS for r in ok if r["recommendation"] == key]
    detailed = ordered[:WEB_LIMIT]
    for key, label, dot in GROUPS:
        group = [r for r in detailed if r["recommendation"] == key]
        if not group:
            continue
        st.subheader(f"{dot} {label}")
        for r in group:
            with st.container(border=True):
                tag = f" · {screener.etf_tag(r)}" if r.get("etfs") else ""
                st.markdown(f"**{r['ticker']}** · {r['company']}  \n"
                            f"{r['financial_strength'].capitalize()} · {r['score']}/10{tag}")
                st.markdown("\n".join(f"- {b}" for b in r["reasoning"]))
                if r["data_gaps"].strip().lower() != "none":
                    st.caption(f"Data gaps: {r['data_gaps']}")
                f = r["filings"]
                links = [f"[10-K, filed {f['10-K']['filed']}]({f['10-K']['url']})"]
                if "10-Q" in f:
                    links.append(f"[10-Q, filed {f['10-Q']['filed']}]({f['10-Q']['url']})")
                st.caption(" · ".join(links))

    if len(ordered) > WEB_LIMIT:
        hidden = ordered[WEB_LIMIT:]
        counts = ", ".join(f"{n} {label.lower()}" for key, label, _ in GROUPS
                           if (n := sum(r["recommendation"] == key for r in hidden)))
        st.info(f"Showing the write-ups for the first {WEB_LIMIT} of {len(ordered)} stocks. "
                f"The other {len(hidden)} ({counts}) are in the PDF report.")

    if errors:
        st.subheader("Skipped")
        for r in errors[:WEB_LIMIT]:
            tag = f" ({screener.etf_tag(r)})" if r.get("etfs") else ""
            st.warning(f"**{r['ticker']}**{tag}: {r['error']}")
        more_note(len(errors), "skipped symbols")

    left, right = st.columns(2)
    try:
        left.download_button("Download report (PDF)", screener.build_pdf(results, etfs),
                             file_name=f"stock-screen-{date.today()}.pdf",
                             mime="application/pdf", type="primary")
    except Exception as exc:
        left.warning(f"PDF unavailable: {exc}")
    right.download_button("Download full results (JSON)", screener.to_json(results, etfs),
                          file_name="results.json", mime="application/json")
    st.caption("A “buy” here means the business is financially strong. No price or "
               "valuation data is used, and this is not financial advice.")