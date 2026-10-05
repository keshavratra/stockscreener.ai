"""
Simple web UI for screener.py. Keep this file in the same folder as screener.py.

    pip install streamlit anthropic requests
    streamlit run app.py

It opens in your browser at http://localhost:8501 and runs only on your computer.
"""
import json
import os
import re

import streamlit as st

import screener

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="centered")

GROUPS = [("buy", "Buy", "🟢"), ("watch", "Watch", "🟡"), ("avoid", "Avoid", "🔴")]


def screen_one(client, lookup, ticker):
    """Screen one ticker. Failures come back as row['error'] instead of raising."""
    row = {"ticker": ticker}
    try:
        if ticker not in lookup:
            raise ValueError("ticker not found on SEC EDGAR")
        cik, name = lookup[ticker]
        filings = screener.latest_filings(cik)
        if "10-K" not in filings:
            raise ValueError("no 10-K on file (foreign filers use 20-F / 40-F)")
        facts = screener.sec_get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        financials = screener.extract_financials(
            facts, filings.get("10-Q", {}).get("accession"))
        row.update(company=name, filings=filings,
                   **screener.assess(client, ticker, name, filings, financials))
    except Exception as exc:
        row["error"] = str(exc)
    return row


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cik_map():
    return screener.load_cik_map()


# ----------------------------------------------------------------- settings
with st.sidebar:
    st.header("Settings")
    sec_contact = st.text_input("SEC contact (name and email)",
                                value=os.environ.get("SEC_USER_AGENT", ""),
                                help="The SEC requires a contact on every data request.")
    st.caption("Filled in automatically if you set SEC_USER_AGENT with setx.")

# The Claude API key is never shown here; it is read from the environment.
api_key = os.environ.get("ANTHROPIC_API_KEY", "")

# -------------------------------------------------------------------- input
st.title("📈 Stock Screener")
st.caption("Rates financial strength from each company's latest 10-K and 10-Q.")

raw = st.text_area("Stocks to screen", placeholder="AAPL, MSFT, F",
                   help="Ticker symbols separated by commas, spaces, or new lines.")
tickers = list(dict.fromkeys(t.upper() for t in re.split(r"[\s,;]+", raw) if t))
run = st.button("Screen stocks", type="primary", disabled=not tickers)

if run:
    if not api_key:
        st.error("No Claude API key found. In PowerShell run "
                 '`setx ANTHROPIC_API_KEY "sk-ant-..."`, then close the terminal, '
                 "open a new one, and start the app again.")
        st.stop()
    if not sec_contact:
        st.error("Fill in the SEC contact in the sidebar first.")
        st.stop()
    os.environ["SEC_USER_AGENT"] = sec_contact
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        lookup = cik_map()
    except Exception as exc:
        st.error(f"Could not reach the SEC: {exc}")
        st.stop()
    results = []
    bar = st.progress(0.0)
    for i, ticker in enumerate(tickers):
        bar.progress(i / len(tickers), text=f"Analyzing {ticker} ({i + 1} of {len(tickers)})")
        results.append(screen_one(client, lookup, ticker))
    bar.empty()
    st.session_state["results"] = results

# ------------------------------------------------------------------ results
results = st.session_state.get("results")
if results:
    ok = sorted((r for r in results if "error" not in r), key=lambda r: -r["score"])
    errors = [r for r in results if "error" in r]

    if ok:
        st.subheader("Summary")
        st.dataframe(
            [{"Ticker": r["ticker"], "Company": r["company"],
              "Recommendation": r["recommendation"].capitalize(),
              "Strength": r["financial_strength"].capitalize(),
              "Score": r["score"]} for r in ok],
            hide_index=True, width="stretch",
            column_config={"Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=10, format="%d / 10")},
        )

    for key, label, dot in GROUPS:
        group = [r for r in ok if r["recommendation"] == key]
        if not group:
            continue
        st.subheader(f"{dot} {label}")
        for r in group:
            with st.container(border=True):
                st.markdown(f"**{r['ticker']}** · {r['company']}  \n"
                            f"{r['financial_strength'].capitalize()} · {r['score']}/10")
                st.markdown("\n".join(f"- {b}" for b in r["reasoning"]))
                if r["data_gaps"].strip().lower() != "none":
                    st.caption(f"Data gaps: {r['data_gaps']}")
                f = r["filings"]
                links = [f"[10-K, filed {f['10-K']['filed']}]({f['10-K']['url']})"]
                if "10-Q" in f:
                    links.append(f"[10-Q, filed {f['10-Q']['filed']}]({f['10-Q']['url']})")
                st.caption(" · ".join(links))

    if errors:
        st.subheader("Skipped")
        for r in errors:
            st.warning(f"**{r['ticker']}**: {r['error']}")

    st.download_button("Download full results (JSON)", json.dumps(results, indent=2),
                       file_name="results.json", mime="application/json")
    st.caption("A “buy” here means the business is financially strong. No price or "
               "valuation data is used, and this is not financial advice.")