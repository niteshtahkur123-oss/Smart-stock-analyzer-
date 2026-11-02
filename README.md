# Smart Stock Analyzer (SSA)

A Streamlit-based NSE stock monitoring prototype. It screens an editable list of NSE symbols for a price range and displays daily technical indicators.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free with Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to a public GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create app, select this repository, branch `main`, and main file `app.py`.
4. Deploy.

## Important limitations
- Uses Yahoo Finance through `yfinance`; this is not an exchange-licensed live market-data feed.
- Data can be delayed, missing, or unavailable. Confirm prices in your broker app.
- Daily candles only in this first prototype. It does not generate 5-minute intraday signals.
- Signals are screening labels, not guaranteed predictions or personalized investment advice.
- The symbol list is editable and is not a complete NSE universe.
