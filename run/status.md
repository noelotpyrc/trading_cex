# 2025-09-15

1. Backfilled the 1h btcusdt.p data (monthly until Aug and daily for Sep) from binance historical data dump, stored in external SSD 
2. Tested out backfilling the most recent data through API
3. Stored the cleaned OHLCV data in duckDB file in external SSD
4. Pulled the cleaned OHLCV data into the data folder as binance_btcusdt_perp_1h.csv
5. Used the binance_btcusdt_perp_1h.csv to generate lookbacks, and then multi TF features, both stored in external SSD
6. Then use the features to make inference with the model files under external SSD's models/run/binance_btcusdt_perp_1h folder, backfilled predictions stored under the inference/binance_btcusdt_perp_1h/prediction folder
7. For 5 and 6, did several data check with the scripts under run/data_check, found one major issue that the OHLCV data pulled from binance official dump may be not so accurate (we found 2024-10-28 20:00:00 with no volume data), compared with the original one used for model training/validation, which is the BINANCE_BTCUSDT.P,60.csv, from tradingview manual download. But for test time period after 2025‑03‑21 04:00:00, there is no data discrepancies.
8. Because the overall discrepancies look minor, we still choose the model trained from original data rather than retraining one for now.

# TODO

1. Create prediction and feature db from backfilling data, then create the updating process
2. Create a local dashboard to pull data from both OHLCV db and prediction+feature db with interactive charts to show:
    1. OHLCV data for given time period (since 2025‑03‑21 04:00:00)
    2. Signals based on predictions on OHLCV data, and if y_true is available, also show signals' correctness
    3. Some other predictor features (from feature importance list)