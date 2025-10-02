# Context

Current lightgbm model for Binance BTCUSDT Perp is using the original OHLCV data file: data/BINANCE_BTCUSDT.P, 60.CSV, which is a 1h based data.

## The feature engineering process 

In general it involved:

1. Generate lookback data for different timeframes (1h, 4h, 12h and 1d) on per row level. Lookback data was stored as pkl files for each timeframe under /Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60
2. Different feature engineering scripts generated different sets of features using lookback data as inputs for efficiency and avoiding data leakage. Different feature outputs were stored under /Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60 too.

## The target variables

1. Different target variables are generated from original OHLCV data with different forward windows.
2. Target variable files are stored under /Volumes/Extreme SSD/trading_data/cex/targets/BINANCE_BTCUSDT.P, 60

## Regime features from HMM

1. One HMM was trained using the same original OHLCV data and same train/val/test splits, on train split only. Then it was used to generate HMM predictions on all the splits.
2. To train the HMM model, several same features from the previous feature engineering process were used, and stored under /Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60 as hmm_v1_features.csv, hmm_v2_features.csv and hmm_features.csv. hmm_features.csv is the final regime features file to be used for HMM training.
3. The prediction outputs (regime features) from HMM model is stored under /Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/regimes.csv, which was then used to further lightgbm model training.

# Issue

Because all the model trainings were done in an iterative manner, so we were continuously adding/removing features into model training, during those process, we generated lots of merged features and target .csv files as model training inputs under /Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60, prefix with `merged`, and two major issues with them:

1. They have overlapped features and targets columns
2. We used many one-off scripts to merge the data, and some of the merged files have different number of rows after the one-off scripts dropped the rows near the recent timestamps, because the some target variables near the recent timestamps can't be calculated (forward time window data does not present in the original data)

# Goal

1. Create two clean .csv files for future model training usage, one for all the features, one for all the targets
2. Two .csv files should have the same number of rows with matching timestamps
3. Also create two .json files to store the metadata info about the features or targets, the purpose of this is to make the column names human readable.  