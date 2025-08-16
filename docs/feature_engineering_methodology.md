# Feature Engineering Methodology

## Core Principle: Row-by-Row Processing with Lookback Windows

This feature engineering system is designed to simulate real-time trading conditions where at any given timestamp, we only have access to historical data (no future data leakage).

## Architecture Overview

### Input
```python
# Standard 1H OHLCV DataFrame
data = pd.DataFrame({
    'open': [100.0, 101.0, 102.0, ...],
    'high': [101.0, 102.0, 103.0, ...], 
    'low': [99.0, 100.0, 101.0, ...],
    'close': [101.0, 102.0, 102.5, ...],
    'volume': [1000, 1500, 800, ...]
}, index=pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00', ...]))
```

### Processing Methodology

For each row `i` in the dataset:

1. **Extract Lookback Window**: Take the last N rows ending at row i
   ```python
   lookback_data = data.iloc[max(0, i+1-N):i+1]  # N rows ending at current row
   ```

2. **Generate Multi-Timeframe Data**: Resample the lookback window to different timeframes
   ```python
   # 1H: Use original lookback data (N rows)
   data_1h = lookback_data
   
   # 4H: Resample N rows of 1H data into ~N/4 rows of 4H data  
   data_4h = lookback_data.resample('4H').agg({
       'open': 'first', 'high': 'max', 'low': 'min', 
       'close': 'last', 'volume': 'sum'
   })
   
   # 1D: Resample N rows of 1H data into ~N/24 rows of 1D data
   data_1d = lookback_data.resample('1D').agg({
       'open': 'first', 'high': 'max', 'low': 'min',
       'close': 'last', 'volume': 'sum'  
   })
   ```

3. **Calculate Features**: Apply the same feature functions to each timeframe's data
   ```python
   # All timeframes use the same feature calculation functions
   features_1h = {
       'sma_20_1H': calculate_sma(data_1h['close'], 20),
       'rsi_14_1H': calculate_rsi(data_1h['close'], 14),
       'bb_percent_20_1H': calculate_bollinger_bands(data_1h['close'], 20)['bb_percent']
   }
   
   features_4h = {
       'sma_20_4H': calculate_sma(data_4h['close'], 20), 
       'rsi_14_4H': calculate_rsi(data_4h['close'], 14),
       'bb_percent_20_4H': calculate_bollinger_bands(data_4h['close'], 20)['bb_percent']
   }
   ```

4. **Store Results**: All features for row i are stored with the same timestamp

### Example Walkthrough

Consider processing row 100 with lookback_window=96 (4 days of hourly data):

```python
# Row 100 timestamp: 2024-01-05 04:00:00
current_row = 100
lookback_window = 96

# Step 1: Extract lookback data (rows 5-100, 96 rows total)
lookback_data = data.iloc[5:101]  # 96 hours of data

# Step 2: Create timeframe datasets
data_1h = lookback_data  # 96 rows of 1H data
data_4h = resample_to_4h(lookback_data)  # ~24 rows of 4H data  
data_1d = resample_to_1d(lookback_data)  # ~4 rows of 1D data

# Step 3: Calculate features for each timeframe
# 1H features (using 96 1H candles)
sma_20_1h = calculate_sma(data_1h['close'], 20)  # Uses last 20 of 96 1H candles
rsi_14_1h = calculate_rsi(data_1h['close'], 14)  # Uses last 14 of 96 1H candles

# 4H features (using ~24 4H candles) 
sma_20_4h = calculate_sma(data_4h['close'], 20)  # Uses last 20 of ~24 4H candles
rsi_14_4h = calculate_rsi(data_4h['close'], 14)  # Uses last 14 of ~24 4H candles

# 1D features (using ~4 1D candles)
sma_20_1d = np.nan  # Not enough data (need 20 days, only have ~4)
rsi_14_1d = np.nan  # Not enough data (need 14 days, only have ~4)

# Step 4: Store all features for row 100
result.loc['2024-01-05 04:00:00'] = {
    'open': 102.5, 'high': 103.0, 'low': 102.0, 'close': 102.8, 'volume': 1200,
    'sma_20_1H': sma_20_1h,
    'rsi_14_1H': rsi_14_1h, 
    'sma_20_4H': sma_20_4h,
    'rsi_14_4H': rsi_14_4h,
    'sma_20_1D': np.nan,
    'rsi_14_1D': np.nan
}
```

## Key Benefits

### 1. **No Future Data Leakage**
- Each row only uses data available up to that timestamp
- Realistic simulation of live trading conditions
- Features can be generated in real-time streaming fashion

### 2. **Consistent Feature Functions**
- Same calculation logic across all timeframes
- `calculate_sma(data, 20)` works whether data is 1H, 4H, or 1D
- Easy to test and validate individual functions

### 3. **Flexible Timeframe Support**
- Easy to add new timeframes (2H, 6H, 1W, etc.)
- Automatic resampling handles frequency conversion
- Features automatically get appropriate suffixes (_1H, _4H, _1D)

### 4. **Configurable Lookback Windows**
- Short lookback (24 rows): Fast computation, recent data focus
- Long lookback (168 rows): More stable features, longer history
- Can optimize based on feature requirements and computation constraints

## Implementation Structure

### Core Components

1. **Feature Functions** (`core_functions.py`)
   ```python
   def calculate_sma(data: pd.Series, window: int) -> float:
       """Pure function: data + parameters → single feature value"""
   
   def calculate_rsi(data: pd.Series, window: int) -> float:
       """Pure function: data + parameters → single feature value"""
   ```

2. **Timeframe Utilities** (`timeframe_utils.py`)
   ```python
   def resample_lookback_data(data: pd.DataFrame, target_freq: str) -> pd.DataFrame:
       """Resample OHLCV data to target frequency"""
   
   def get_lookback_window(data: pd.DataFrame, current_idx: int, window_size: int) -> pd.DataFrame:
       """Extract lookback window for current row"""
   ```

3. **Feature Pipeline** (`feature_pipeline.py`)
   ```python
   def generate_features_for_row(data: pd.DataFrame, row_idx: int, config: dict) -> dict:
       """Generate all features for a single row"""
   
   def generate_all_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
       """Process entire dataset row by row"""
   ```

### Configuration Format

```python
config = {
    'lookback_window': 168,  # 1 week of hourly data
    'timeframes': ['1H', '4H', '1D'],
    'features': {
        'sma': {'windows': [10, 20, 50]},
        'rsi': {'windows': [14, 21]}, 
        'bollinger_bands': {'windows': [20], 'std_dev': [2.0]},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'volume_ratio': {'windows': [10, 20]}
    }
}
```

### Output Format

```python
# Each row contains original OHLCV + all computed features
enhanced_data = pd.DataFrame({
    # Original data
    'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...],
    
    # 1H timeframe features
    'sma_10_1H': [...], 'sma_20_1H': [...], 'sma_50_1H': [...],
    'rsi_14_1H': [...], 'rsi_21_1H': [...],
    'bb_upper_20_1H': [...], 'bb_percent_20_1H': [...],
    
    # 4H timeframe features  
    'sma_10_4H': [...], 'sma_20_4H': [...], 'sma_50_4H': [...],
    'rsi_14_4H': [...], 'rsi_21_4H': [...],
    'bb_upper_20_4H': [...], 'bb_percent_20_4H': [...],
    
    # 1D timeframe features
    'sma_10_1D': [...], 'sma_20_1D': [...], 'sma_50_1D': [...],
    'rsi_14_1D': [...], 'rsi_21_1D': [...],
    'bb_upper_20_1D': [...], 'bb_percent_20_1D': [...],
    
    # Volume features
    'volume_ratio_10_1H': [...], 'obv_1H': [...], 'vwap_1H': [...]
})
```

## Performance Considerations

### Memory Management
- Process data in chunks for large datasets
- Clear intermediate timeframe data after each row
- Use appropriate dtypes (float32 vs float64)

### Computation Optimization  
- Vectorize calculations within feature functions
- Cache expensive operations (rolling calculations)
- Parallelize across multiple rows when possible

### Scalability Targets
- **Small dataset**: 1K rows, 50 features → < 1 second
- **Medium dataset**: 10K rows, 100 features → < 30 seconds  
- **Large dataset**: 100K rows, 200 features → < 10 minutes

## Testing Strategy

### Unit Tests
```python
def test_sma_calculation():
    """Test SMA with known input/output."""
    data = pd.Series([1, 2, 3, 4, 5])
    assert calculate_sma(data, 3) == 4.0  # (3+4+5)/3

def test_insufficient_data():
    """Test graceful handling of insufficient data."""
    data = pd.Series([1, 2])
    assert np.isnan(calculate_sma(data, 5))
```

### Integration Tests
```python
def test_row_processing():
    """Test complete row processing pipeline."""
    data = create_sample_ohlcv(periods=100)
    config = {'lookback_window': 50, 'timeframes': ['1H', '4H']}
    
    features = generate_features_for_row(data, 99, config)
    assert 'sma_20_1H' in features
    assert 'sma_20_4H' in features
```

### Validation Tests
```python
def test_no_future_leakage():
    """Ensure no future data is used."""
    data = create_trending_data()
    # Modify future data dramatically
    data.iloc[50:] *= 10
    
    # Features at row 49 should be unaffected
    features_before = generate_features_for_row(data[:50], 49, config)
    features_after = generate_features_for_row(data, 49, config)
    
    assert features_before == features_after
```

This methodology ensures robust, realistic feature engineering that can be deployed in live trading systems while maintaining simplicity and testability.