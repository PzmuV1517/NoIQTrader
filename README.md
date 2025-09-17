# NoIQTrader - AI-Powered Bitcoin Trading System

## Phase 1: Data Collection & Feature Engineering COMPLETED

An AI-powered Bitcoin trading system that uses machine learning to predict BTC price movements and simulate paper trades.

### Project Overview

**Current Status**: Phase 1 Complete - Ready for ML Modeling
- 5 years of BTC-USD historical data (1,808 trading days)
- 64 engineered features including technical indicators
- Clean, analysis-ready dataset
- Comprehensive data exploration notebook

### Project Structure

```
NoIQTrader/
├── data/                          # Data storage
│   ├── btc_raw_data.csv          # Raw BTC OHLCV data
│   ├── btc_featured_data.csv     # Processed dataset with features
│   └── feature_summary.csv       # Feature documentation
├── src/                          # Source code modules
│   ├── data_loader.py           # Yahoo Finance data fetching
│   ├── feature_engineer.py     # Technical indicators & features
│   └── utils.py                # Utility functions (future)
├── notebooks/                   # Analysis notebooks
│   └── btc_analysis.ipynb      # Complete data exploration
├── models/                      # ML models (Phase 2)
├── venv/                       # Python virtual environment
├── requirements.txt            # Project dependencies
├── process_data.py            # Data processing script
├── start_analysis.sh          # Quick-start script
└── README.md                  # This file
```

### Features Implemented

**Data Collection**
- Historical BTC-USD data from Yahoo Finance (2020-2025)
- Robust error handling and data validation
- Automatic caching to minimize API calls

**Technical Indicators**
- Moving Averages: MA10, MA50, MA200
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands with position indicators
- Rolling volatility (multiple windows)

**Feature Engineering**
- Lag features (1, 2, 3 days)
- Price ratios and momentum indicators
- Candlestick pattern features
- Volume analysis features
- 64 total features ready for ML modeling

**Data Quality**
- 99.6% data completeness
- Comprehensive missing value handling
- Data validation and integrity checks

### Quick Start

1. **Setup Environment**:
```bash
# Clone and navigate to project
cd NoIQTrader

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

2. **Process Data** (already done, but you can re-run):
```bash
python process_data.py
```

3. **Start Analysis**:
```bash
# Option 1: Use the quick-start script
./start_analysis.sh

# Option 2: Manual Jupyter start
source venv/bin/activate
jupyter notebook notebooks/btc_analysis.ipynb
```

### Current Dataset Stats

- **Time Period**: September 2020 - September 2025 (5 years)
- **Data Points**: 1,808 trading days
- **Features**: 64 (7 original + 57 engineered)
- **Current BTC Price**: $117,266.92
- **5-Year Return**: 1,005.8%
- **Data Quality**: 99.6% complete

### Analysis Capabilities

The Jupyter notebook provides:
- Interactive BTC price charts with technical indicators
- Daily returns distribution analysis
- Feature correlation matrices
- Technical indicator effectiveness analysis
- Risk metrics and volatility analysis
- Comprehensive data quality reports

### Technology Stack

- **Data**: yfinance, pandas, numpy
- **Visualization**: matplotlib, plotly, seaborn
- **Technical Analysis**: Custom implementations
- **Analysis**: Jupyter notebooks
- **ML Ready**: scikit-learn compatible format

### Next Steps (Phase 2)

Ready for implementation:
1. **Machine Learning Models**
   - Logistic Regression for signal classification
   - Random Forest for non-linear patterns
   - LSTM/Transformer for sequence modeling

2. **Paper Trading Simulation**
   - Virtual portfolio management
   - Performance tracking and backtesting
   - Risk management strategies

3. **Web Application**
   - Real-time price monitoring
   - Interactive trading dashboard
   - Model performance visualization

### Contributing

This project is designed to be modular and extensible. Feel free to:
- Add new technical indicators
- Implement additional ML models
- Enhance visualization capabilities
- Extend to other cryptocurrencies

### License

MIT License - Feel free to use and modify for your trading analysis needs.
