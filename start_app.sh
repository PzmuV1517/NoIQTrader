#!/bin/bash

# NoIQTrader Startup Script
# This script activates the virtual environment and starts the Streamlit web application

echo " Starting NoIQTrader Web Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo " Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required files exist
if [ ! -f "data/btc_with_predictions.csv" ]; then
    echo " Creating predictions dataset..."
    python -c "
import sys
sys.path.append('src')
from model_persistence import create_predictions_dataset
create_predictions_dataset('data/btc_featured_data.csv')
"
fi

# Start Streamlit app
echo " Starting Streamlit server..."
echo " Access the app at: http://localhost:8501"
echo " Press Ctrl+C to stop the server"

python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
