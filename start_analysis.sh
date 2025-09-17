#!/bin/bash
# NoIQTrader - Start Analysis Environment

echo " Starting NoIQTrader Analysis Environment"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

echo " Virtual environment activated"
echo " Starting Jupyter Notebook..."
echo ""
echo " Open your browser to: http://localhost:8888"
echo " Navigate to: notebooks/btc_analysis.ipynb"
echo ""
echo "Press Ctrl+C to stop Jupyter when finished"
echo ""

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
