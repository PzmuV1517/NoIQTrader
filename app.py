"""
NoIQTrader - Web Interface

Interactive Streamlit application for Bitcoin trading strategy visualization
and live predictions using machine learning models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.paper_trader import BacktestEngine
    from src.model_persistence import ModelManager
    from src.ml_models import TradingSignalPredictor
except ImportError:
    # Fallback for direct execution
    sys.path.append('src')
    from paper_trader import BacktestEngine
    from model_persistence import ModelManager
    from ml_models import TradingSignalPredictor


# Page configuration
st.set_page_config(
    page_title="NoIQTrader - AI Bitcoin Trading",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .profit-positive {
        color: #00ff00;
        font-weight: bold;
    }
    
    .profit-negative {
        color: #ff0000;
        font-weight: bold;
    }
    
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_backtest_data():
    """Load backtest results with caching."""
    try:
        engine = BacktestEngine('data/btc_with_predictions.csv')
        results = engine.run_backtest(
            prediction_column='model_prediction',
            initial_cash=10000,
            start_date='2024-01-01'
        )
        return results, engine
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_data
def load_model_info():
    """Load model information with caching."""
    try:
        manager = ModelManager()
        predictor = manager.load_latest_models('data/btc_featured_data.csv')
        return manager.get_model_info(), predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def create_price_chart_with_trades(data, buy_dates, buy_prices, sell_dates, sell_prices):
    """Create interactive price chart with trade markers."""
    fig = go.Figure()
    
    # Bitcoin price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='BTC Price',
        line=dict(color='#F7931E', width=2),
        hovertemplate='<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>'
    ))
    
    # Buy markers
    if len(buy_dates) > 0:
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='Buy Signals',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(color='darkgreen', width=2)
            ),
            hovertemplate='<b>BUY</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
    
    # Sell markers
    if len(sell_dates) > 0:
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name='Sell Signals',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='<b>SELL</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Bitcoin Price with AI Trading Signals',
            'font': {'size': 24, 'color': '#2E2E2E'},
            'x': 0.5
        },
        xaxis_title='Date',
        yaxis_title='BTC Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_portfolio_chart(portfolio_history, data_dates):
    """Create portfolio value over time chart."""
    portfolio_values = [item['value'] for item in portfolio_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data_dates[:len(portfolio_values)],
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3),
        fill='tonexty',
        hovertemplate='<b>%{x}</b><br>Portfolio: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add initial value line
    fig.add_hline(
        y=10000,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Investment ($10,000)"
    )
    
    fig.update_layout(
        title={
            'text': 'Portfolio Value Over Time',
            'font': {'size': 20, 'color': '#2E2E2E'},
            'x': 0.5
        },
        xaxis_title='Date',
        yaxis_title='Portfolio Value (USD)',
        template='plotly_white',
        height=400,
        hovermode='x'
    )
    
    return fig


def create_performance_metrics(performance):
    """Create performance metrics display."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit_color = "profit-positive" if performance['total_return_pct'] > 0 else "profit-negative"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Return</h3>
            <h2 class="{profit_color}">{performance['total_return_pct']:+.2f}%</h2>
            <p>${performance['absolute_profit']:+,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Portfolio Value</h3>
            <h2>${performance['final_value']:,.2f}</h2>
            <p>From ${performance['initial_value']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Max Drawdown</h3>
            <h2>{performance['max_drawdown_pct']:.2f}%</h2>
            <p>Risk Metric</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        benchmark_color = "profit-positive" if performance['vs_benchmark'] > 0 else "profit-negative"
        st.markdown(f"""
        <div class="metric-card">
            <h3>vs Buy & Hold</h3>
            <h2 class="{benchmark_color}">{performance['vs_benchmark']:+.2f}%</h2>
            <p>Outperformance</p>
        </div>
        """, unsafe_allow_html=True)


def create_prediction_panel(model_info, predictor):
    """Create current prediction panel."""
    st.markdown("## Latest AI Prediction")
    
    try:
        # Get latest prediction
        latest_prediction = predictor.predict_next_action('random_forest')
        
        # Create prediction display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Action recommendation
            action = latest_prediction['action']
            confidence = latest_prediction['confidence']
            
            action_color = {
                'Buy': '#00ff00',
                'Sell': '#ff0000', 
                'Hold': '#ffaa00'
            }.get(action, '#gray')
            
            st.markdown(f"""
            <div style="background-color: {action_color}22; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid {action_color};">
                <h2 style="color: {action_color}; margin: 0;">{action.upper()}</h2>
                <h3 style="margin: 0.5rem 0;">Confidence: {confidence:.1%}</h3>
                <p style="margin: 0;">Current BTC: ${latest_prediction['current_price']:,.2f}</p>
                <p style="margin: 0; font-size: 0.9rem; color: gray;">As of {latest_prediction['date']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Probability distribution
            probs = latest_prediction['probabilities']
            prob_fig = go.Figure(data=[
                go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=['red', 'gray', 'green'],
                    text=[f'{v:.1%}' for v in probs.values()],
                    textposition='auto'
                )
            ])
            
            prob_fig.update_layout(
                title='Action Probabilities',
                xaxis_title='Action',
                yaxis_title='Probability',
                template='plotly_white',
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(prob_fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating prediction: {e}")


def create_trade_history_table(trade_history):
    """Create trade history table."""
    # Filter only buy/sell trades
    trades = [trade for trade in trade_history if trade['action'] in ['Buy', 'Sell']]
    
    if not trades:
        st.warning("No buy/sell trades found.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Select and format columns
    display_df = df[['date', 'action', 'btc_price', 'portfolio_value_after', 'confidence']].copy()
    display_df.columns = ['Date', 'Action', 'BTC Price', 'Portfolio Value', 'Confidence']
    
    # Format columns
    display_df['BTC Price'] = display_df['BTC Price'].apply(lambda x: f"${x:,.2f}")
    display_df['Portfolio Value'] = display_df['Portfolio Value'].apply(lambda x: f"${x:,.2f}")
    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
    
    # Color code actions
    def color_action(val):
        if val == 'Buy':
            return 'background-color: #90EE9090'
        elif val == 'Sell':
            return 'background-color: #FFB6C190'
        return ''
    
    styled_df = display_df.style.applymap(color_action, subset=['Action'])
    
    st.dataframe(styled_df, use_container_width=True, height=400)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">NoIQTrader</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Bitcoin Trading Strategy</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        
        page = st.selectbox(
            "Choose a section:",
            ["Dashboard", "Trading Performance", "AI Predictions", "Trade History", "Model Info"]
        )
        
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### About NoIQTrader")
        st.markdown("""
        - **AI Models**: Random Forest & Logistic Regression
        - **Strategy**: Technical indicator-based signals  
        - **Features**: 57 engineered features
        - **Backtesting**: 2024-2025 period
        - **Virtual Portfolio**: $10,000 starting capital
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model status
        st.markdown("### System Status")
        model_info, predictor = load_model_info()
        if model_info:
            st.success("Models Loaded")
            st.info(f"Features: {model_info.get('feature_count', 'N/A')}")
            st.info(f"Models: {len(model_info.get('models_available', []))}")
        else:
            st.error("Models Not Available")
    
    # Load data
    results, engine = load_backtest_data()
    
    if results is None:
        st.error("Could not load backtest data. Please check your data files.")
        return
    
    # Main content based on page selection
    if page == "Dashboard":
        st.markdown("## Portfolio Overview")
        
        # Performance metrics
        create_performance_metrics(results['performance'])
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price chart with trades
            buy_dates, buy_prices, sell_dates, sell_prices = engine.get_trade_markers()
            price_fig = create_price_chart_with_trades(
                results['data'], buy_dates, buy_prices, sell_dates, sell_prices
            )
            st.plotly_chart(price_fig, use_container_width=True)
        
        with col2:
            # Portfolio value chart
            portfolio_fig = create_portfolio_chart(
                results['portfolio_history'], 
                results['data'].index
            )
            st.plotly_chart(portfolio_fig, use_container_width=True)
    
    elif page == "Trading Performance":
        st.markdown("## Detailed Performance Analysis")
        
        # Performance metrics
        create_performance_metrics(results['performance'])
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sharpe Ratio", f"{results['performance']['sharpe_ratio']:.3f}")
            st.metric("Total Trades", results['performance']['total_trades'])
        
        with col2:
            st.metric("Win Rate", f"{results['performance']['win_rate']:.1f}%")
            st.metric("Total Fees", f"${results['performance']['total_fees']:.2f}")
        
        with col3:
            st.metric("Volatility", f"{results['performance']['volatility']:.3f}")
            st.metric("Current Cash", f"${results['performance']['current_cash']:.2f}")
        
        # Portfolio evolution chart
        portfolio_fig = create_portfolio_chart(
            results['portfolio_history'], 
            results['data'].index
        )
        st.plotly_chart(portfolio_fig, use_container_width=True)
        
        # Drawdown chart
        drawdowns = [item['drawdown'] * 100 for item in results['portfolio_history']]
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            x=results['data'].index[:len(drawdowns)],
            y=drawdowns,
            mode='lines',
            fill='tonexty',
            name='Drawdown %',
            line=dict(color='red', width=2)
        ))
        drawdown_fig.update_layout(
            title='Portfolio Drawdown Over Time',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(drawdown_fig, use_container_width=True)
    
    elif page == "AI Predictions":
        if model_info and predictor:
            create_prediction_panel(model_info, predictor)
            
            st.markdown("---")
            
            # Model comparison
            st.markdown("## Model Performance Comparison")
            
            models_data = []
            for model_name in model_info.get('models_available', []):
                metrics = model_info.get(f'{model_name}_metrics', {})
                models_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Test Accuracy': metrics.get('test_accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1 Score': metrics.get('f1_score', 0)
                })
            
            if models_data:
                comparison_df = pd.DataFrame(models_data)
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.error("AI prediction models are not available.")
    
    elif page == "Trade History":
        st.markdown("## Complete Trade History")
        
        # Trade summary
        buy_count = sum(1 for trade in results['trade_history'] if trade['action'] == 'Buy')
        sell_count = sum(1 for trade in results['trade_history'] if trade['action'] == 'Sell')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Signals", buy_count)
        with col2:
            st.metric("Sell Signals", sell_count)
        with col3:
            st.metric("Hold Days", len(results['trade_history']) - buy_count - sell_count)
        
        # Trade history table
        create_trade_history_table(results['trade_history'])
    
    elif page == "Model Info":
        st.markdown("## Model Information")
        
        if model_info:
            # Model overview
            st.markdown("### Available Models")
            for model in model_info.get('models_available', []):
                metrics = model_info.get(f'{model}_metrics', {})
                with st.expander(f"{model.replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
                        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                    with col2:
                        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
            
            # Feature information
            st.markdown("### Feature Engineering")
            st.info(f"Total Features: {model_info.get('feature_count', 'N/A')}")
            
            if 'features' in model_info:
                feature_types = {}
                for feature in model_info['features']:
                    if any(x in feature for x in ['MA', 'RSI', 'MACD', 'BB']):
                        feature_types.setdefault('Technical Indicators', []).append(feature)
                    elif 'volatility' in feature or 'ATR' in feature:
                        feature_types.setdefault('Volatility', []).append(feature)
                    elif 'lag' in feature:
                        feature_types.setdefault('Lag Features', []).append(feature)
                    else:
                        feature_types.setdefault('Other', []).append(feature)
                
                for feat_type, features in feature_types.items():
                    with st.expander(f"{feat_type} ({len(features)} features)"):
                        st.write(", ".join(features[:10]))
                        if len(features) > 10:
                            st.write(f"... and {len(features) - 10} more")
        else:
            st.error("Model information not available.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888; margin-top: 2rem;">NoIQTrader - AI-Powered Bitcoin Trading | Built with Streamlit & Machine Learning</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
