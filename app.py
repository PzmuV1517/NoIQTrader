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
    # Add RL model import
    from src.rl_model import RLTradingModel
except ImportError:
    # Fallback for direct execution
    sys.path.append('src')
    from paper_trader import BacktestEngine
    from model_persistence import ModelManager
    from ml_models import TradingSignalPredictor
    try:
        from rl_model import RLTradingModel
    except ImportError:
        RLTradingModel = None


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


@st.cache_data(ttl=300)  # Cache for 5 minutes only
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


@st.cache_data(ttl=300)  # Cache for 5 minutes only
def load_rl_backtest_data():
    """Load RL model backtest results with caching."""
    try:
        from src.model_persistence import ModelManager
        manager = ModelManager()
        
        # Load RL model
        rl_model = manager.load_rl_model()
        if not rl_model or not rl_model.is_trained:
            return None
        
        # Load data for backtesting
        data = pd.read_csv('data/btc_featured_data.csv', index_col=0, parse_dates=True)
        recent_data = data.tail(365)  # Use last year of data
        
        # Create trade history by simulating RL trading decisions
        trade_history = []
        portfolio_value = 10000  # Starting balance
        btc_holdings = 0
        cash = portfolio_value
        
        for i, (date, row) in enumerate(recent_data.iterrows()):
            if i < 20:  # Need enough data for lookback window
                continue
                
            # Get RL prediction
            window_data = recent_data.iloc[max(0, i-50):i+1]
            prediction = rl_model.predict(window_data)
            
            action = prediction.get('action', 0)
            action_name = prediction.get('action_name', 'Hold')
            confidence = prediction.get('confidence', 0)
            
            current_price = row['Close']
            
            # Execute trades based on RL action
            if action == 1 and cash > 100:  # Buy
                btc_amount = cash * 0.95 / current_price  # Use 95% of cash
                btc_holdings += btc_amount
                cash = cash * 0.05  # Keep 5% as cash
                
                trade_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'Buy',
                    'btc_price': current_price,
                    'amount': btc_amount,
                    'confidence': confidence,
                    'portfolio_value_after': cash + btc_holdings * current_price
                })
                
            elif action == -1 and btc_holdings > 0:  # Sell
                cash += btc_holdings * current_price * 0.999  # 0.1% transaction cost
                
                trade_history.append({
                    'date': date.strftime('%Y-%m-%d'), 
                    'action': 'Sell',
                    'btc_price': current_price,
                    'amount': btc_holdings,
                    'confidence': confidence,
                    'portfolio_value_after': cash
                })
                
                btc_holdings = 0
            
            # Track portfolio value for holds
            current_portfolio_value = cash + btc_holdings * current_price
        
        # Calculate final metrics
        final_value = cash + btc_holdings * recent_data.iloc[-1]['Close']
        total_return = (final_value - 10000) / 10000
        
        return {
            'trade_history': trade_history,
            'total_return': total_return,
            'final_portfolio_value': final_value,
            'initial_cash': 10000
        }
        
    except Exception as e:
        print(f"Error creating RL backtest: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes only
def load_model_info():
    """Load model information with caching."""
    try:
        manager = ModelManager()
        # Get comprehensive model info including RL models
        all_model_info = manager.get_all_model_info('data/btc_featured_data.csv')
        
        # Try to load ML predictor
        try:
            predictor = manager.load_latest_models('data/btc_featured_data.csv')
        except:
            predictor = None
        
        return all_model_info, predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None


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


def create_rl_price_chart_with_trades(data, rl_trade_history):
    """Create interactive price chart with RL trade markers."""
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
    
    # Extract RL trade data
    if rl_trade_history:
        buy_dates = []
        buy_prices = []
        sell_dates = []
        sell_prices = []
        
        for trade in rl_trade_history:
            trade_date = pd.to_datetime(trade['date'])
            if trade['action'] == 'Buy':
                buy_dates.append(trade_date)
                buy_prices.append(trade['btc_price'])
            elif trade['action'] == 'Sell':
                sell_dates.append(trade_date)
                sell_prices.append(trade['btc_price'])
        
        # Buy markers
        if len(buy_dates) > 0:
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='RL Buy Signals',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#00FF00',
                    line=dict(color='#00AA00', width=2)
                ),
                hovertemplate='<b>RL BUY</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
        
        # Sell markers
        if len(sell_dates) > 0:
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='RL Sell Signals',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#FF0080',
                    line=dict(color='#AA0055', width=2)
                ),
                hovertemplate='<b>RL SELL</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': 'Bitcoin Price with RL Trading Signals',
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


def create_rl_portfolio_chart(rl_trade_history):
    """Create RL portfolio value over time chart."""
    if not rl_trade_history:
        return None
    
    fig = go.Figure()
    
    # Extract portfolio values and dates
    dates = [pd.to_datetime(trade['date']) for trade in rl_trade_history]
    portfolio_values = [trade['portfolio_value_after'] for trade in rl_trade_history]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='RL Portfolio Value',
        line=dict(color='#9400D3', width=3),
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
            'text': 'RL Portfolio Value Over Time',
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


def create_comparison_chart(data, buy_dates, buy_prices, sell_dates, sell_prices, rl_trade_history):
    """Create comparison chart showing both ML and RL trades."""
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
    
    # ML Buy markers
    if len(buy_dates) > 0:
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='ML Buy Signals',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green',
                line=dict(color='darkgreen', width=1)
            ),
            hovertemplate='<b>ML BUY</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
    
    # ML Sell markers
    if len(sell_dates) > 0:
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name='ML Sell Signals',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red',
                line=dict(color='darkred', width=1)
            ),
            hovertemplate='<b>ML SELL</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
    
    # RL trade markers
    if rl_trade_history:
        rl_buy_dates = []
        rl_buy_prices = []
        rl_sell_dates = []
        rl_sell_prices = []
        
        for trade in rl_trade_history:
            trade_date = pd.to_datetime(trade['date'])
            if trade['action'] == 'Buy':
                rl_buy_dates.append(trade_date)
                rl_buy_prices.append(trade['btc_price'])
            elif trade['action'] == 'Sell':
                rl_sell_dates.append(trade_date)
                rl_sell_prices.append(trade['btc_price'])
        
        # RL Buy markers
        if len(rl_buy_dates) > 0:
            fig.add_trace(go.Scatter(
                x=rl_buy_dates,
                y=rl_buy_prices,
                mode='markers',
                name='RL Buy Signals',
                marker=dict(
                    symbol='diamond',
                    size=10,
                    color='#00FF00',
                    line=dict(color='#00AA00', width=1)
                ),
                hovertemplate='<b>RL BUY</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
        
        # RL Sell markers
        if len(rl_sell_dates) > 0:
            fig.add_trace(go.Scatter(
                x=rl_sell_dates,
                y=rl_sell_prices,
                mode='markers',
                name='RL Sell Signals',
                marker=dict(
                    symbol='diamond',
                    size=10,
                    color='#FF0080',
                    line=dict(color='#AA0055', width=1)
                ),
                hovertemplate='<b>RL SELL</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': 'ML vs RL Trading Signals Comparison',
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


def create_prediction_panel(model_info, predictor):
    """Create current prediction panel."""
    st.markdown("## Latest AI Prediction")
    
    # Create tabs for different model types
    tab1, tab2 = st.tabs(["ML Models", "RL Model"])
    
    with tab1:
        create_ml_prediction_panel(model_info, predictor)
    
    with tab2:
        create_rl_prediction_panel(model_info)


def create_ml_prediction_panel(model_info, predictor):
    """Create ML model prediction panel."""
    try:
        # Get latest prediction from best ML model
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
                <p style="margin: 0; font-size: 0.9rem; color: gray;">Model: Random Forest</p>
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
                title='ML Model Probabilities',
                xaxis_title='Action',
                yaxis_title='Probability',
                template='plotly_white',
                height=300,
                showlegend=False
            )
            st.plotly_chart(prob_fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating ML prediction: {e}")


def create_rl_prediction_panel(model_info):
    """Create RL model prediction panel."""
    rl_info = model_info.get('rl_model', {})
    
    if not rl_info.get('available', False):
        st.info("RL model is not available. Train the RL model using: `python train_rl_model.py`")
        return
    
    if not rl_info.get('is_trained', False):
        st.warning("RL model is available but not trained. Run training to get predictions.")
        return
    
    try:
        # Load RL model and make prediction
        from src.model_persistence import ModelManager
        manager = ModelManager()
        rl_model = manager.load_rl_model()
        
        if rl_model:
            # Load recent data for prediction
            try:
                data = pd.read_csv('data/btc_featured_data.csv', index_col=0, parse_dates=True)
                recent_data = data.tail(50)  # Use last 50 days
                
                # Get RL prediction
                rl_prediction = rl_model.predict(recent_data, return_probabilities=True)
                
                # Create prediction display
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Action recommendation
                    action = rl_prediction['action_name']
                    confidence = rl_prediction['confidence']
                    
                    action_color = {
                        'Buy': '#00ff00',
                        'Sell': '#ff0000', 
                        'Hold': '#ffaa00'
                    }.get(action, '#gray')
                    
                    st.markdown(f"""
                    <div style="background-color: {action_color}22; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid {action_color};">
                        <h2 style="color: {action_color}; margin: 0;">{action.upper()}</h2>
                        <h3 style="margin: 0.5rem 0;">Confidence: {confidence:.1%}</h3>
                        <p style="margin: 0;">Current BTC: ${recent_data['Close'].iloc[-1]:,.2f}</p>
                        <p style="margin: 0; font-size: 0.9rem; color: gray;">Model: Deep Q-Network</p>
                        <p style="margin: 0; font-size: 0.9rem; color: gray;">As of {recent_data.index[-1].strftime('%Y-%m-%d')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Q-values and probabilities
                    if 'probabilities' in rl_prediction:
                        probs = rl_prediction['probabilities']
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
                            title='RL Model Probabilities',
                            xaxis_title='Action',
                            yaxis_title='Probability',
                            template='plotly_white',
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(prob_fig, use_container_width=True)
                    
                    # Show Q-values
                    st.markdown("**Q-Values:**")
                    q_values = rl_prediction['q_values']
                    q_df = pd.DataFrame({
                        'Action': ['Hold', 'Buy', 'Sell'],
                        'Q-Value': q_values
                    })
                    st.dataframe(q_df, use_container_width=True)
                
                # Show RL model performance if available
                if 'performance' in rl_info:
                    perf = rl_info['performance']
                    st.markdown("**RL Model Performance:**")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Episodes Trained", perf.get('episodes_trained', 0))
                    with perf_col2:
                        st.metric("Total Return", f"{perf.get('total_return', 0):.2%}")
                    with perf_col3:
                        st.metric("Best Portfolio", f"${perf.get('best_portfolio_value', 0):,.2f}")
                
            except Exception as e:
                st.error(f"Error loading data for RL prediction: {e}")
        else:
            st.error("Could not load RL model")
            
    except Exception as e:
        st.error(f"Error generating RL prediction: {e}")


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
            ["Dashboard", "Trading Performance", "AI Predictions", "Trade History", "Model Info"],
            index=0,
            help="Select the page you want to view"
        )
        
        st.markdown("### About NoIQTrader")
        st.markdown("""
        - **AI Models**: Random Forest, Logistic Regression & RL
        - **Strategy**: Technical indicator-based signals  
        - **Features**: 64 engineered features
        - **Backtesting**: 2024-2025 period
        - **Virtual Portfolio**: $10,000 starting capital
        """)
        
        # Model status
        st.markdown("### System Status")
        model_info, predictor = load_model_info()
        if model_info:
            st.success(" Models Loaded")
            st.info(f" Features: {model_info.get('feature_count', 'N/A')}")
            
            # Count available models
            ml_models = len(model_info.get('models_available', []))
            rl_available = model_info.get('rl_model', {}).get('available', False)
            total_models = ml_models + (1 if rl_available else 0)
            st.info(f" Models: {total_models} ({ml_models} ML + {1 if rl_available else 0} RL)")
            
            if rl_available:
                rl_trained = model_info.get('rl_model', {}).get('is_trained', False)
                st.info(f" RL Status: {'Trained' if rl_trained else 'Available'}")
        else:
            st.error(" Models Not Available")
        
        # Refresh button
        st.markdown("### Data Control")
        if st.button(" Refresh All Data", help="Clear cache and reload all data"):
            st.cache_data.clear()
            st.rerun()
        
        # Auto-refresh info
        st.caption(f"Data cached for 5 minutes. Last loaded: {datetime.now().strftime('%H:%M:%S')}")
    
    # Load data
    results, engine = load_backtest_data()
    
    if results is None:
        st.error("Could not load backtest data. Please check your data files.")
        return
    
    # Main content based on page selection
    if page == "Dashboard":
        # Header with timestamp
        col_title, col_time = st.columns([3, 1])
        with col_title:
            st.markdown("## Portfolio Overview")
        with col_time:
            st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Performance metrics
        create_performance_metrics(results['performance'])
        
        st.markdown("---")
        
        # Create tabs for different model views
        tab1, tab2, tab3 = st.tabs([" ML Model Trades", " RL Model Trades", " Model Comparison"])
        
        with tab1:
            st.markdown("### Random Forest Model Performance")
            # Charts
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price chart with ML trades
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
        
        with tab2:
            st.markdown("### RL Model Performance")
            # Load RL data
            try:
                rl_results = load_rl_backtest_data()
                if rl_results and rl_results['trade_history']:
                    # RL Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        rl_buy_count = sum(1 for trade in rl_results['trade_history'] if trade['action'] == 'Buy')
                        st.metric("Buy Signals", rl_buy_count)
                    with col2:
                        rl_sell_count = sum(1 for trade in rl_results['trade_history'] if trade['action'] == 'Sell')
                        st.metric("Sell Signals", rl_sell_count)
                    with col3:
                        st.metric("Total Return", f"{rl_results.get('total_return', 0):.2%}")
                    with col4:
                        st.metric("Final Value", f"${rl_results.get('final_portfolio_value', 0):,.2f}")
                    
                    # RL Charts
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # RL Price chart with trades
                        rl_price_fig = create_rl_price_chart_with_trades(
                            results['data'], rl_results['trade_history']
                        )
                        st.plotly_chart(rl_price_fig, use_container_width=True)
                    
                    with col2:
                        # RL Portfolio value chart
                        rl_portfolio_fig = create_rl_portfolio_chart(rl_results['trade_history'])
                        if rl_portfolio_fig:
                            st.plotly_chart(rl_portfolio_fig, use_container_width=True)
                        else:
                            st.info("Portfolio chart not available")
                else:
                    st.warning("RL model data not available. The model may still be training.")
                    st.info(" Check back after the RL training completes!")
            except Exception as e:
                st.error(f"Error loading RL data: {e}")
        
        with tab3:
            st.markdown("### ML vs RL Model Comparison")
            try:
                rl_results = load_rl_backtest_data()
                if rl_results and rl_results['trade_history']:
                    # Comparison metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Random Forest Model")
                        st.metric("Total Return", f"{results['performance']['total_return_pct']:.2f}%")
                        st.metric("Final Value", f"${results['performance']['final_value']:,.2f}")
                        ml_trades = results['performance']['total_trades']
                        st.metric("Total Trades", ml_trades)
                    
                    with col2:
                        st.markdown("#### RL Model")  
                        st.metric("Total Return", f"{rl_results.get('total_return', 0):.2%}")
                        st.metric("Final Value", f"${rl_results.get('final_portfolio_value', 0):,.2f}")
                        rl_trades = len(rl_results['trade_history'])
                        st.metric("Total Trades", rl_trades)
                    
                    # Comparison chart
                    buy_dates, buy_prices, sell_dates, sell_prices = engine.get_trade_markers()
                    comparison_fig = create_comparison_chart(
                        results['data'], buy_dates, buy_prices, sell_dates, sell_prices, 
                        rl_results['trade_history']
                    )
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Performance summary
                    st.markdown("#### Performance Summary")
                    ml_return = results['performance']['total_return_pct']
                    rl_return = rl_results.get('total_return', 0) * 100
                    
                    if ml_return > rl_return:
                        st.success(f" Random Forest outperforms RL by {ml_return - rl_return:.2f}%")
                    elif rl_return > ml_return:
                        st.success(f" RL outperforms Random Forest by {rl_return - ml_return:.2f}%")
                    else:
                        st.info(" Both models show similar performance")
                else:
                    st.warning("RL comparison not available. RL model may still be training.")
            except Exception as e:
                st.error(f"Error creating comparison: {e}")
    
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
        st.markdown("## Trading Model Comparison")
        
        # Create tabs for different models
        tab1, tab2 = st.tabs([" Random Forest Model", " RL Model"])
        
        with tab1:
            st.markdown("### Random Forest Trading History")
            st.info("This shows the current main dashboard performance (1486.54% return)")
            
            # Trade summary for Random Forest
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
        
        with tab2:
            st.markdown("### RL Model Trading History")
            
            # Load and run RL model backtest
            try:
                rl_results = load_rl_backtest_data()
                if rl_results:
                    # RL Trade summary
                    rl_buy_count = sum(1 for trade in rl_results['trade_history'] if trade['action'] == 'Buy')
                    rl_sell_count = sum(1 for trade in rl_results['trade_history'] if trade['action'] == 'Sell')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Buy Signals", rl_buy_count)
                    with col2:
                        st.metric("Sell Signals", rl_sell_count)
                    with col3:
                        st.metric("Hold Days", len(rl_results['trade_history']) - rl_buy_count - rl_sell_count)
                    with col4:
                        st.metric("Total Return", f"{rl_results.get('total_return', 0):.2%}")
                    
                    # RL Trade history table
                    create_trade_history_table(rl_results['trade_history'])
                else:
                    st.warning("RL model trade history not available. The model may still be training or not yet evaluated.")
            except Exception as e:
                st.error(f"Error loading RL trade history: {e}")
                st.info(" The RL model may still be training. Check back after training completes!")
    
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
