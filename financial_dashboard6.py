import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
from datetime import datetime, timedelta
import warnings
from functools import lru_cache
import json

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set page configuration with improved title and icon
st.set_page_config(
    page_title="WealthVision Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved aesthetics and dark mode support
st.markdown("""
    <style>
    :root {
        --background-color: #f5f7fa;
        --text-color: #2c3e50;
        --accent-color: #007bff;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --card-bg: white;
        --border-color: #eaeaea;
    }
    
    /* Dark mode colors */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #121212;
            --text-color: #e0e0e0;
            --accent-color: #4da6ff;
            --card-bg: #1e1e1e;
            --border-color: #333333;
        }
    }
    
    .main { 
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stButton>button { 
        background-color: var(--accent-color); 
        color: white;
        font-weight: 500;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    h1, h2, h3 { 
        color: var(--text-color);
        font-weight: 600;
    }
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-color);
    }
    .metric-label {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.8;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dotted var(--text-color);
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: var(--text-color);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .tab-content {
        padding: 1rem;
        border: 1px solid var(--border-color);
        border-radius: 0 0 8px 8px;
        margin-top: -1px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with better organization
if 'initialized' not in st.session_state:
    st.session_state.holdings = []
    st.session_state.personal_data = {
        "age": 30,
        "risk_appetite": "Medium",
        "investment_horizon": 10,
        "monthly_income": 50000.0,
        "monthly_expenses": 30000.0,
        "savings": 100000.0,
        "debt_amount": 0.0,
        "monthly_debt_payment": 0.0,
        "inflation_rate": 0.06
    }
    st.session_state.financial_goal = 10000000  # Default goal: ₹1 crore
    st.session_state.market_data_cache = {}
    st.session_state.initialized = True

# API configuration - moved to a central location
API_CONFIG = {
    "FMP_API_KEY": "SfoG8QykrZHrL6jBW4EqwCCT1DsUsEmU",  # Replace with your actual key
    "ALPHA_VANTAGE_API_KEY": "SfoG8QykrZHrL6jBW4EqwCCT1DsUsEmU"  # Replace with your actual key
}

# ====== PORTFOLIO OPTIMIZATION FUNCTIONS ======

def calculate_expected_returns(prices):
    """Calculate expected returns using mean historical returns"""
    returns = prices.pct_change().dropna()
    expected_returns = returns.mean() * 252  # Annualize daily returns
    return expected_returns

def calculate_covariance_matrix(prices):
    """Calculate sample covariance matrix from historical prices"""
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252  # Annualize daily covariance
    return cov_matrix

def portfolio_return(weights, expected_returns):
    """Calculate portfolio expected return"""
    return np.sum(weights * expected_returns)

def portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility (risk)"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.05):
    """Calculate portfolio Sharpe ratio"""
    p_return = portfolio_return(weights, expected_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return (p_return - risk_free_rate) / p_volatility

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.05):
    """Negative Sharpe ratio for minimization"""
    return -portfolio_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate)

def minimize_volatility(weights, expected_returns, cov_matrix, target_return):
    """Objective function for volatility minimization with target return"""
    p_volatility = portfolio_volatility(weights, cov_matrix)
    p_return = portfolio_return(weights, expected_returns)
    return p_volatility + 100 * max(0, target_return - p_return)**2  # Penalty for not meeting target return

def optimize_portfolio(expected_returns, cov_matrix, objective='sharpe', target_return=None):
    """
    Optimize portfolio based on specified objective:
    - 'sharpe': Maximize Sharpe ratio
    - 'min_volatility': Minimize volatility
    - 'efficient_return': Minimize volatility for a given target return
    """
    num_assets = len(expected_returns)
    
    # Initial guess (equal weights)
    initial_weights = np.ones(num_assets) / num_assets
    
    # Constraints
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weight between 0-100%
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    if objective == 'sharpe':
        # Maximize Sharpe Ratio
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
    elif objective == 'min_volatility':
        # Minimize Volatility
        result = minimize(
            portfolio_volatility,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
    elif objective == 'efficient_return' and target_return is not None:
        # Efficient Return (minimize volatility for a given target return)
        result = minimize(
            minimize_volatility,
            initial_weights,
            args=(expected_returns, cov_matrix, target_return),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
    else:
        raise ValueError("Invalid optimization objective or missing target return")
    
    # Clean small weights (less than 0.5%)
    weights = result['x']
    weights[weights < 0.005] = 0
    weights = weights / np.sum(weights)  # Normalize to ensure sum is 1.0
    
    # Calculate portfolio performance
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = portfolio_sharpe_ratio(weights, expected_returns, cov_matrix)
    
    return {
        'weights': dict(zip(expected_returns.index, weights)),
        'performance': {
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    }

def generate_efficient_frontier(expected_returns, cov_matrix, points=50):
    """Generate points along the efficient frontier"""
    min_ret = min(expected_returns)
    max_ret = max(expected_returns)
    
    target_returns = np.linspace(min_ret, max_ret, points)
    efficient_frontier = []
    
    for target in target_returns:
        try:
            result = optimize_portfolio(expected_returns, cov_matrix, 'efficient_return', target)
            efficient_frontier.append({
                'return': result['performance']['return'],
                'volatility': result['performance']['volatility']
            })
        except:
            continue
    
    return pd.DataFrame(efficient_frontier)

# ====== HELPER FUNCTIONS ======

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_info(ticker):
    """Fetch comprehensive stock information with improved error handling"""
    try:
        # Basic info from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Additional sector data from FMP if yfinance fails
        if 'sector' not in info or info['sector'] is None:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={API_CONFIG['FMP_API_KEY']}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                fmp_data = response.json()
                if fmp_data and len(fmp_data) > 0:
                    info['sector'] = fmp_data[0].get('sector', 'Unknown')
                    info['industry'] = fmp_data[0].get('industry', 'Unknown')
                    
        return {
            "name": info.get('shortName', ticker),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "price": info.get('currentPrice', 0),
            "currency": info.get('currency', 'USD'),
            "beta": info.get('beta', 0),
            "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
    except Exception as e:
        return {
            "name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "price": 0,
            "currency": "Unknown",
            "beta": 0,
            "dividend_yield": 0
        }

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_top_performers(sector, limit=5):
    """Fetch top-performing stocks in a sector with fallback mechanism"""
    try:
        # Try FMP API first
        url = f"https://financialmodelingprep.com/api/v3/stock-screener?sector={sector}&limit={limit}&apikey={API_CONFIG['FMP_API_KEY']}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                return [stock["symbol"] for stock in data]
        
        # Fallback to predefined lists if API fails
        sector_fallbacks = {
            "Technology": ["INFY.NS", "TCS.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS"],
            "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
            "Consumer Goods": ["HUL.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "MARICO.NS"],
            "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
            "Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "NTPC.NS", "POWERGRID.NS"]
        }
        return sector_fallbacks.get(sector, [])
    except Exception:
        return []

def monte_carlo_simulation(initial_value, expected_return, volatility, years, n_simulations=1000):
    """Perform Monte Carlo simulation with optimized NumPy operations"""
    # More efficient time step calculation
    n_steps = int(years * 12)  # Monthly steps for better performance
    dt = 1 / 12
    
    # Preallocate array for better memory management
    simulations = np.zeros((n_simulations, n_steps + 1))
    simulations[:, 0] = initial_value
    
    # Vectorized operations for better performance
    for t in range(1, n_steps + 1):
        # Generate all random shocks at once
        random_shocks = np.random.normal(0, 1, n_simulations)
        # Vectorized update of all simulations at once
        simulations[:, t] = simulations[:, t-1] * np.exp(
            (expected_return - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks
        )
    
    return simulations

@st.cache_data(ttl=43200)  # Cache for 12 hours
def get_historical_data(tickers, period="5y"):
    """Fetch historical stock data with improved error handling"""
    if not tickers:
        return pd.DataFrame()
        
    try:
        data = yf.download(tickers, period=period)["Adj Close"]
        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data, columns=[tickers[0]])
        return data
    except Exception as e:
        st.warning(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def calculate_portfolio_risk_return(holdings):
    """Calculate portfolio's expected return and risk metrics"""
    if not holdings:
        return 0, 0, 0
        
    # Get stock tickers
    tickers = [h["ticker_or_name"] for h in holdings 
              if h["asset_class"] in ["Stocks", "Mutual Funds", "US Stocks"]]
    
    if not tickers:
        # Calculate weighted average for non-stock assets
        weights = [h["amount"] for h in holdings]
        total = sum(weights)
        weights = [w/total for w in weights]
        expected_return = sum(h["expected_return"] * w for h, w in zip(holdings, weights))
        return expected_return, 0.1, 0  # Assume fixed volatility for non-stock holdings
    
    try:
        # Get historical data
        data = get_historical_data(tickers)
        if data.empty:
            raise ValueError("No historical data available")
            
        # Calculate returns and covariance
        returns = data.pct_change().dropna()
        annual_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Calculate portfolio metrics
        weights = np.array([next((h["amount"] for h in holdings if h["ticker_or_name"] == ticker), 0) 
                           for ticker in data.columns])
        weights = weights / weights.sum()
        
        portfolio_return = np.sum(annual_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 5%)
        risk_free_rate = 0.05
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    except Exception as e:
        # Fall back to weighted average for expected return
        weights = [h["amount"] for h in holdings]
        total = sum(weights)
        weights = [w/total for w in weights]
        expected_return = sum(h["expected_return"] * w for h, w in zip(holdings, weights))
        return expected_return, 0.15, 0  # Assume higher volatility due to error

def save_to_file():
    """Save current portfolio and personal data to file"""
    data = {
        "personal_data": st.session_state.personal_data,
        "holdings": st.session_state.holdings,
        "financial_goal": st.session_state.financial_goal,
        "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to JSON string
    json_data = json.dumps(data, indent=4)
    
    # Offer download through Streamlit
    st.download_button(
        label="Download Portfolio Data",
        data=json_data,
        file_name="wealthvision_portfolio.json",
        mime="application/json"
    )

def load_from_file():
    """Load portfolio and personal data from uploaded file"""
    uploaded_file = st.file_uploader("Upload your saved portfolio file", type=["json"])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.session_state.personal_data = data.get("personal_data", {})
            st.session_state.holdings = data.get("holdings", [])
            st.session_state.financial_goal = data.get("financial_goal", 10000000)
            st.success(f"Successfully loaded data saved on {data.get('saved_date', 'unknown date')}")
            return True
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return False

def display_metric_card(title, value, tooltip=None, delta=None, delta_suffix=None):
    """Display a metric in a styled card with optional tooltip and delta"""
    html = f"""
    <div class="metric-card">
        <div class="metric-label">
    """
    if tooltip:
        html += f"""
        <span class="tooltip">{title}
            <span class="tooltiptext">{tooltip}</span>
        </span>
        """
    else:
        html += title
    
    html += f"""
        </div>
        <div class="metric-value">{value}</div>
    """
    
    if delta is not None:
        color = "green" if delta >= 0 else "red"
        symbol = "+" if delta > 0 else ""
        html += f"""
        <div style="color: {color}; font-size: 0.9rem;">
            {symbol}{delta:.2f}{delta_suffix if delta_suffix else ""}
        </div>
        """
    
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ====== MAIN APP ======

# Navigation with improved sidebar
st.sidebar.title("WealthVision Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", [
    "🏠 Dashboard Overview",
    "👤 Personal Finance",
    "💼 Investment Portfolio",
    "📊 Portfolio Optimization",
    "📈 Market Analysis",
    "⚙️ Settings"
])

# Sidebar additional widgets
with st.sidebar.expander("Data Management"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Data"):
            save_to_file()
    with col2:
        if st.button("Load Data"):
            load_successful = load_from_file()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; opacity: 0.7;">
WealthVision v2.0<br>
© 2025 Financial Insights Ltd.
</div>
""", unsafe_allow_html=True)

# ====== DASHBOARD OVERVIEW PAGE ======
if page == "🏠 Dashboard Overview":
    st.title("Financial Dashboard Overview")
    
    # Calculate key metrics
    if st.session_state.holdings:
        total_investments = sum(h["amount"] for h in st.session_state.holdings)
        total_debt = st.session_state.personal_data.get("debt_amount", 0)
        total_net_worth = st.session_state.personal_data.get("savings", 0) + total_investments - total_debt
        monthly_income = st.session_state.personal_data.get("monthly_income", 0)
        monthly_expenses = st.session_state.personal_data.get("monthly_expenses", 0) + st.session_state.personal_data.get("monthly_debt_payment", 0)
        monthly_savings = monthly_income - monthly_expenses
        savings_rate = monthly_savings / monthly_income if monthly_income > 0 else 0
        
        # Main dashboard metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card("Net Worth", f"₹{total_net_worth:,.2f}", 
                               "Your total assets minus liabilities")
        with col2:
            display_metric_card("Monthly Savings", f"₹{monthly_savings:,.2f}", 
                               "Income minus expenses and debt payments", 
                               delta=savings_rate*100, delta_suffix="%")
        with col3:
            # Calculate portfolio return and risk
            returns, risk, sharpe = calculate_portfolio_risk_return(st.session_state.holdings)
            display_metric_card("Portfolio Return", f"{returns:.2%}", 
                               "Expected annual return on investments",
                               delta=sharpe, delta_suffix=" Sharpe")
        
        # Portfolio allocation
        st.markdown("### Portfolio Allocation")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Asset Allocation Chart with improved visualization
            holdings_df = pd.DataFrame(st.session_state.holdings)
            
            if not holdings_df.empty:
                # Group by asset class
                allocation = holdings_df.groupby("asset_class")["amount"].sum().reset_index()
                allocation["percentage"] = allocation["amount"] / allocation["amount"].sum() * 100
                
                # Create donut chart
                fig = go.Figure(data=[go.Pie(
                    labels=allocation["asset_class"],
                    values=allocation["amount"],
                    hole=.4,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Plotly)
                )])
                
                fig.update_layout(
                    title="Asset Allocation",
                    showlegend=True,
                    height=400,
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            # Goal progress chart
            fig = go.Figure()
            
            # Calculate years until goal based on current savings rate and investments
            years_data = np.arange(0, 30, 1)
            monthly_contribution = monthly_savings
            initial_amount = total_net_worth
            annual_return = returns
            inflation_rate = st.session_state.personal_data.get("inflation_rate", 0.06)
            
            # Project growth with compound interest and monthly contributions
            future_values = []
            for year in years_data:
                inflation_adjusted_goal = st.session_state.financial_goal * (1 + inflation_rate) ** year
                future_value = initial_amount * (1 + annual_return) ** year
                future_value += monthly_contribution * 12 * ((1 + annual_return) ** year - 1) / annual_return if annual_return > 0 else monthly_contribution * 12 * year
                future_values.append(future_value)
                
                # Check if goal is reached
                if future_value >= inflation_adjusted_goal:
                    break
            
            # Goal progress chart
            goal_progress = min(100, (total_net_worth / st.session_state.financial_goal) * 100)
            
            # Create a half donut chart for goal progress
            fig = go.Figure()
            
            # Add remaining portion (transparent)
            fig.add_trace(go.Pie(
                values=[goal_progress, 100-goal_progress],
                labels=["Progress", "Remaining"],
                hole=0.7,
                direction='clockwise',
                sort=False,
                marker=dict(colors=['#4CAF50', 'rgba(0,0,0,0)']),
                textinfo='none',
                showlegend=False,
                rotation=90,
            ))
            
            # Add text in center
            fig.update_layout(
                title="Goal Progress",
                annotations=[dict(
                    text=f"{goal_progress:.1f}%",
                    x=0.5, y=0.5,
                    font=dict(size=20, color='#4CAF50'),
                    showarrow=False
                )],
                height=200,
                margin=dict(t=30, b=0, l=0, r=0),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Years to financial goal
            years_to_goal = next((i for i, v in enumerate(future_values) if v >= st.session_state.financial_goal * (1 + inflation_rate) ** i), None)
            
            if years_to_goal is not None:
                st.markdown(f"### Years to Financial Goal: **{years_to_goal}**")
                st.progress(min(1.0, 30 / years_to_goal))
            else:
                st.warning("At current rates, you may not reach your financial goal within 30 years.")
        
        # Financial Health Score
        st.markdown("### Financial Health Score")
        
        # Calculate financial health score based on key metrics
        debt_to_income = total_debt / (monthly_income * 12) if monthly_income > 0 else 1
        emergency_fund_ratio = st.session_state.personal_data.get("savings", 0) / (monthly_expenses * 6)
        investment_ratio = total_investments / total_net_worth if total_net_worth > 0 else 0
        
        # Score components (0-100 scale)
        savings_score = min(100, savings_rate * 200)  # 50% savings rate = perfect score
        debt_score = max(0, 100 - debt_to_income * 200)  # 50% DTI = 0 score
        emergency_score = min(100, emergency_fund_ratio * 100)  # 1x 6-month expenses = perfect score
        investment_score = min(100, investment_ratio * 125)  # 80% invested = perfect score
        
        # Weighted average
        health_score = int(0.3 * savings_score + 0.25 * debt_score + 0.25 * emergency_score + 0.2 * investment_score)
        
        # Display score with color coded gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff5252"},
                    {'range': [40, 70], 'color': "#ffd740"},
                    {'range': [70, 100], 'color': "#4caf50"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': health_score
                }
            }
        ))
        
        # Set layout
        fig.update_layout(
            height=200,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Health score interpretation
            if health_score >= 80:
                st.success("Your financial health is excellent! Continue your disciplined approach.")
            elif health_score >= 60:
                st.info("Good financial health. Focus on the improvement areas below.")
            elif health_score >= 40:
                st.warning("Your financial health needs attention. See recommendations below.")
            else:
                st.error("Urgent attention needed for your financial health. Follow the recommendations.")
        
        # Financial Health Breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_metric_card("Savings Rate", f"{savings_score:.0f}/100", 
                               "Score based on your monthly savings rate")
        with col2:
            display_metric_card("Debt Management", f"{debt_score:.0f}/100", 
                               "Score based on your debt-to-income ratio")
        with col3:
            display_metric_card("Emergency Fund", f"{emergency_score:.0f}/100", 
                               "Score based on your emergency fund adequacy")
        with col4:
            display_metric_card("Investment Strategy", f"{investment_score:.0f}/100", 
                               "Score based on your investment allocation" )
