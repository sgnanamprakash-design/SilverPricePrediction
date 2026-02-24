
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Silver Price Predictor",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_package():

    with open('silver_model.pkl', 'rb') as f:
        package = pickle.load(f)
    return package


@st.cache_data
def load_data():
    """Load processed dataset and test results for charts."""
    processed = pd.read_csv('processed_data.csv', parse_dates=['Date'])
    test_results = pd.read_csv('test_results.csv', parse_dates=['Date'])
    return processed, test_results

required_files = ['silver_model.pkl', 'test_results.csv', 'processed_data.csv']
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("⚠️ Model files not found! Please run the training script first.")
    st.code("python train_model.py", language="bash")
    st.info(f"Missing files: {', '.join(missing)}")
    st.stop()

# Load the SINGLE pickle file and extract everything from it
package  = load_model_package()
model    = package['model']       # The trained LinearRegression model
scaler   = package['scaler']      # The StandardScaler
features = package['features']    # List of 12 feature names
metrics  = package['metrics']     # Dictionary of evaluation scores

df, test_results = load_data()

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.9; }
    .metric-card h1 { margin: 5px 0 0 0; font-size: 28px; }

    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-green h3 { margin: 0; font-size: 14px; opacity: 0.9; }
    .metric-card-green h1 { margin: 5px 0 0 0; font-size: 28px; }

    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-orange h3 { margin: 0; font-size: 14px; opacity: 0.9; }
    .metric-card-orange h1 { margin: 5px 0 0 0; font-size: 32px; }

    .info-box {
        background-color: #f0f7ff; border-left: 5px solid #1f77b4;
        padding: 15px; border-radius: 5px; margin: 10px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 🪙 Silver Price Predictor")
    st.markdown("---")

    page = st.radio(
        " ",
        [
         "🔮 Predict Silver Price"
         ],
        index=0
    )



if page == "🔮 Predict Silver Price":


    with st.form("prediction_form"):
        st.markdown("### 📝 Enter Yesterday's Silver Market Data")

        last_row = df.iloc[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**💰 Price Data**")
            prev_close = st.number_input("Yesterday's Closing Price ($)", min_value=0.01, max_value=500.0,
                                         value=float(last_row['Close']), step=0.01, format="%.2f",
                                         help="The price silver closed at yesterday")
            prev_high = st.number_input("Yesterday's Highest Price ($)", min_value=0.01, max_value=500.0,
                                        value=float(last_row['High']), step=0.01, format="%.2f",
                                        help="The highest price silver reached yesterday")
            prev_low = st.number_input("Yesterday's Lowest Price ($)", min_value=0.01, max_value=500.0,
                                       value=float(last_row['Low']), step=0.01, format="%.2f",
                                       help="The lowest price silver reached yesterday")

        with col2:
            st.markdown("**📊 Market Data**")
            prev_volume = st.number_input("Yesterday's Trading Volume", min_value=0, max_value=1000000,
                                          value=int(last_row['Volume']), step=1,
                                          help="Number of silver contracts traded yesterday")
            close_pct_chg = st.number_input("Yesterday's % Price Change", min_value=-50.0, max_value=50.0,
                                            value=0.0, step=0.01, format="%.2f",
                                            help="How much (%) did the price change yesterday?")

        with col3:
            st.markdown("**📅 Prediction Date**")
            pred_date = st.date_input("Today's Date", value=datetime.today(),
                                      help="Select the date you want the prediction for")
            st.markdown("")
            st.markdown("**ℹ️ Auto-calculated from date:**")
            st.markdown(f"- Day of Week: **{pred_date.strftime('%A')}**\n"
                        f"- Quarter: **Q{(pred_date.month - 1) // 3 + 1}**\n"
                        f"- Day of Year: **{pred_date.timetuple().tm_yday}**")

        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predict Silver Price", use_container_width=True)

    if submitted:
        # Extract date components (auto-calculated)
        year        = pred_date.year
        month       = pred_date.month
        day         = pred_date.day
        day_of_week = pred_date.weekday()
        day_of_year = pred_date.timetuple().tm_yday
        quarter     = (month - 1) // 3 + 1
        high_low_diff = prev_high - prev_low

        # Build feature array in EXACT order: must match training order
        input_data = np.array([[
            year, month, day, day_of_week, day_of_year, quarter,
            prev_close, prev_high, prev_low, prev_volume,
            high_low_diff, close_pct_chg
        ]])

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]

        # Display result
        st.markdown("---")
        st.markdown("## Prediction Result")

        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            st.markdown(f"""
            <div class="metric-card-orange">
                <h3>PREDICTED SILVER PRICE FOR {pred_date.strftime('%d-%b-%Y')}</h3>
                <h1>${predicted_price:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        rc1, rc2 = st.columns(2)

