
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('silver_prices_data.csv')

df.drop(columns=['Volume'], inplace=True)
df.rename(columns={'Open': 'Volume'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

df['Year']       = df['Date'].dt.year
df['Month']      = df['Date'].dt.month
df['Day']        = df['Date'].dt.day
df['DayOfWeek']  = df['Date'].dt.dayofweek
df['DayOfYear']  = df['Date'].dt.dayofyear
df['Quarter']    = df['Date'].dt.quarter


df['Prev_Close']    = df['Close'].shift(1)
df['Prev_High']     = df['High'].shift(1)
df['Prev_Low']      = df['Low'].shift(1)
df['Prev_Volume']   = df['Volume'].shift(1)
df['High_Low_Diff'] = (df['High'] - df['Low']).shift(1)
df['Close_Pct_Chg'] = df['Close'].pct_change().shift(1) * 100

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

feature_cols = [
    'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Quarter',
    'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Volume',
    'High_Low_Diff', 'Close_Pct_Chg',
]

X = df[feature_cols]
y = df['Price']

split_point = int(len(X) * 0.8)

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

test_dates  = df['Date'].iloc[split_point:].values
train_dates = df['Date'].iloc[:split_point].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred  = model.predict(X_test_scaled)

train_mae  = mean_absolute_error(y_train, y_train_pred)
test_mae   = mean_absolute_error(y_test, y_test_pred)
train_mse  = mean_squared_error(y_train, y_train_pred)
test_mse   = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse  = np.sqrt(test_mse)
train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test, y_test_pred)
test_mape  = np.mean(np.abs((y_test.values - y_test_pred) / y_test.values)) * 100

# Feature importance
coeff_df = pd.DataFrame({
    'Feature': feature_cols, 'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

model_package = {

    'model': model,


    'scaler': scaler,


    'features': feature_cols,


    'metrics': {
        'train_mae':  train_mae,
        'test_mae':   test_mae,
        'train_mse':  train_mse,
        'test_mse':   test_mse,
        'train_rmse': train_rmse,
        'test_rmse':  test_rmse,
        'train_r2':   train_r2,
        'test_r2':    test_r2,
        'test_mape':  test_mape,
        'feature_importance': coeff_df.to_dict('records'),
        'intercept':  model.intercept_,
        'train_size': len(X_train),
        'test_size':  len(X_test),
        'total_features': len(feature_cols),
    }
}

# Save as ONE pickle file
with open('silver_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)


test_results = pd.DataFrame({
    'Date': df['Date'].iloc[split_point:].values,
    'Actual': y_test.values,
    'Predicted': y_test_pred
})
test_results.to_csv('test_results.csv', index=False)
df.to_csv('processed_data.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Silver Price Prediction — Linear Regression (No MA/Volatility)',
             fontsize=16, fontweight='bold', y=1.01)


axes[0, 0].plot(test_dates, y_test.values, label='Actual', color='#1f77b4', linewidth=1.2)
axes[0, 0].plot(test_dates, y_test_pred, label='Predicted', color='#ff7f0e', linewidth=1.2, linestyle='--')
axes[0, 0].set_title('Actual vs Predicted (Test Set)', fontweight='bold')
axes[0, 0].set_xlabel('Date'); axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].legend(); axes[0, 0].tick_params(axis='x', rotation=45); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='#2ca02c')
min_v, max_v = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
axes[0, 1].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[0, 1].set_title('Scatter: Actual vs Predicted', fontweight='bold')
axes[0, 1].set_xlabel('Actual'); axes[0, 1].set_ylabel('Predicted')
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)


residuals = y_test.values - y_test_pred
axes[1, 0].hist(residuals, bins=40, color='#9467bd', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[1, 0].set_title('Error Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Error (Actual − Predicted)'); axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)


axes[1, 1].plot(train_dates, y_train.values, color='#1f77b4', linewidth=0.8, label='Training')
axes[1, 1].plot(test_dates, y_test.values, color='#2ca02c', linewidth=0.8, label='Test Actual')
axes[1, 1].plot(test_dates, y_test_pred, color='#ff7f0e', linewidth=0.8, linestyle='--', label='Test Predicted')
axes[1, 1].axvline(x=df['Date'].iloc[split_point], color='red', linestyle='--', linewidth=1.5, label='Split')
axes[1, 1].set_title('Full Timeline', fontweight='bold')
axes[1, 1].set_xlabel('Date'); axes[1, 1].set_ylabel('Price (USD)')
axes[1, 1].legend(fontsize=9); axes[1, 1].tick_params(axis='x', rotation=45); axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('silver_prediction_results.png', dpi=150, bbox_inches='tight')

sample = pd.DataFrame({
    'Date':      df['Date'].iloc[-10:].dt.strftime('%d-%b-%Y').values,
    'Actual($)': np.round(y_test.values[-10:], 2),
    'Predicted($)': np.round(y_test_pred[-10:], 2),
    'Error($)':  np.round(y_test.values[-10:] - y_test_pred[-10:], 2)
})
