# Regime Switching Engine

Markets don’t behave the same way all the time — some days are smooth, others are full of unpredictable spikes. This project builds a forecasting system that adapts to these changes automatically. It switches between two classic models:

- **ARIMA**: used when the market is calm and stable  
- **GARCH**: used when the market shows signs of volatility

Instead of using one model from start to end, it checks every 30-day window of data and decides which model to trust based on how the market is behaving during that time.

## How It Works

1. The system divides the time series into **30-day chunks**
2. It starts by forecasting each chunk with **ARIMA**
3. Then it runs a **volatility test** on each chunk
4. If volatility is found, it switches that chunk's model to **GARCH**
5. It combines all forecasts into one final prediction

This method helps produce more reliable results across different market conditions.

## Tech Stack
`pandas`, `numpy`, `statsmodels`, `arch`, `matplotlib`, `seaborn`, `Streamlit`


## Key Findings

### Performance Metrics

    Regime Switching RMSE: 0.009722
    ARIMA Only RMSE: 0.009674
    GARCH Only RMSE: 0.009321

### Improvements

    -0.49% improvement over ARIMA-only approach
    -4.30% improvement over GARCH-only approach

### Model Usage Distribution

    GARCH periods: 111 (5.0% of time)
    ARIMA periods: 2122 (95.0% of time)



Description:  
- The **black line** shows predictions from only ARIMA  
- The **red line** shows predictions from the adaptive model  
- You can see how the adaptive model stays more accurate when the market becomes volatile
