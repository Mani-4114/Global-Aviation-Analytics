Global Airline Price Analysis & Prediction
This project provides a comprehensive data-driven analysis of global airline ticket prices from 2019 to 2025. It examines how major geopolitical events, fuel price fluctuations, and operational factors influence airfares across different regions and airline types.

🚀 Project Overview
The core of this project is to understand the drivers of airfare volatility and build a predictive model to estimate ticket prices using linear regression. The analysis is structured around 9 Key Analytical Objectives, ranging from descriptive statistics to predictive modeling.

📊 Analytical Objectives
Global Fare Trends: Analysis of price fluctuations across conflict phases (e.g., COVID-19, Ukraine War Shock).

Fuel Volatility Impact: Correlating Jet Fuel price changes with passenger fuel surcharges.

Carrier Comparison: Performance and pricing contrast between Flag Carriers and Low-Cost Airlines.

Regional Resilience: Measuring Year-over-Year (YoY) price sensitivity across global markets.

Fare Component Breakdown: Investigating Base Fare vs. Taxes vs. Surcharges by route length.

Load Factor Distribution: Analyzing flight occupancy rates and their impact on revenue.

Inflation Drivers: Time-series analysis of the rising costs in the aviation industry.

Linear Regression Model: Predicting Total Fare based on fuel costs, distance, and load factors.

Pair Plot Analysis: Visualizing high-dimensional relationships between all key metrics.

🤖 Machine Learning Performance
We implemented a Linear Regression model to predict airfares. The model achieved strong predictive power:

R² Score: 0.797 (The model explains approximately 80% of price variance).

Root Mean Squared Error (RMSE): $663.25.

Top Insights:

Fuel Sensitivity: Every $1 increase in Jet Fuel per barrel results in a significant uptick in total ticket cost.

Distance Factor: Route distance remains the most consistent predictor of base fare pricing.

🛠️ Tech Stack
Language: Python 3.x

Data Manipulation: pandas, numpy

Visualization: seaborn, matplotlib

Machine Learning: scikit-learn

📁 Dataset Description
The dataset includes 14,000+ records with the following key features:

total_fare_usd: The target variable for prediction.

conflict_phase: Categorical marker for global geopolitical events.

jet_fuel_usd_barrel: Market price of aviation fuel.

avg_route_km: Distance of the flight route.

airline_type: Flag Carrier vs. Low Cost.

load_factor_pct: Flight occupancy percentage.

💻 How to Run
Clone this repository:

Bash
git clone https://github.com/Mani-4114/Global-Aviation-Analytics.git
Install dependencies:

Bash
pip install pandas seaborn matplotlib scikit-learn
Run the analysis script:

Bash
python project_analysis.py
📈 Visualizations
The script generates 9 distinct visualizations, including:

Boxplots for carrier comparison.

Stacked Bar Charts for fare components.

Scatter Plots for regression analysis and fuel correlations.

Pair Plots for multi-variable discovery.
