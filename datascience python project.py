import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Dataset

df = pd.read_csv("C:\\Users\\manik\\Downloads\\airline_ticket_prices.csv")
df['month'] = pd.to_datetime(df['month'])

sns.set_theme(style="whitegrid")

# 1: Global Fare Trends Across Conflict Phases 
plt.figure(figsize=(10, 6))
phase_order = df.groupby('conflict_phase')['total_fare_usd'].mean().sort_values().index

ax1 = sns.barplot(data=df, x='total_fare_usd', y='conflict_phase', order=phase_order, palette='viridis', hue='conflict_phase')
if ax1.legend_: ax1.legend_.remove()
plt.title('Obj 1: Average Total Fare by Conflict Phase')
plt.xlabel('Average Total Fare (USD)')
plt.tight_layout()
plt.show()  

# 2: Impact of Fuel Volatility
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='jet_fuel_usd_barrel', y='fuel_surcharge_usd', alpha=0.3, color='red')
plt.title('Obj 2: Jet Fuel Price vs. Fuel Surcharge')
plt.xlabel('Jet Fuel Price (USD/barrel)')
plt.ylabel('Fuel Surcharge (USD)')
plt.tight_layout()
plt.show()  

# Objective 3: Flag Carrier vs. Low-Cost Airlines 
plt.figure(figsize=(8, 6))
ax3 = sns.boxplot(data=df, x='airline_type', y='total_fare_usd', palette='Set2', hue='airline_type')
if ax3.legend_: ax3.legend_.remove()
plt.title('Obj 3: Fare Comparison by Airline Type')
plt.ylabel('Total Fare (USD)')
plt.tight_layout()
plt.show()  

# 4: Regional Market Resilience (YoY Change) 
plt.figure(figsize=(10, 6))
region_yoy = df.groupby('region')['yoy_price_change_pct'].mean().sort_values()
region_yoy.plot(kind='barh', color='darkblue')
plt.title('Obj 4: Average YoY Price Change % by Region')
plt.xlabel('Average YoY Price Change (%)')
plt.tight_layout()
plt.show()  

#  5: Fare Components by Route Class 
plt.figure(figsize=(12, 6))
route_comp = df.groupby('route_class')[['base_fare_usd', 'fuel_surcharge_usd', 'taxes_fees_usd']].mean()
route_comp.plot(kind='bar', stacked=True, figsize=(12,6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Obj 5: Average Fare Components by Route Class')
plt.ylabel('USD')
plt.xticks(rotation=15)
plt.legend(title='Component')
plt.tight_layout()
plt.show()   

#  6: Load Factor Analysis 
plt.figure(figsize=(10, 6))
sns.histplot(df['load_factor_pct'], bins=30, kde=True, color='purple')
plt.title('Obj 6: Distribution of Flight Load Factors')
plt.xlabel('Load Factor (%)')
plt.tight_layout()
plt.show() 


#  7: Drivers of Ticket Price Inflation (Time Series) 

plt.figure(figsize=(12, 6))
time_comp = df.groupby('month')[['base_fare_usd', 'fuel_surcharge_usd', 'taxes_fees_usd']].mean()
plt.stackplot(time_comp.index, time_comp.T, labels=time_comp.columns, colors=['#aec7e8', '#ffbb78', '#98df8a'])
plt.title('Obj 7: Evolution of Fare Components Over Time')
plt.xlabel('Date')
plt.ylabel('USD')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()  



#  8: Linear Regression Model (Prediction)


df_reg = df.dropna(subset=['total_fare_usd', 'jet_fuel_usd_barrel', 'avg_route_km', 'load_factor_pct'])
X = df_reg[['jet_fuel_usd_barrel', 'avg_route_km', 'load_factor_pct', 'airline_type']]
X = pd.get_dummies(X, columns=['airline_type'], drop_first=True)
y = df_reg['total_fare_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='darkblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Obj 8: Linear Regression \n Actual vs. Predicted (R²={r2:.3f})')
plt.xlabel('Actual Total Fare (USD)')
plt.ylabel('Predicted Total Fare (USD)')
plt.tight_layout()
plt.show() 

print(f"Regression Performance Metrics:")
print(f" - R² Score: {r2:.4f}")
print(f" - RMSE: ${rmse:.2f}")



# 9: Pair Plot

# Selecting the most relevant numerical features for a meaningful pairwise analysis
pair_plot_features = ['total_fare_usd', 'jet_fuel_usd_barrel', 'avg_route_km', 'load_factor_pct', 'airline_type']

# Creating the Pair Plot
# We use 'airline_type' as hue to distinguish between Flag Carriers and Low-Cost airlines
print("Generating Pair Plot... (This may take a moment due to data size)")
sns.pairplot(df[pair_plot_features], hue='airline_type', palette='husl', corner=True)

plt.suptitle('Objective 9: Pair Plot of Key Airline Metrics', y=1.02)
plt.show()