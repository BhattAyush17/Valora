import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Valora - Smart Property Valuation", layout="wide")

st.markdown("""
    <style>
    .valora-title {
        text-align: center;
        font-size: 150px;
        font-family: 'Segoe UI', 'Arial', 'sans-serif';
        font-weight: 900;
        background: linear-gradient(90deg, #1a1a1a 0%, #fff 50%, #1a1a1a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        margin-bottom: 0.5em;
        margin-top: 0.2em;
        letter-spacing: 4px;
        text-shadow: 4px 4px 18px #fff, 2px 2px 10px #000;
        border-bottom: 5px solid #fff;
        padding-bottom: 25px;
    }
    </style>
    <h1 class="valora-title">Valora</h1>
""", unsafe_allow_html=True)

st.write("""
Valora is an intelligent, data-driven house price prediction and property analysis tool designed for anyone who wants instant, trustworthy insights from their own real estate data. Simply upload a CSV of property listings with features like area, bedrooms, amenities, and more—Valora will instantly estimate property prices, summarize the market, and reveal what features matter most.

**Why Valora?**
- **Bulk Predictions:** Instantly predict prices for hundreds of properties at once.
- **Smart Market Summaries:** Get average, median, min, and max predicted prices, plus top picks and location-wise trends.
- **Feature Insights:** See which features (like area, bedrooms, furnishing) drive property prices in your dataset.
- **Accuracy Metrics:** If you provide actual prices, Valora shows error metrics and prediction reliability.
- **Custom Visualizations:** Compare price distributions, feature importance, and top properties in easy-to-understand charts.
- **Investor-Friendly:** Spot the best deals, identify premium listings, and analyze what really influences market value.

**Typical Use Cases**
- Home sellers and buyers checking fair market prices.
- Agents and investors analyzing bulk listings and market trends.
- Developers evaluating how amenities impact value.
- Anyone wanting instant, expert-level property analysis from their own spreadsheet.

---
**Get started:** Upload your property data and unlock the power of smart, transparent price prediction and market insights — all in one place, with Valora!
""")

# ====== MODEL SELECTION DROPDOWN WITH INFO ======
st.sidebar.header("🔎 Model Selection")
model_name = st.sidebar.selectbox(
    "Choose Prediction Model:",
    [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "K-Nearest Neighbors"
    ],
    index=0
)

# Dictionary mapping model names to helpful descriptions
model_descriptions = {
    "Linear Regression": (
        "Best for quick, interpretable estimates when relationships between features and price are mostly linear. "
        "Ideal for simple and clean datasets."
    ),
    "Random Forest": (
        "Great for handling complex, non-linear relationships and mixed data types. "
        "Robust against outliers and works well for most property datasets."
    ),
    "Gradient Boosting": (
        "Offers even higher accuracy on challenging datasets with subtle patterns. "
        "Best for advanced users needing powerful, flexible predictions."
    ),
    "K-Nearest Neighbors": (
        "Simple and effective for small datasets with clear clusters. "
        "Not ideal for large or high-dimensional data."
    )
}

st.sidebar.markdown(f"**About this model:** {model_descriptions[model_name]}")

# Upload Data
st.header("📁 Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if not uploaded_file:
    st.stop()
df = pd.read_csv(uploaded_file)
st.write("Preview of uploaded data:")
st.dataframe(df.head())

# Select Target, Features, & Location
st.header("⚙️ Select Target, Inputs & Location Column")
columns = df.columns.tolist()
target_col = st.selectbox("Select target column (e.g. price):", columns)
feature_cols = st.multiselect(
    "Select input columns (features):",
    [col for col in columns if col != target_col],
    default=[col for col in columns if col != target_col][:min(5, len(columns)-1)]
)
location_col = st.selectbox(
    "Select location column (optional):",
    ["(None)"] + [col for col in feature_cols if df[col].nunique() < 30],
    index=0
)
if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# Data Preparation
X_raw = df[feature_cols]
y = df[target_col]
has_actual = np.issubdtype(y.dtype, np.number)
non_numeric_feats = X_raw.select_dtypes(exclude=np.number).columns.tolist()
X = pd.get_dummies(X_raw, drop_first=True) if non_numeric_feats else X_raw.copy()
test_size = st.slider("Test set size (% for accuracy metrics)", 10, 50, 20)
random_state = st.number_input("Random State", value=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=int(random_state)
)

# === Choose Model Based on Dropdown ===
if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_name == "Gradient Boosting":
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
elif model_name == "K-Nearest Neighbors":
    model = KNeighborsRegressor(n_neighbors=5)

# === Fit and Predict as usual ===
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
full_pred = model.predict(X)
df["Predicted Price"] = full_pred

# Combined Insights Graph
st.header("📊 Key Insights at a Glance")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Feature Importance (Red)
# For non-linear models, use feature_importances_ if available, else coef_
if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
elif hasattr(model, "coef_"):
    importance = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False)
else:
    importance = pd.Series([0]*len(X.columns), index=X.columns)

importance.head(7).plot(kind="bar", ax=axs[0], color="red")
axs[0].set_title("Top Feature Importance", fontsize=13, color="black")
axs[0].set_ylabel("Impact on Price", color="black")
axs[0].tick_params(axis='x', labelrotation=30, colors='black')
axs[0].tick_params(axis='y', colors='black')
axs[0].spines['bottom'].set_color('black')
axs[0].spines['left'].set_color('black')

# Price Distribution (Black)
axs[1].hist(df["Predicted Price"], bins=25, color="black", edgecolor="red")
axs[1].set_title("Predicted Price Distribution", fontsize=13, color="red")
axs[1].set_xlabel("Predicted Price", color="black")
axs[1].set_ylabel("Count", color="black")
axs[1].tick_params(axis='x', colors='black')
axs[1].tick_params(axis='y', colors='black')
axs[1].spines['bottom'].set_color('black')
axs[1].spines['left'].set_color('black')

fig.tight_layout()
st.pyplot(fig)

# Final Written Summary
st.header("📝 Market Summary & Best Picks")
avg_price = df["Predicted Price"].mean()
med_price = df["Predicted Price"].median()
min_price = df["Predicted Price"].min()
max_price = df["Predicted Price"].max()
st.write(f"**Average predicted price:** {avg_price:,.0f} | **Median:** {med_price:,.0f} | **Lowest:** {min_price:,.0f} | **Highest:** {max_price:,.0f}")

# Top 3 Costliest and Cheapest Properties
st.subheader("🏆 Top 3 Costliest Properties")
costliest = df.sort_values("Predicted Price", ascending=False).head(3)
for i, row in enumerate(costliest.itertuples(), 1):
    price = row[-1]  # Predicted Price is last column
    area = getattr(row, 'area', getattr(row, 'area_sqft', 'N/A'))
    bedrooms = getattr(row, 'bedrooms', 'N/A')
    st.write(
        f"**{i}. Price:** {price:,.0f} | Area: {area}, Bedrooms: {bedrooms} — Premium property with top features."
    )

st.subheader("💸 Top 3 Cheapest Properties")
cheapest = df.sort_values("Predicted Price", ascending=True).head(3)
for i, row in enumerate(cheapest.itertuples(), 1):
    price = row[-1]  # Predicted Price is last column
    area = getattr(row, 'area', getattr(row, 'area_sqft', 'N/A'))
    bedrooms = getattr(row, 'bedrooms', 'N/A')
    st.write(
        f"**{i}. Price:** {price:,.0f} | Area: {area}, Bedrooms: {bedrooms} — Budget-friendly option."
    )

# Location-wise if available
if location_col != "(None)":
    st.subheader(f"📍 Average Price by {location_col}")
    loc_avg = df.groupby(location_col)["Predicted Price"].mean().sort_values(ascending=False)
    fig_loc, ax_loc = plt.subplots(figsize=(8, 3))
    loc_avg.plot(kind="bar", ax=ax_loc, color="black", edgecolor="red")
    ax_loc.set_ylabel("Avg Predicted Price", color="black")
    ax_loc.set_title(f"Average Predicted Price by {location_col}", color="red")
    ax_loc.tick_params(axis='x', labelrotation=30, colors='black')
    ax_loc.tick_params(axis='y', colors='black')
    ax_loc.spines['bottom'].set_color('black')
    ax_loc.spines['left'].set_color('black')
    st.pyplot(fig_loc)
    st.write(loc_avg)

# Accuracy Metrics (if actual price exists)
if has_actual:
    st.subheader("📈 Prediction Accuracy")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**RMSE:** {rmse:,.0f} | **MAE:** {mae:,.0f} | **R²:** {r2:.2f}")

    # Actual vs Predicted Scatter (black + red)
    fig_sp, ax_sp = plt.subplots(figsize=(5, 5))
    ax_sp.scatter(y_test, y_pred, color="red", alpha=0.7, label="Prediction")
    ax_sp.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "black", lw=2, label="Perfect Prediction")
    ax_sp.set_xlabel("Actual Price", color="black")
    ax_sp.set_ylabel("Predicted Price", color="black")
    ax_sp.set_title("Actual vs Predicted (Test Data)", color="red")
    ax_sp.legend()
    ax_sp.tick_params(axis='x', colors='black')
    ax_sp.tick_params(axis='y', colors='black')
    ax_sp.spines['bottom'].set_color('black')
    ax_sp.spines['left'].set_color('black')
    st.pyplot(fig_sp)

# Downloadable Output
st.header("📥 Download Your Results")
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV with Predicted Price",
    data=csv,
    file_name="Valora_Predicted_Prices.csv",
    mime="text/csv"
)

st.markdown("---")
st.write("Made with ❤️ and care for your data-driven journey. Thank you for trusting Valora to help you make smarter property decisions – may your next move always feel like home!")