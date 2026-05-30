# Turkey Electricity Market Dashboard & Forecasting ⚡📊

An interactive **Streamlit** analytics dashboard application developed to analyze, visualize, and forecast historical prices and electricity consumption trends in the Turkish Electricity Market using machine learning models.

This project focuses on exploring Day-Ahead Market (GÖP), Intra-Day Market (GİP), Market Clearing Price (PTF), and system consumption data to deliver data-driven future insights through predictive modeling.

## 🚀 Features

- **Interactive Dashboard:** Dynamic time-series data filtering and visualization powered by Streamlit.
- **Advanced Data Visualizations:** Granular hourly, daily, and seasonal breakdown of load and price trends.
- **Machine Learning Forecasting:** Future load and price forecasting built with robust time-series trained models (such as Random Forest).

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.x
- **UI & Visualization:** `streamlit`, `plotly`
- **Data Analysis & Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Random Forest, etc.)

## 📊 Data Source & Citation

The historical electricity market data used in this project is sourced from a public dataset available on **Kaggle**, covering Turkey's power consumption, generation, and pricing trends between 2018 and 2023.

*   **Dataset Link:** [Energy Consumption and Pricing Türkiye (2018-2023)](https://www.kaggle.com/datasets/ahmetzamanis/energy-consumption-and-pricing-trkiye2018-2023)
*   **Citation:** Zamanis, A. (2024). *Energy Consumption and Pricing Türkiye (2018-2023)*. Kaggle.

## 📦 Installation & Setup

Follow these steps to set up and run the project locally on your machine:

### 1. Clone the Repository
bash
git clone https://github.com/melikekara3/turkey-electricity-market-dashboard.git
cd turkey-electricity-market-dashboard
### 2. Create and Activate a Virtual Environment (Recommended)
Bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
### 3. Install Dependencies
pip install -r requirements.txt
### 4. Run the Application

streamlit run app.py
