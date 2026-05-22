import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import r2_score

st.markdown("""
<h1 style="
    background: linear-gradient(90deg, #3b82f6, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: sans-serif;
">
⚡ Turkey Electricity Market Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.stApp {
    background-color: #f1f5f9 !important;
}

span[data-baseweb="tag"] {
    background-color: #3b82f6 !important;
    color: white !important;
}

div[data-baseweb="select"] > div {
    background-color: #e2e8f0 !important; 
    border: 1px solid #cbd5e1 !important;
}

div[role="tablist"] {
    border: none !important;
    gap: 15px !important;
}

button[role="tab"] {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    padding: 12px 25px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
}

button[role="tab"][aria-selected="true"] {
    background-color: #1e293b !important;
    color: #3b82f6 !important;
    border: 2px solid #3b82f6 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-2px) !important;
}

div.stButton > button {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    padding: 20px 30px !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    text-transform: uppercase !important;
}

div.stButton > button:hover {
    background-color: #0f172a !important;
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3) !important;
}

.desc-box {
    background-color: rgba(59, 130, 246, 0.1);
    border-left: 5px solid #3b82f6;
    padding: 15px;
    border-radius: 8px;
    color: inherit !important;
    margin-bottom: 20px;
    font-family: sans-serif;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    cons_model = joblib.load('consumption_model.pkl')
    price_model = joblib.load('price_model_usd.pkl')
    df = pd.read_csv('clean_data.csv')
    return cons_model, price_model, df


cons_model, price_model, df = load_assets()

tab1, tab2 = st.tabs([" Energy Analysis (2018-2022)", " Prediction Screen"])

with tab1:
    st.title("Turkey Energy Consumption and Source Analysis")

    yil_secimi = st.multiselect("Filter Years:", [2018, 2019, 2020, 2021, 2022], default=[2019, 2020, 2021])
    df_f = df[df['year'].isin(yil_secimi)]

    st.subheader("📊 Period Summary")
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        max_cons = df_f['consumption_MWh'].max()
        st.metric("Peak Consumption", f"{max_cons:,.0f} MWh")

    with kpi2:
        avg_price = df_f['USD/MWh'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")

    with kpi3:
        kaynaklar = ['wind', 'solar', 'natural_gas', 'hydro_dam', 'lignite', 'coal_imported']
        top_source = df_f[kaynaklar].sum().idxmax().replace('_', ' ').title()
        st.metric("Leading Source", top_source)

    st.divider()

    st.subheader("🕒 1. Annual Change: The Rise of Daily Rhythm")

    df_anim_fix = df[df['year'] <= 2022].copy()

    df_hour_full = df_anim_fix.groupby(['year', 'hour'])['consumption_MWh'].mean().reset_index()
    df_hour_full = df_hour_full.sort_values(['year', 'hour'])

    y_range_hour = [df_hour_full['consumption_MWh'].min() * 0.95, df_hour_full['consumption_MWh'].max() * 1.05]

    fig_hour = px.line(
        df_hour_full,
        x='hour', y='consumption_MWh',
        animation_frame='year',
        markers=True,
        range_y=y_range_hour,
        title="24-Hour Consumption Cycle (2018-2022 Actual)",
        template="plotly_white",
        color_discrete_sequence=['#4ade80']
    )
    fig_hour.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a")
    )
    fig_hour.update_yaxes(gridcolor="rgba(0,0,0,0.1)")

    fig_hour.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000
    fig_hour.layout.sliders[0].currentvalue.prefix = "Year: "
    st.plotly_chart(fig_hour, use_container_width=True)
    st.markdown("""
        <div class="desc-box">
            <b>🔍 Analysis Note:</b> This animation shows Turkey's 24-hour energy pulse. 
            The upward shift of the graph over the years represents <b>Capacity Growth</b> driven by industrialization and population growth.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📅 2. Monthly Change: Seasonal Cycle")

    df_month_full = df_anim_fix.groupby(['year', 'month'])['consumption_MWh'].mean().reset_index()
    df_month_full = df_month_full.sort_values(['year', 'month'])

    ay_isimleri = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    df_month_full['Month_Name'] = df_month_full['month'].map(ay_isimleri)

    fig_month = px.bar(
        df_month_full,
        x='Month_Name',
        y='consumption_MWh',
        animation_frame='year',
        color='consumption_MWh',
        range_y=[0, df_month_full['consumption_MWh'].max() * 1.1],
        color_continuous_scale='Viridis',
        template="plotly"
    )

    fig_month.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2500
    st.plotly_chart(fig_month, use_container_width=True)
    st.markdown("""
        <div class="desc-box" style="border-left-color: #ffb703;">
            <b>📅 Seasonal Interpretation:</b> The rise of columns during summer months (July-August) reflects air conditioning use, 
            while the winter rise represents lighting and heating needs. Observe how <b>summer peaks</b> have become more aggressive between 2018 and 2022.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🌡️ Consumption Density Analysis (Hour vs. Day)")
    st.write("This chart shows the 'Hot Spots' where consumption is most intense on a weekly and hourly basis.")

    heatmap_data = df_f.groupby(['day_of_week', 'hour'])['consumption_MWh'].mean().unstack()

    gunler = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data.index = gunler

    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="MWh"),
        x=list(range(24)),
        y=gunler,
        color_continuous_scale='YlOrRd',
        title="Weekly Energy Consumption Pulse",
        template="plotly",
        aspect="auto"
    )

    fig_heat.update_xaxes(side="top")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("""
        <div class="desc-box" style="border-left-color: #e63946;">
            <b>🌡️ Density Guide:</b> This table represents the average of thousands of days across selected years. 
            Dark red areas indicate "Peak Hours" (evening rush), while lighter colors on weekends symbolize <b>baseload</b> periods where industrial load decreases.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("⚡ Source-Based Production Panorama")

    f_col1, f_col2 = st.columns([2, 1])

    with f_col1:
        secili_kaynaklar = st.multiselect(
            "Sources to Analyze:",
            options=['natural_gas', 'hydro_dam', 'lignite', 'wind', 'solar', 'coal_imported'],
            default=['natural_gas', 'wind', 'solar', 'hydro_dam'],
            key="kaynak_ana_filtre"
        )

    with f_col2:
        grafik_gorunumu = st.radio("Chart Focus:", ["Time Series (Bar)", "General Distribution (Sunburst)"],
                                   horizontal=True)

    st.divider()

    df_prod = df[df['year'].isin(yil_secimi)].groupby(['year', 'month'])[secili_kaynaklar].mean().reset_index()
    df_prod_melted = df_prod.melt(id_vars=['year', 'month'], value_vars=secili_kaynaklar, var_name='Source',
                                  value_name='MWh')

    df_prod_melted['Date'] = df_prod_melted['year'].astype(str) + "-" + df_prod_melted['month'].astype(str).str.zfill(2)

    if grafik_gorunumu == "Time Series (Bar)":
        fig = px.bar(
            df_prod_melted,
            x='Date',
            y='MWh',
            color='Source',
            title="Monthly Total Production Capacity and Breakdown by Source",
            barmode='stack',
            template="plotly",
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    else:
        df_sun = df_prod_melted.groupby(['year', 'Source'])['MWh'].sum().reset_index()
        renk_haritasi = {
            'natural_gas': '#E63946',
            'wind': '#A8DADC',
            'solar': '#FFB703',
            'hydro_dam': '#457B9D',
            'lignite': '#1D3557',
            'coal_imported': '#6D6875'
        }

        fig_sun = px.sunburst(
            df_sun,
            path=['year', 'Source'],
            values='MWh',
            color='Source',
            color_discrete_map=renk_haritasi,
            template="plotly"
        )

        fig_sun.update_layout(
            showlegend=True,
            legend=dict(
                title="Energy Sources",
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1
            )
        )

        for kaynak, renk in renk_haritasi.items():
            fig_sun.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=renk),
                legendgroup=kaynak,
                showlegend=True,
                name=kaynak
            ))

        st.plotly_chart(fig_sun, use_container_width=True)
    st.markdown("""
                    <div class="desc-box" style="border-left-color: #457b9d;">
                        <b>⚡ Production Mix:</b> You can monitor Turkey's energy independence journey here. 
                        Compare the balance between the share of natural gas and renewable (Wind/Solar) sources <b>interactively</b> based on your selected years.
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.title("🔮 2023 Smart Prediction Panel")

    if 'model_run' not in st.session_state:
        st.session_state.model_run = False

    if st.button("🚀 RUN MODEL FOR 2023 ANALYSIS ", use_container_width=True, key="run_2023_final_v4"):
        st.session_state.model_run = True

    if st.session_state.model_run:
        df_2023 = df[df['year'] == 2023].copy()

        if df_2023.empty:
            st.error("Error: No data found for the year 2023 in the dataset!")
        else:
            df_2023['trend_index'] = range(len(df_2023))
            df_2023['price_rolling_24h'] = df_2023['USD/MWh'].rolling(window=24).mean()
            df_2023['price_rolling_1w'] = df_2023['USD/MWh'].rolling(window=168).mean()
            df_2023['price_usd_lag_24h'] = df_2023['USD/MWh'].shift(24)
            df_2023['price_usd_lag_48h'] = df_2023['USD/MWh'].shift(48)
            df_2023.bfill(inplace=True)

            cols_cons = ['hour', 'day_of_week', 'month', 'is_weekend', 'consumption_lag_24h', 'consumption_lag_1w',
                         'price_usd_lag_24h', 'hour_sin', 'hour_cos']

            df_2023['prediction_cons'] = cons_model.predict(df_2023[cols_cons])

            cols_price = [
                'hour', 'day_of_week', 'month', 'is_weekend',
                'consumption_MWh', 'consumption_lag_24h',
                'hour_sin', 'hour_cos', 'price_usd_lag_24h',
                'trend_index', 'price_rolling_24h', 'price_rolling_1w', 'price_usd_lag_48h'
            ]

            df_price_features = df_2023[cols_price].copy()
            df_price_features['consumption_MWh'] = df_2023['prediction_cons']

            df_2023['prediction_price'] = price_model.predict(df_price_features)

            st.divider()
            ay_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            df_2023['Month_Name'] = df_2023['month'].map(ay_map)

            secili_aylar = st.multiselect(
                "📅 Select Analysis Period (Affects All Charts):",
                options=list(ay_map.values()),
                default=["January"],
                key="master_selector_2023"
            )

            if secili_aylar:
                plot_df = df_2023[df_2023['Month_Name'].isin(secili_aylar)].sort_values('datetime').copy()

                plot_df['Actual_Cons'] = df.loc[plot_df.index, 'consumption_MWh']
                plot_df['Actual_Price'] = df.loc[plot_df.index, 'USD/MWh']

                st.markdown("### ⚡ 1. Energy Consumption Forecast Analysis")

                cp1, cp2 = st.columns([1, 2])
                with cp1:
                    show_actual_cons = st.toggle("🔍 Show Actual Data", key="toggle_cons_23")

                with cp2:
                    r2_cons = r2_score(plot_df['Actual_Cons'], plot_df['prediction_cons'])
                    st.markdown(f"""
                        <div style="background-color: #f0fdf4; padding: 10px; border-radius: 10px; border-left: 5px solid #22c55e; font-family: sans-serif;">
                            <small style="color: #166534;">Consumption Model R² Score</small><br>
                            <b style="color: #166534; font-size: 18px;">{r2_cons * 100:.2f}% Accuracy</b>
                        </div>
                    """, unsafe_allow_html=True)

                fig_cons = go.Figure()
                fig_cons.add_trace(go.Scatter(x=plot_df['datetime'], y=plot_df['prediction_cons'],
                                              name='AI Forecast', line=dict(color='#3b82f6', width=2.5)))
                if show_actual_cons:
                    fig_cons.add_trace(go.Scatter(x=plot_df['datetime'], y=plot_df['Actual_Cons'],
                                                  name='Actual',
                                                  line=dict(color='#0f172a', width=1.5, dash='dot')))
                fig_cons.update_layout(template="plotly_white", hovermode="x unified", height=400)
                st.plotly_chart(fig_cons, use_container_width=True)

                st.divider()
                st.markdown("### 💵 2. Electricity Price Forecast Analysis")

                fp1, fp2 = st.columns([1, 2])
                with fp1:
                    show_actual_price = st.toggle("🔍 Show Actual Price", key="toggle_price_23")

                with fp2:
                    r2_price = r2_score(plot_df['Actual_Price'], plot_df['prediction_price'])
                    st.markdown(f"""
                        <div style="background-color: #fffbeb; padding: 10px; border-radius: 10px; border-left: 5px solid #f59e0b; font-family: sans-serif;">
                            <small style="color: #92400e;">Price Model R² Score</small><br>
                            <b style="color: #92400e; font-size: 18px;">{r2_price * 100:.2f}% Accuracy</b>
                        </div>
                    """, unsafe_allow_html=True)

                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=plot_df['datetime'], y=plot_df['prediction_price'],
                                               name='Price Forecast', fill='tozeroy',
                                               line=dict(color='#fbbf24', width=2.5)))
                if show_actual_price:
                    fig_price.add_trace(go.Scatter(x=plot_df['datetime'], y=plot_df['Actual_Price'],
                                                   name='Actual Price',
                                                   line=dict(color='#b91c1c', width=1.5, dash='dash')))
                fig_price.update_layout(template="plotly_white", hovermode="x unified", height=400)
                st.plotly_chart(fig_price, use_container_width=True)


            else:
                st.warning("Please select a month!")