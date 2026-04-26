import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# page configuration
st.set_page_config(
    page_title="PMJDY Financial Inclusion Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

#some css work
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMarkdownContainer"] .info-box,
    .info-box {
        background-color: #f0f4ff !important;
        border-left: 4px solid #4a6fa5 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 4px !important;
        margin: 0.5rem 0 !important;
        font-size: 0.92rem !important;
        color: #1a1a1a !important;
    }
    div[data-testid="stMarkdownContainer"] .info-box *,
    .info-box * {
        color: #1a1a1a !important;
    }
    div[data-testid="stMarkdownContainer"] .warn-box,
    .warn-box {
        background-color: #fff8e1 !important;
        border-left: 4px solid #f9a825 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 4px !important;
        margin: 0.5rem 0 !important;
        font-size: 0.92rem !important;
        color: #1a1a1a !important;
    }
    div[data-testid="stMarkdownContainer"] .warn-box *,
    .warn-box * {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

#image helper
CHART_DIR = "results/visualisations"

def show_chart(filename, caption=None):
    path = os.path.join(CHART_DIR, filename)
    if not os.path.exists(path):
        st.error(
            f"Chart not found: {path}\n\n"
        )
        return
    st.image(path, caption=caption, use_container_width=True)

# data loading
@st.cache_data
def load_data():
    
    path = r"C:\Users\Mihika\Desktop\dsm_project_final\dsm project\data\cleaned\final_dataset.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else: 
        st.error(
        "Dataset not found. Add your final_dataset.csv to the data/cleaned/ folder in your repo and redeploy."
        )
        st.stop()


df = load_data()

if "Year" in df.columns:
    df_state = df.sort_values("Year", ascending=False).drop_duplicates("state_name").copy()
else:
    df_state = df.drop_duplicates("state_name").copy()

# Fix Delhi negative NFHS anomaly
if "women_bank_account_pct" in df_state.columns:
    df_state["women_bank_account_pct"] = df_state["women_bank_account_pct"].apply(
        lambda x: np.nan if pd.notna(x) and x < 0 else x
    )


M1_INTERCEPT = 0.676   # = mean of rupay_activation_ratio across 36 states

# Standardised coefficients (from Cell 8 output)
M1_COEFS = {
    "literacy_rate":         -0.0303,
    "electricity_access_pct": 0.0254,
    "women_bank_account_pct":-0.0249,
    "female_workforce_pct":  -0.0096,
    "rural_share":            0.0078,
    "sc_share":               0.0021,
}

# Exact means and stds from df_state.describe() on your final_dataset.csv
M1_STATS = {
    #                          mean     std
    "literacy_rate":          (67.29,   8.54),
    "electricity_access_pct": (98.44,   1.68),
    "women_bank_account_pct": (80.49,   6.61),
    "female_workforce_pct":   (26.90,   9.67),
    "rural_share":            (0.74,    0.19),
    "sc_share":               (11.85,   8.67),
}

# --- Model 2: Predicting avg_balance_per_account ---
# Features: rural_share, female_workforce_pct, literacy_rate,
#           electricity_access_pct, women_property_ownership_pct,
#           women_bank_account_pct, clean_water_pct
# Intercept = mean(avg_balance) = 6335

M2_INTERCEPT = 6335.0

M2_COEFS = {
    "rural_share":                 1579.887,
    "female_workforce_pct":       -1171.875,
    "literacy_rate":               1154.198,
    "electricity_access_pct":       831.359,
    "women_property_ownership_pct": 527.008,
    "women_bank_account_pct":      -485.937,
    "clean_water_pct":               -4.899,
}

M2_STATS = {
    "rural_share":                 (0.74,    0.19),
    "female_workforce_pct":        (26.90,   9.67),
    "literacy_rate":               (67.29,   8.54),
    "electricity_access_pct":      (98.44,   1.68),
    "women_property_ownership_pct":(40.77,  18.03),
    "women_bank_account_pct":      (80.49,   6.61),
    "clean_water_pct":             (94.99,   4.45),
}


def predict_model1(literacy, elec, women_bank, workforce, rural_share, sc_share):
    """Predict rupay_activation_ratio using exact OLS coefficients."""
    inputs = {
        "literacy_rate":         literacy,
        "electricity_access_pct": elec,
        "women_bank_account_pct": women_bank,
        "female_workforce_pct":   workforce,
        "rural_share":            rural_share / 100,   # slider is 0-100, model uses 0-1
        "sc_share":               sc_share,
    }
    pred = M1_INTERCEPT
    for feat, val in inputs.items():
        mean, std = M1_STATS[feat]
        z = (val - mean) / std
        pred += M1_COEFS[feat] * z
    return float(np.clip(pred, 0.42, 0.85))


def predict_model2(rural_share, workforce, literacy, elec, property_own, women_bank, clean_water):
    """Predict avg_balance_per_account using exact OLS coefficients."""
    inputs = {
        "rural_share":                 rural_share / 100,
        "female_workforce_pct":        workforce,
        "literacy_rate":               literacy,
        "electricity_access_pct":      elec,
        "women_property_ownership_pct":property_own,
        "women_bank_account_pct":      women_bank,
        "clean_water_pct":             clean_water,
    }
    pred = M2_INTERCEPT
    for feat, val in inputs.items():
        mean, std = M2_STATS[feat]
        z = (val - mean) / std
        pred += M2_COEFS[feat] * z
    return float(np.clip(pred, 1000, 25000))


# sidebar
st.sidebar.title("PMJDY Dashboard")
st.sidebar.caption("Financial Inclusion Quality Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA Explorer", "Predictive Model", "Cluster Analysis", "Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: PMJDY portal, NFHS-5, Census 2011, RBI Handbook")

# PAGE 1 — OVERVIEW

if page == "Overview":
    st.title("Analysing Financial Inclusion Quality Under PMJDY")
    st.markdown(
        "India opened 57.86 crore bank accounts, but how many people actually use them?"
    )
    st.markdown("---")

    mean_ratio = df_state["rupay_activation_ratio"].mean()
    mean_bal   = df_state["avg_balance_per_account"].mean()
    rural_avg  = df_state["rural_share"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total PMJDY Accounts", "57.86 Cr")
    col2.metric(
        "Mean RuPay Activation",
        f"{mean_ratio:.1%}",
        delta=f"-{1 - mean_ratio:.1%} likely dormant"
    )
    col3.metric("Avg Balance per Account", f"Rs {mean_bal:,.0f}")
    col4.metric("Rural Account Share", f"{rural_avg:.1%}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Distribution of key variables across states")
        st.caption(
            "Histograms showing the spread of RuPay activation ratio, average balance, "
            "and other key variables across 36 states and UTs."
        )
        show_chart("01_distributions.png")

    with col_right:
        st.subheader("What this project measures")
        st.markdown("""
We distinguish between two types of inclusion:

- **Real inclusion**: Accounts with active transactions, positive balances, RuPay cards in use
- **Nominal inclusion**: Accounts that exist on paper but have never been meaningfully used

A bank account that nobody uses is not financial inclusion.
        """)

        st.markdown("""<div class="info-box">
<strong>Research questions</strong><br>
1. What share of accounts are actively used, and how does this vary by state?<br>
2. What socioeconomic factors predict genuine vs nominal inclusion?<br>
3. Is there a measurable gender gap in account activity?
</div>""", unsafe_allow_html=True)

        st.subheader("Datasets used")
        st.dataframe(
            pd.DataFrame({
                "Dataset":      ["PMJDY Statewise", "Census 2011", "NFHS-5", "RBI Handbook"],
                "Coverage":     ["36 states/UTs", "State-level agg.", "36 states", "36 states"],
                "Key variable": ["RuPay ratio", "Literacy rate", "Women bank %", "Branches/lakh"]
            }),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Key numbers from the data")
        st.markdown(f"""
| Metric | Value |
|---|---|
| Mean RuPay activation ratio | {mean_ratio:.3f} |
| Min activation (state) | {df_state['rupay_activation_ratio'].min():.3f} |
| Max activation (state) | {df_state['rupay_activation_ratio'].max():.3f} |
| Mean avg balance | Rs {mean_bal:,.0f} |
| Rural share (national avg) | {rural_avg:.1%} |
        """)

# PAGE 2 — EDA EXPLORER

elif page == "EDA Explorer":
    st.title("Exploratory Data Analysis")
    st.markdown("Patterns in financial inclusion across 36 states and union territories.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distributions and Correlations", "Scatter Analysis", "Gender Analysis"])

    with tab1:
        st.subheader("Variable distributions")
        st.caption(
            "Histograms of key variables: RuPay activation ratio, average balance, "
            "literacy rate, rural share, and related socioeconomic indicators."
        )
        show_chart("01_distributions.png")

        st.markdown("---")
        st.subheader("Correlation heatmap")
        st.caption(
            "Pairwise correlations between all key variables. "
            "Note the negative correlation between literacy and RuPay activation — explained below."
        )
        show_chart("02_correlation_heatmap.png")

        st.markdown("""<div class="info-box">
<strong>Why is literacy negatively correlated with RuPay activation?</strong><br>
More literate states (Kerala, Himachal Pradesh, Goa) already had strong formal banking
before PMJDY launched. PMJDY accounts there are secondary accounts households rarely use —
not because literacy is bad for inclusion, but because these states did not need PMJDY
as their primary banking vehicle. The same logic explains the mobile phone correlation.
</div>""", unsafe_allow_html=True)

    with tab2:
        st.subheader("Scatter plots: socioeconomic predictors vs inclusion quality")
        st.caption(
            "Each panel shows a predictor variable plotted against RuPay activation ratio "
            "or average balance per account, with state labels and regression lines."
        )
        show_chart("03_scatter_plots.png")

    with tab3:
        st.subheader("Women's bank account usage by state (NFHS-5)")
        st.caption(
            "Percentage of women who report having a bank or savings account they themselves use. "
            "Red = below national median, green = above."
        )
        show_chart("08_gender_account_usage.png")

        st.markdown("---")
        st.subheader("Literacy gender gap vs RuPay activation")
        st.caption(
            "States with larger male-female literacy gaps tend to show lower RuPay activation ratios."
        )
        show_chart("09_gender_gap_scatter.png")

        st.markdown("---")
        st.subheader("Change in women's bank account usage: NFHS-4 to NFHS-5")
        show_chart("nfhs_change_2.png")

        st.markdown("""<div class="info-box">
<strong>Key finding</strong><br>
Average improvement: +20.46 percentage points. The biggest improvers — Bihar (+51.3%),
Manipur (+39.9%), Madhya Pradesh (~+38%) — are all states with heavy DBT transfer activity.
Government transfers appear to be the primary activation mechanism for women's accounts,
not voluntary saving behaviour.
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Multi-indicator NFHS change heatmap")
        st.caption(
            "While bank account usage improved sharply, female workforce participation "
            "and women's property ownership improved by only ~2% over the same period."
        )
        show_chart("nfhs_heatmap_2.png")


# PAGE 3 — PREDICTIVE MODEL

elif page == "Predictive Model":
    st.title("Predictive Model")
    st.markdown(
        "Two OLS regression models and a Random Forest trained on 36 state-level observations."
    )
    st.markdown("---")

    tab1, tab2 = st.tabs(["Model Results", "Prediction Simulator"])

    # Tab 1: Model Results
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Regression coefficients (both models)")
            st.caption(
                "Standardised OLS coefficients. "
                "Model 1 predicts RuPay activation ratio. "
                "Model 2 predicts average balance per account."
            )
            show_chart("04_regression_coefficients.png")

        with col2:
            st.subheader("Random Forest feature importances")
            st.caption(
                "Relative importance scores from a 500-tree Random Forest predicting "
                "RuPay activation ratio. Uses the full 11-feature set."
            )
            show_chart("05_random_forest_importance.png")

        st.markdown("---")
        st.subheader("Model performance summary")
        st.dataframe(
            pd.DataFrame({
                "Model":         ["OLS — Model 1 (RuPay activation)", "OLS — Model 2 (Avg balance)", "Random Forest"],
                "Features":      ["literacy, electricity, women bank %, workforce, rural share, SC share",
                                  "rural share, workforce, literacy, electricity, property own, women bank %, clean water",
                                  "11 socioeconomic features"],
                "In-sample R2":  [0.301, 0.418, 0.852],
                "CV R2":         [-0.413, -4.022, -0.222]
            }),
            hide_index=True,
            use_container_width=True
        )

        st.markdown("""<div class="warn-box">
<strong>Overfitting caveat</strong><br>
With only 36 state-level observations, all models overfit badly. The negative cross-validated
R2 scores confirm this. These results are descriptive associations — they tell us which
variables are correlated with inclusion quality in the current data, not what causes it.
District-level PMJDY data (~700 districts) would enable more robust modelling.
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Anomaly detection: over and underperforming states")
        st.caption(
            "Ridge regression residuals (LOO-CV) showing which states achieve better or worse "
            "inclusion quality than their socioeconomic profile predicts."
        )
        show_chart("anomaly_detection.png")

        st.markdown("""<div class="info-box">
<strong>Overperformers</strong> (better than predicted): Lakshadweep +18.6%, Nagaland +14.4%,
Gujarat +11.4%, Tamil Nadu +11.0%<br><br>
<strong>Underperformers</strong> (worse than predicted): Mizoram, Kerala, Assam —
structural conditions are better than outcomes suggest, pointing to implementation gaps.
</div>""", unsafe_allow_html=True)

    # Tab 2: Prediction Simulator 
    with tab2:
        st.subheader("Simulate predicted outcomes for a hypothetical state")
        st.markdown(
            "Adjust the socioeconomic inputs below. "
            "The predictions use the exact standardised OLS coefficients from your models — "
            "not approximations."
        )

        st.markdown("---")

        #  Shared inputs (used by both models) 
        st.markdown("**Inputs used in both models**")
        col1, col2, col3 = st.columns(3)
        with col1:
            inp_literacy  = st.slider("Literacy rate (%)", 40, 100, 67,
                                      help="Census 2011 state literacy rate. Range: 50–84. National mean: 67.3%")
        with col2:
            inp_elec      = st.slider("Electricity access (%)", 90, 100, 98,
                                      help="NFHS-5 households with electricity. Range: 93–100. National mean: 98.4%")
        with col3:
            inp_women_bank = st.slider("Women bank account % (NFHS-5)", 60, 100, 80,
                                       help="% women who use their own bank account. Range: 67–94. National mean: 80.5%")

        col4, col5, col6 = st.columns(3)
        with col4:
            inp_workforce  = st.slider("Female workforce participation (%)", 5, 50, 27,
                                       help="Census: % women who worked in past year. Range: 10–45. National mean: 26.9%")
        with col5:
            inp_rural      = st.slider("Rural account share (%)", 5, 100, 74,
                                       help="% of PMJDY accounts at rural/semi-urban branches. Range: 9–98. National mean: 74%")
        with col6:
            inp_sc         = st.slider("SC population share (%)", 0, 35, 12,
                                       help="Scheduled Caste share of state population. Range: 0–32. National mean: 11.9%")

        # Model 2 only inputs 
        st.markdown("**Additional inputs for Model 2 (avg balance prediction only)**")
        col7, col8 = st.columns(2)
        with col7:
            inp_property   = st.slider("Women property ownership (%)", 0, 75, 41,
                                       help="NFHS-5: % women owning house/land. Range: -2–71. National mean: 40.8%")
        with col8:
            inp_water      = st.slider("Clean water access (%)", 75, 100, 95,
                                       help="NFHS-5: % households with clean water. Range: 80–100. National mean: 95%")

        #  Run predictions 
        pred_ratio   = predict_model1(inp_literacy, inp_elec, inp_women_bank,
                                      inp_workforce, inp_rural, inp_sc)
        pred_balance = predict_model2(inp_rural, inp_workforce, inp_literacy,
                                      inp_elec, inp_property, inp_women_bank, inp_water)

        nat_ratio   = df_state["rupay_activation_ratio"].mean()
        nat_balance = df_state["avg_balance_per_account"].mean()

        st.markdown("---")
        st.subheader("Predicted outcomes")

        col_g1, col_g2, col_interp = st.columns([1, 1, 2])

        # Gauge 1: RuPay activation ratio
        with col_g1:
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_ratio,
                number={"valueformat": ".3f"},
                title={"text": "Model 1<br>RuPay Activation Ratio", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0.4, 0.9], "tickformat": ".2f"},
                    "bar":  {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0.40, 0.58], "color": "#fde8e8"},
                        {"range": [0.58, 0.68], "color": "#fef9e7"},
                        {"range": [0.68, 0.90], "color": "#e8f5e9"},
                    ],
                    "threshold": {"line": {"color": "grey", "width": 2}, "value": nat_ratio}
                }
            ))
            fig1.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig1, use_container_width=True)
            st.caption(f"Grey line = national avg ({nat_ratio:.3f})")

        # Gauge 2: Average balance
        with col_g2:
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_balance,
                number={"valueformat": ",.0f", "prefix": "Rs "},
                title={"text": "Model 2<br>Avg Balance / Account", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [1000, 20000]},
                    "bar":  {"color": "#2ca02c"},
                    "steps": [
                        {"range": [1000,  4000], "color": "#fde8e8"},
                        {"range": [4000,  7000], "color": "#fef9e7"},
                        {"range": [7000, 20000], "color": "#e8f5e9"},
                    ],
                    "threshold": {"line": {"color": "grey", "width": 2}, "value": nat_balance}
                }
            ))
            fig2.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"Grey line = national avg (Rs {nat_balance:,.0f})")

        # Interpretation
        with col_interp:
            # Tier interpretation for Model 1
            if pred_ratio >= 0.72:
                tier1 = "Above average"
                msg1  = "Strong account activation likely. High electricity access and positive rural deployment are the main contributors."
            elif pred_ratio >= 0.64:
                tier1 = "Moderate"
                msg1  = "Near national average. Targeted DBT routing could push activation higher without requiring structural change."
            else:
                tier1 = "Poor inclusion"
                msg1  = "Nominal inclusion likely. Accounts may have been opened but are not being used. Infrastructure investment and BC deployment are the priority."

            # Tier interpretation for Model 2
            if pred_balance >= 7000:
                tier2 = "High"
                msg2  = "Higher-than-average balances suggest accounts are being used for saving or receiving regular transfers."
            elif pred_balance >= 4500:
                tier2 = "Moderate"
                msg2  = "Near national average. Balance growth likely depends on routing more DBT payments through these accounts."
            else:
                tier2 = "Low"
                msg2  = "Low average balance. Accounts are likely opened but not used for saving. Rural share and female workforce participation are the main drag factors."

            st.markdown(f"**Model 1 — RuPay activation: {tier1}**")
            st.markdown(f"Predicted ratio: **{pred_ratio:.3f}** vs national avg {nat_ratio:.3f}")
            st.markdown(f'<div class="info-box">{msg1}</div>', unsafe_allow_html=True)

            st.markdown(f"**Model 2 — Avg balance: {tier2}**")
            st.markdown(f"Predicted balance: **Rs {pred_balance:,.0f}** vs national avg Rs {nat_balance:,.0f}")
            st.markdown(f'<div class="info-box">{msg2}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Coefficient breakdown table — shows contribution of each input
        st.subheader("How each input is driving the Model 1 prediction")
        st.caption("Shows the contribution of each variable to the predicted RuPay activation ratio.")

        inputs_m1 = {
            "literacy_rate":         inp_literacy,
            "electricity_access_pct": inp_elec,
            "women_bank_account_pct": inp_women_bank,
            "female_workforce_pct":   inp_workforce,
            "rural_share":            inp_rural / 100,
            "sc_share":               inp_sc,
        }
        contributions = []
        for feat, val in inputs_m1.items():
            mean, std = M1_STATS[feat]
            z = (val - mean) / std
            contrib = M1_COEFS[feat] * z
            contributions.append({
                "Feature":      feat.replace("_", " ").title(),
                "Your input":   f"{val:.2f}" if feat != "rural_share" else f"{val*100:.1f}%",
                "National mean":f"{mean:.2f}" if feat != "rural_share" else f"{mean*100:.1f}%",
                "z-score":      round(z, 2),
                "Contribution": round(contrib, 4),
                "Direction":    "positive" if contrib >= 0 else "negative"
            })

        contrib_df = pd.DataFrame(contributions).sort_values("Contribution", key=abs, ascending=False)
        st.dataframe(contrib_df, hide_index=True, use_container_width=True)



# PAGE 4 — CLUSTER ANALYSIS

elif page == "Cluster Analysis":
    st.title("State Clustering: Inclusion Tiers")
    st.markdown(
        "K-Means clustering (K=6) groups 36 states into tiers of financial inclusion quality "
        "using RuPay activation ratio, average balance, accounts per capita, women's bank usage, "
        "literacy rate, and rural share."
    )
    st.markdown("---")

    tab1, tab2 = st.tabs(["Cluster Results", "Method"])

    with tab1:
        st.subheader("K-Means PCA plot: states by inclusion tier")
        st.caption(
            "PCA reduces the six clustering features to two dimensions for visualisation. "
            "Each point is a state, coloured by its assigned inclusion tier."
        )
        show_chart("07_kmeans_pca.png")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tier descriptions")
            st.markdown("""
| Tier | Character | Example states |
|---|---|---|
| Tier 1 — High Inclusion | Highest activation and balance | Lakshadweep |
| Tier 2 — Above Average | Strong all-round | Gujarat, Haryana, Nagaland, Maharashtra |
| Tier 3 — Moderate | Near national average | 5 mid-range states |
| Tier 4 — Below Average | Large states with genuine gaps | UP, Bihar, Rajasthan |
| Tier 5 — Poor Inclusion | Infrastructure barriers | Assam, Manipur |
| Tier 6 — Nominal Inclusion | Redundancy problem | Kerala, Mizoram, Tripura |
            """)

        with col2:
            st.markdown("""<div class="info-box">
<strong>The Kerala-Mizoram paradox</strong><br>
Both states have high literacy but land in the lowest activation tier. They had strong
formal banking penetration before PMJDY launched. PMJDY accounts are effectively
duplicate accounts that households never use — this is a product-market fit problem,
not a literacy or infrastructure problem.
</div>""", unsafe_allow_html=True)

            st.markdown("""<div class="info-box">
<strong>Overperformers vs underperformers</strong><br>
Lakshadweep, Nagaland, Gujarat and Tamil Nadu achieve better inclusion than their
socioeconomic profile predicts. Mizoram, Kerala and Assam achieve worse — pointing
to implementation gaps that policy can address without waiting for structural development.
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Anomaly detection: residuals from predicted activation")
        show_chart("anomaly_detection.png")

    with tab2:
        st.subheader("Choosing the number of clusters")
        st.caption(
            "Both the elbow method (inertia) and silhouette score analysis pointed to K=6 "
            "as the optimal number of clusters."
        )
        show_chart("06_elbow_silhouette.png")

        st.markdown("---")
        st.subheader("Hierarchical clustering dendrogram")
        st.caption(
            "Ward linkage dendrogram as a cross-check. Broad groupings are consistent "
            "with the K-Means result. Adjusted Rand Index > 0.5 for K=4 and K=5."
        )
        show_chart("07b_dendrogram.png")

        st.markdown("""<div class="info-box">
<strong>Why K=6?</strong><br>
The silhouette score peaks at K=6, and the hierarchical dendrogram shows six natural
break points. The Adjusted Rand Index between K-Means and hierarchical results confirms
reasonable agreement between the two methods.
</div>""", unsafe_allow_html=True)



# PAGE 5 — RECOMMENDATIONS

elif page == "Recommendations":
    st.title("Policy Recommendations")
    st.markdown(
        "Evidence-based interventions derived from the clustering, regression, and gender analyses."
    )
    st.markdown("---")

    recommendations = [
        {
            "title": "1. Differentiate policy by inclusion tier",
            "body": (
                "A single national policy approach does not work. Tier 6 states (Kerala, Mizoram, Tripura) "
                "have a redundancy problem — PMJDY accounts are duplicates of existing accounts. "
                "Policy here should focus on converting dormant PMJDY accounts into active primary "
                "accounts via DBT routing, not opening new ones. "
                "Tier 4 and 5 states (Bihar, Uttar Pradesh, Assam, Manipur) have genuine activation "
                "gaps driven by low awareness and weak infrastructure. Here the priority is "
                "banking correspondent deployment and mobile banking literacy camps."
            ),
            "target": "Tier 6: Kerala, Mizoram, Tripura | Tier 4/5: Bihar, UP, Assam, Manipur"
        },
        {
            "title": "2. Treat mobile phone access as infrastructure",
            "body": (
                "Women's mobile phone ownership is the single strongest predictor of account activation "
                "across all model types. Mobile phones are the gateway to UPI, balance checks, and "
                "transaction alerts that make accounts genuinely useful. Policy should bundle PMJDY "
                "account opening with subsidised mobile access for women — similar to Rajasthan's "
                "Indira Gandhi Smartphone Yojana. This will do more than awareness campaigns alone."
            ),
            "target": "States in Tiers 4 and 5 where women's mobile penetration is below 50%"
        },
        {
            "title": "3. Use DBT transfers as an activation mechanism",
            "body": (
                "The NFHS-4 to NFHS-5 comparison shows the largest improvements in women's bank "
                "account usage occurred in states with heavy DBT activity: Bihar (+51.3%), "
                "Manipur (+39.9%), Madhya Pradesh (~+38%). Mandating that all G2P transfers — "
                "MGNREGS wages, PM-KISAN instalments, scholarship disbursements — flow through "
                "PMJDY accounts in Tier 4, 5 and 6 states would have an immediate impact."
            ),
            "target": "Bihar, Uttar Pradesh, Madhya Pradesh, Rajasthan, Assam"
        },
        {
            "title": "4. Address the gender gap with structural interventions",
            "body": (
                "The literacy gender gap is a significant predictor of lower women's inclusion quality. "
                "Financial literacy without general literacy has limited impact. "
                "States with large gender literacy gaps need adult literacy programmes running "
                "alongside financial BC training. Interventions that assume basic reading ability "
                "will not reach the households most excluded from the formal economy."
            ),
            "target": "States where the literacy gender gap exceeds 15 percentage points"
        },
        {
            "title": "5. Audit anomalous underperformers",
            "body": (
                "Mizoram, Kerala and Assam all underperform relative to their socioeconomic conditions, "
                "suggesting implementation gaps rather than structural barriers. "
                "For Kerala: convert dormant accounts to primary accounts via DBT. "
                "For Assam and Mizoram: a district-level audit of banking correspondent density "
                "and geographic coverage is the recommended first step."
            ),
            "target": "Mizoram, Kerala, Assam"
        },
    ]

    for rec in recommendations:
        with st.expander(rec["title"], expanded=True):
            st.markdown(rec["body"])
            st.caption(f"Target states / regions: {rec['target']}")

    st.markdown("---")
    st.subheader("Limitations and future work")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Current limitations**
- Census data is from 2011 — misses 15 years of electrification and urbanisation gains
- No explicit zero-balance account counts in PMJDY data, so RuPay activation is an imperfect dormancy proxy
- Regression models overfit to 36 observations — not suitable for causal inference
- Delhi NFHS anomaly (negative women's bank percentage from case-mismatch) was excluded rather than corrected
        """)
    with col2:
        st.markdown("""
**Future work**
- District-level PMJDY data would increase sample size by ~10x and enable robust causal analysis
- A difference-in-differences design using NFHS-4 vs NFHS-5 could isolate the effect of DBT routing
- A qualitative survey in Tier 5 and 6 states would identify barriers that quantitative data cannot show
        """)