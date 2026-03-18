import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

SIMULATIONS = 10000

# Default tax/plan constants (overridden by UI below)
STANDARD_DEDUCTION = 30000
BRACKETS = [
 (23850, 0.10),
 (96950, 0.12),
 (206700, 0.22),
 (394600, 0.24),
 (501050, 0.32),
 (751600, 0.35),
 (float("inf"), 0.37),
]
SOCIAL_SECURITY_RATE = 0.062
SOCIAL_SECURITY_WAGE_BASE = 168600
MEDICARE_RATE = 0.0145
ADDITIONAL_MEDICARE_RATE = 0.009
ADDITIONAL_MEDICARE_THRESHOLD = 250000
K401_LIMIT = 24500
HSA_LIMIT = 4500

# -----------------------------
# TAX CALCULATIONS
# -----------------------------

def federal_tax(taxable_income):

    tax = 0
    prev_limit = 0

    for limit, rate in BRACKETS:

        if taxable_income > limit:
            tax += (limit - prev_limit) * rate
            prev_limit = limit
        else:
            tax += (taxable_income - prev_limit) * rate
            break

    return max(tax, 0)


def tax_liability(income):
    taxable = max(0, income - STANDARD_DEDUCTION)
    return federal_tax(taxable)


def shapley_tax(incomes):

    a, b = incomes

    tax_a_first = tax_liability(a) - tax_liability(0)
    tax_a_second = tax_liability(a + b) - tax_liability(b)

    tax_b_first = tax_liability(b) - tax_liability(0)
    tax_b_second = tax_liability(a + b) - tax_liability(a)

    shapley_a = 0.5 * (tax_a_first + tax_a_second)
    shapley_b = 0.5 * (tax_b_first + tax_b_second)

    return [shapley_a, shapley_b]

# -----------------------------
# FICA
# -----------------------------

def fica_tax(wages):

    ss_tax = min(wages, SOCIAL_SECURITY_WAGE_BASE) * SOCIAL_SECURITY_RATE
    medicare_tax = wages * MEDICARE_RATE

    additional_medicare = max(
        0,
        wages - ADDITIONAL_MEDICARE_THRESHOLD
    ) * ADDITIONAL_MEDICARE_RATE

    return ss_tax + medicare_tax + additional_medicare


# -----------------------------
# AFTER TAX INCOME
# -----------------------------

def personal_after_tax_income(salary, partner_income, k401, hsa):

    pretax = k401 + hsa
    incomes = [salary - pretax, partner_income]
    tax_split = shapley_tax(incomes)
    my_federal_tax = tax_split[0]
    payroll_tax = fica_tax(salary)

    return salary - pretax - my_federal_tax - payroll_tax

# -----------------------------
# SALARY PATH
# -----------------------------

def generate_salary_path(start_salary, years, salary_growth):

    return np.array([
        start_salary * (1 + salary_growth) ** y
        for y in range(years)
    ])

# -----------------------------
# PRETAX CONTRIBUTION SOLVER
# -----------------------------

def max_pretax_contribution(salary, partner_income, spending):

    def objective(pretax):
        net_income = personal_after_tax_income(salary, partner_income, pretax, 0)
        return net_income - spending

    try:
        pretax = brentq(objective, 0, K401_LIMIT + HSA_LIMIT)
    except ValueError:
        pretax = K401_LIMIT + HSA_LIMIT

    return pretax

# -----------------------------
# CONTRIBUTIONS
# -----------------------------

def compute_contributions(salaries, partner_income, spending, match_rate):

    years = len(salaries)
    contributions = np.zeros(years)

    for y in range(years):

        salary = salaries[y]
        pretax = max_pretax_contribution(salary, partner_income, spending)
        k401 = min(pretax, K401_LIMIT)
        hsa = min(max(0, pretax - k401), HSA_LIMIT)
        net_income = personal_after_tax_income(salary, partner_income, k401, hsa)
        match = salary * match_rate
        excess = net_income - spending

        if excess < 0:
            return None

        contributions[y] = pretax + match + excess

    return contributions

# -----------------------------
# RETURN PATHS
# -----------------------------

def generate_return_paths(years, mean_return, return_std, phi=-0.15):

    log_mean = np.log(1 + mean_return) - 0.5 * return_std ** 2
    noise_std = return_std * np.sqrt(1 - phi ** 2)
    log_returns = np.zeros((SIMULATIONS, years))
    log_returns[:, 0] = np.random.normal(log_mean, return_std, SIMULATIONS)

    for t in range(1, years):
        noise = np.random.normal(0, noise_std, SIMULATIONS)
        log_returns[:, t] = log_mean + phi * (log_returns[:, t-1] - log_mean) + noise

    return np.exp(log_returns)

# -----------------------------
# FAST FINAL PORTFOLIOS
# -----------------------------

def compute_final_portfolios(current_portfolio, contributions, growth):

    cum_growth = np.cumprod(growth, axis=1)
    start_component = current_portfolio * cum_growth[:, -1]

    reverse_growth = np.cumprod(growth[:, ::-1], axis=1)[:, ::-1]
    future_growth = np.ones_like(growth)
    future_growth[:, :-1] = reverse_growth[:, 1:]

    contribution_component = (contributions * future_growth).sum(axis=1)

    return start_component + contribution_component

# -----------------------------
# FULL PATHS (for charts)
# -----------------------------

def simulate_wealth_paths(current_portfolio, contributions, growth):

    sims, years = growth.shape
    wealth = np.zeros((sims, years + 1))
    wealth[:, 0] = current_portfolio

    for y in range(years):
        wealth[:, y+1] = wealth[:, y] * growth[:, y] + contributions[y]

    return wealth

# -----------------------------
# SALARY MODEL
# -----------------------------

def simulate_salary_model(
    start_salary, partner_income, years, spending,
    current_portfolio, target_portfolio, growth,
    salary_growth, match_rate, inheritance, inheritance_year
):

    salaries = generate_salary_path(start_salary, years, salary_growth)
    contributions = compute_contributions(salaries, partner_income, spending, match_rate)

    if contributions is None:
        return None, 0, None, None, None

    if inheritance_year is not None:
        contributions[inheritance_year - 1] += inheritance

    portfolios = compute_final_portfolios(current_portfolio, contributions, growth)
    wealth_paths = simulate_wealth_paths(current_portfolio, contributions, growth)
    success_rate = np.mean(portfolios >= target_portfolio)

    contributions[inheritance_year - 1] -= inheritance

    return portfolios, success_rate, salaries, contributions, wealth_paths

# -----------------------------
# SALARY SOLVER
# -----------------------------

def solve_required_salary(
    partner_income, years, spending, current_portfolio, target_portfolio,
    mean_return, return_std, salary_growth, match_rate, target_success,
    inheritance, inheritance_year
):

    np.random.seed(42)
    growth = generate_return_paths(years, mean_return, return_std)

    def objective(salary):
        _, success, _, _, _ = simulate_salary_model(
            salary, partner_income, years, spending, current_portfolio,
            target_portfolio, growth, salary_growth, match_rate,
            inheritance, inheritance_year
        )
        return success - target_success

    salary = brentq(objective, 50000, 600000)
    return salary, growth

# -----------------------------
# STREAMLIT UI
# -----------------------------

# -----------------------------
# SIDEBAR — all inputs
# -----------------------------

with st.sidebar:

    st.title("Inputs")

    # Household
    st.subheader("Household")
    partner_income    = st.number_input("Partner Income ($)",    0, 1_000_000,  300_000, step=5_000)
    spending          = st.number_input("Annual Spending ($)",   0,   500_000,   50_000, step=1_000)
    current_portfolio = st.number_input("Current Portfolio ($)", 0, 5_000_000,  250_000, step=10_000)

    st.divider()

    # Goal
    st.subheader("Goal")
    target_portfolio = st.number_input("Target Portfolio ($)", 0, 20_000_000, 4_000_000, step=100_000)
    gc1, gc2 = st.columns(2)
    with gc1:
        years = st.slider("Years", 5, 40, 25)
    with gc2:
        target_success = st.slider("Success Rate", 50, 99, 70, format="%d%%") / 100

    st.divider()

    # Inheritance
    st.subheader("Inheritance")
    ic1, ic2 = st.columns(2)
    with ic1:
        inheritance = st.number_input("Amount ($)", 0, 10_000_000, 1_500_000, step=50_000)
    with ic2:
        inheritance_year = st.number_input("Year", 1, years, years)

    st.divider()

    # Market
    st.subheader("Market")
    mc1, mc2 = st.columns(2)
    with mc1:
        mean_return   = st.slider("Mean Return",   0.0, 15.0, 4.0, step=0.5, format="%.1f%%") / 100
        return_std    = st.slider("Volatility (σ)", 0,   30,  20,             format="%d%%")   / 100
    with mc2:
        salary_growth = st.slider("Salary Growth", 0.0, 10.0, 2.0, step=0.5, format="%.1f%%") / 100

    st.divider()

    # Tax & Plan Settings
    with st.expander("Tax & Plan Settings"):

        st.caption("Defaults reflect current US tax law.")

        st.markdown("**401(k) & HSA**")
        p1, p2, p3 = st.columns(3)
        with p1:
            K401_LIMIT = st.number_input("401(k) ($)", 0, 100_000, 24_500, step=500)
        with p2:
            HSA_LIMIT  = st.number_input("HSA ($)",    0,  20_000,  4_500, step=100)
        with p3:
            match_rate = st.slider("Match", 0, 8, 7, format="%d%%") / 100

        st.markdown("**Federal Tax**")
        STANDARD_DEDUCTION = st.number_input("Standard Deduction ($)", 0, 100_000, 30_000, step=500)

        st.markdown("**FICA**")
        f1, f2 = st.columns(2)
        with f1:
            SOCIAL_SECURITY_RATE     = st.number_input("SS Rate",        0.0, 0.20,    0.062,  step=0.001, format="%.3f")
            SOCIAL_SECURITY_WAGE_BASE = st.number_input("SS Wage Base ($)", 0, 500_000, 168_600, step=1_000)
        with f2:
            MEDICARE_RATE                = st.number_input("Medicare Rate",          0.0, 0.10, 0.0145, step=0.001,  format="%.4f")
            ADDITIONAL_MEDICARE_RATE     = st.number_input("Add'l Medicare Rate",    0.0, 0.10, 0.009,  step=0.001,  format="%.3f")
            ADDITIONAL_MEDICARE_THRESHOLD = st.number_input("Add'l Threshold ($)",   0, 1_000_000, 250_000, step=10_000)

        st.markdown("**Bracket Thresholds** *(rates fixed)*")
        bracket_defaults = [23_850, 96_950, 206_700, 394_600, 501_050, 751_600]
        bracket_rates    = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
        bracket_labels   = ["10%", "12%", "22%", "24%", "32%", "35%"]
        new_limits = []
        bl1, bl2 = st.columns(2)
        bcols = [bl1, bl2, bl1, bl2, bl1, bl2]
        for i, (lbl, lim) in enumerate(zip(bracket_labels, bracket_defaults)):
            with bcols[i]:
                new_limits.append(st.number_input(f"{lbl} top ($)", 0, 2_000_000, lim, step=500, key=f"bracket_{i}"))
        BRACKETS = list(zip(new_limits, bracket_rates[:-1])) + [(float("inf"), 0.37)]

    st.divider()

    run = st.button("▶  Solve Required Salary", use_container_width=True, type="primary")

# -----------------------------
# MAIN AREA
# -----------------------------

st.title("Retirement Salary Monte Carlo")
st.caption("Solve for the minimum starting salary required to reach a target portfolio with a given probability.")

if not run:
    st.info("Configure inputs in the sidebar, then click **Solve Required Salary**.")
    st.stop()

# -----------------------------
# SOLVE + RESULTS
# -----------------------------

with st.spinner("Running Monte Carlo simulation..."):
    required_salary, growth = solve_required_salary(
        partner_income, years, spending, current_portfolio, target_portfolio,
        mean_return, return_std, salary_growth, match_rate, target_success,
        inheritance, inheritance_year
    )
    portfolios, success, salaries, contributions, wealth_paths = simulate_salary_model(
        required_salary, partner_income, years, spending, current_portfolio,
        target_portfolio, growth, salary_growth, match_rate,
        inheritance, inheritance_year
    )

years_axis = list(range(years + 1))
contrib_x  = years_axis[1:]

st.subheader("Results")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Required Salary",       f"${required_salary:,.0f}")
with m2:
    st.metric("Success Rate",          f"{success:.1%}")
with m3:
    st.metric("Median Ending Portfolio", f"${np.median(portfolios):,.0f}")

st.divider()

# -----------------------------
# SALARY PATH
# -----------------------------

fig_salary = go.Figure()
fig_salary.add_trace(go.Bar(
    x=contrib_x, y=salaries, name="Salary",
))
fig_salary.update_layout(
    title="Salary Path",
    xaxis_title="Year", yaxis_title="Salary",
    xaxis=dict(dtick=1),
    yaxis=dict(tickprefix="$", tickformat=",.0f"),
    height=300
)
st.plotly_chart(fig_salary, use_container_width=True)

st.divider()

# -----------------------------
# HISTOGRAM + CONTRIBUTIONS
# -----------------------------

r1c1, r1c2 = st.columns(2, gap="medium")

with r1c1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=portfolios, nbinsx=50))
    fig.add_vline(x=target_portfolio, line_dash="dash", line_color="red",
                  annotation_text="Target", annotation_position="top right")
    fig.update_layout(
        title="Ending Portfolio Distribution",
        xaxis_title="Portfolio Value", yaxis_title="Count",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=contrib_x, y=contributions, name="Contributions"))
    fig4.update_layout(
        title="Annual Contributions",
        xaxis_title="Year", yaxis_title="Contribution",
        xaxis=dict(dtick=1),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380
    )
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# WEALTH PATHS + FI PROBABILITY
# -----------------------------

r2c1, r2c2 = st.columns(2, gap="medium")

with r2c1:
    if wealth_paths is None:
        wealth_paths = np.zeros((1, years + 1))

    p10 = np.percentile(wealth_paths, 10, axis=0)
    p50 = np.percentile(wealth_paths, 50, axis=0)
    p90 = np.percentile(wealth_paths, 90, axis=0)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=years_axis, y=p90, name="90th Percentile",
                              line=dict(dash="dot", width=1), opacity=0.6))
    fig2.add_trace(go.Scatter(x=years_axis, y=p50, name="Median",
                              line=dict(width=2.5)))
    fig2.add_trace(go.Scatter(x=years_axis, y=p10, name="10th Percentile",
                              line=dict(dash="dot", width=1), opacity=0.6))
    fig2.add_hline(y=target_portfolio, line_dash="dash", line_color="red",
                   opacity=0.5, annotation_text="Target", annotation_font_size=11)
    fig2.update_layout(
        title="Monte Carlo Wealth Paths",
        xaxis_title="Year", yaxis_title="Portfolio Value",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380
    )
    st.plotly_chart(fig2, use_container_width=True)

with r2c2:
    prob_by_year = np.mean(wealth_paths >= target_portfolio, axis=0)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=years_axis, y=prob_by_year,
        mode="lines", name="P(FI)", fill="tozeroy", line=dict(width=2)
    ))
    fig3.update_layout(
        title="Probability of Reaching Target Portfolio",
        xaxis_title="Year", yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=380
    )
    st.plotly_chart(fig3, use_container_width=True)