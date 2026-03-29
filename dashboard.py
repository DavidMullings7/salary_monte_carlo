import logging
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
    tax_a_first  = tax_liability(a) - tax_liability(0)
    tax_a_second = tax_liability(a + b) - tax_liability(b)
    tax_b_first  = tax_liability(b) - tax_liability(0)
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
    additional_medicare = max(0, wages - ADDITIONAL_MEDICARE_THRESHOLD) * ADDITIONAL_MEDICARE_RATE
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
    return np.array([start_salary * (1 + salary_growth) ** y for y in range(years)])

# -----------------------------
# PRETAX CONTRIBUTION SOLVER
# -----------------------------

def max_pretax_contribution(salary, partner_income, spending):
    def objective(pretax):
        net_income = personal_after_tax_income(salary, partner_income, pretax, 0)
        return net_income - spending
    try:
        pretax = float(brentq(objective, 0, K401_LIMIT + HSA_LIMIT))
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
        if excess < -1.0:
            return None
        excess = max(excess, 0)
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


def generate_joint_return_paths(accum_years, retire_years, accum_mean, retire_mean,
                                 return_std, phi=-0.15):
    """Generate a single AR(1) process spanning accumulation and retirement, then split.

    Using one continuous process means the market state at the retirement boundary
    carries over: a crash in the final accumulation years depresses early retirement
    returns through the AR(1) dynamics, correctly modelling sequence-of-returns risk
    at the transition. Two independently seeded draws would miss this correlation
    entirely.

    The unconditional mean shifts at year accum_years (e.g. from a growth-oriented
    portfolio to a more conservative one), while the innovation structure — volatility,
    autocorrelation, and the latent state — remains continuous.

    Returns:
        accum_growth: (SIMULATIONS, accum_years)
        retire_growth: (SIMULATIONS, retire_years)
    """
    total_years = accum_years + retire_years
    noise_std = return_std * np.sqrt(1 - phi ** 2)

    log_mean_accum  = np.log(1 + accum_mean)  - 0.5 * return_std ** 2
    log_mean_retire = np.log(1 + retire_mean) - 0.5 * return_std ** 2

    # Per-period mean vector: accumulation mean for first accum_years, then retirement mean
    log_means = np.empty(total_years)
    log_means[:accum_years]  = log_mean_accum
    log_means[accum_years:]  = log_mean_retire

    log_returns = np.zeros((SIMULATIONS, total_years))
    log_returns[:, 0] = np.random.normal(log_means[0], return_std, SIMULATIONS)

    for t in range(1, total_years):
        noise = np.random.normal(0, noise_std, SIMULATIONS)
        # AR(1): deviation is measured against the mean of the *previous* period so
        # the autocorrelation term is well-defined even when the mean shifts at t=accum_years.
        log_returns[:, t] = log_means[t] + phi * (log_returns[:, t-1] - log_means[t-1]) + noise

    growth = np.exp(log_returns)
    return growth[:, :accum_years], growth[:, accum_years:]

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
# DECUMULATION
# -----------------------------

def required_portfolio_from_spend(monthly_spend, retirement_years, annual_return, social_security_monthly=0):
    """Amortization (PV of annuity): the lump-sum portfolio at retirement needed to fund
    (monthly_spend - social_security_monthly) per month for retirement_years years
    at annual_return, drawing down to zero."""
    net_monthly = monthly_spend - social_security_monthly
    if net_monthly <= 0:
        return 0.0
    r = annual_return / 12   # monthly rate
    n = retirement_years * 12
    if r <= 0:
        return net_monthly * n
    return net_monthly * ((1 + r) ** n - 1) / (r * (1 + r) ** n)


def make_annual_draws(monthly_spend, social_security_monthly, ss_start_year, retirement_years):
    """Build the per-year portfolio draw schedule for retirement.

    During the SS gap (years 0..ss_start_year-1) the full monthly_spend must be
    funded from the portfolio. Once SS begins, the draw is reduced by
    social_security_monthly. ss_start_year=0 means SS starts immediately.

    Returns an array of shape (retirement_years,)."""
    full_draw    = monthly_spend * 12
    reduced_draw = max(0.0, (monthly_spend - social_security_monthly) * 12)
    draws = np.full(retirement_years, reduced_draw)
    draws[:ss_start_year] = full_draw
    return draws


def simulate_decumulation_paths(starting_portfolios, annual_draws, retirement_years, ret_growth):
    """Monte Carlo drawdown using pre-generated return paths.
    Each simulation uses its own ending accumulation value as the starting portfolio.
    annual_draws is a (retirement_years,) array of per-year withdrawals, allowing
    the draw to vary (e.g. higher before Social Security begins).
    Returns wealth_paths (SIMULATIONS, retirement_years+1) and the portfolio survival rate."""
    sims = len(starting_portfolios)
    wealth = np.zeros((sims, retirement_years + 1))
    wealth[:, 0] = starting_portfolios
    for y in range(retirement_years):
        wealth[:, y + 1] = np.maximum(wealth[:, y] * ret_growth[:, y] - annual_draws[y], 0)
    survival_rate = float(np.mean(wealth[:, -1] > 0))
    return wealth, survival_rate


def solve_required_retirement_portfolio(
    monthly_spend, retirement_years, retire_mean, return_std,
    social_security_monthly, ss_start_year, target_survival_rate, ret_growth
):
    """Find the portfolio P at retirement such that the Monte Carlo survival rate equals
    target_survival_rate, using the caller-supplied ret_growth paths.

    Accepting pre-generated paths (rather than seeding internally) is what makes
    cross-period consistency possible: the caller generates accumulation and retirement
    paths from a single AR(1) process via generate_joint_return_paths, so the market
    state at the retirement boundary is shared across both phases.

    Returns (required_portfolio, annual_draws) where annual_draws is the per-year
    withdrawal schedule used in both the solver and the final simulation."""
    annual_draws = make_annual_draws(
        monthly_spend, social_security_monthly, ss_start_year, retirement_years
    )

    if annual_draws.max() == 0:
        return 0.0, annual_draws

    def survival_at(portfolio):
        starting = np.full(SIMULATIONS, portfolio)
        _, rate = simulate_decumulation_paths(starting, annual_draws, retirement_years, ret_growth)
        return rate

    # Upper bound: PV assuming the worst case (no SS ever) × 6, which covers targets
    # up to ~99% survival across typical return/volatility assumptions.
    det_pv_no_ss = required_portfolio_from_spend(monthly_spend, retirement_years, retire_mean, 0)
    lo = 0.0
    hi = max(det_pv_no_ss * 6, 1_000_000.0)

    required_portfolio = float(brentq(lambda p: survival_at(p) - target_survival_rate, lo, hi))
    return required_portfolio, annual_draws

# -----------------------------
# SALARY MODEL
# -----------------------------

def simulate_salary_model(
    start_salary, partner_income, years, spending,
    current_portfolio, target_portfolio, growth,
    salary_growth, match_rate, inheritance, inheritance_year
):
    logger.info(
        "simulate_salary_model | start_salary=%s partner_income=%s years=%s spending=%s "
        "current_portfolio=%s target_portfolio=%s salary_growth=%s match_rate=%s "
        "inheritance=%s inheritance_year=%s",
        start_salary, partner_income, years, spending,
        current_portfolio, target_portfolio, salary_growth, match_rate,
        inheritance, inheritance_year,
    )
    salaries = generate_salary_path(start_salary, years, salary_growth)
    contributions = compute_contributions(salaries, partner_income, spending, match_rate)
    if contributions is None:
        return None, 0, None, None, None
    if inheritance_year is not None:
        contributions[inheritance_year - 1] += inheritance
    portfolios = compute_final_portfolios(current_portfolio, contributions, growth)
    wealth_paths = simulate_wealth_paths(current_portfolio, contributions, growth)
    success_rate = np.mean(portfolios >= target_portfolio)
    if inheritance_year is not None:
        contributions[inheritance_year - 1] -= inheritance
    return portfolios, success_rate, salaries, contributions, wealth_paths

# -----------------------------
# SALARY SOLVER
# -----------------------------

def solve_required_salary(
    partner_income, years, spending, current_portfolio, target_portfolio,
    salary_growth, match_rate, target_success, inheritance, inheritance_year,
    accum_growth
):
    def objective(salary):
        _, success, _, _, _ = simulate_salary_model(
            salary, partner_income, years, spending, current_portfolio,
            target_portfolio, accum_growth, salary_growth, match_rate,
            inheritance, inheritance_year
        )
        return success - target_success

    salary = float(brentq(objective, 50000, 600000))
    return salary

# ==============================
# STREAMLIT UI
# ==============================

# -----------------------------
# SIDEBAR — all inputs
# -----------------------------

with st.sidebar:

    st.title("Inputs")

    # Household
    st.subheader("Household")
    partner_income    = st.number_input("Partner Income ($)",           0, 1_000_000, 300_000, step=5_000)
    spending          = st.number_input("Working-Years Spending ($/yr)", 0,   500_000,  50_000, step=1_000)
    current_portfolio = st.number_input("Current Portfolio ($)",         0, 10_000_000, 240_000, step=10_000)

    st.divider()

    # Retirement Income Goal — the primary input
    st.subheader("Retirement Income Goal")
    monthly_retirement_spend = st.number_input(
        "Monthly Spend in Retirement ($)", 0, 50_000, 9_000, step=500,
        help="Target monthly spending once retired, in today's dollars."
    )
    rc1, rc2 = st.columns(2)
    with rc1:
        retirement_years = st.slider("Years in Retirement", 10, 50, 40)
        target_survival_rate = st.slider(
            "Portfolio Survival Rate", 50, 99, 75, format="%d%%",
            help="Probability that the portfolio lasts the full retirement. "
                 "Higher values require a larger portfolio, accounting for sequence-of-returns risk."
        ) / 100
    with rc2:
        social_security_monthly = st.number_input(
            "Social Security ($/mo)", 0, 10_000, 2_500, step=100,
            help="Expected monthly Social Security benefit. Offsets required portfolio draw."
        )
        if social_security_monthly == 0:
            ss_start_year = 0
            st.caption("SS Delay (years) is ignored when Social Security monthly benefit is $0.")
        else:
            ss_start_year = st.number_input(
                "SS Delay (years)", 0, retirement_years, 20, step=1,
                help="Years after retirement before Social Security begins. "
                     "During this gap the full monthly spend is drawn from the portfolio."
            )

    st.divider()

    # Accumulation
    st.subheader("Accumulation")
    gc1, gc2 = st.columns(2)
    with gc1:
        years = st.slider("Years to Retirement", 5, 40, 25)
    with gc2:
        target_success = st.slider("Success Rate", 50, 99, 70, format="%d%%") / 100

    st.divider()

    # Inheritance
    st.subheader("Inheritance")
    ic1, ic2 = st.columns(2)
    with ic1:
        inheritance = st.number_input("Amount ($)", 0, 10_000_000, 1_500_000, step=50_000)
    with ic2:
        # inheritance_year = st.number_input("Year", 1, years, years)
        if inheritance == 0:
            inheritance_year = 0
            st.caption("Inheritance year is ignored when inheritance is $0.")
        else:
            inheritance_year = st.number_input("Year", 1, years, years, help="The year in which the inheritance is received.")

    st.divider()

    # Market
    st.subheader("Market")
    mc1, mc2 = st.columns(2)
    with mc1:
        mean_return       = st.slider("Accum. Return",     0.0, 15.0, 4.0, step=0.5, format="%.1f%%") / 100
        return_std        = st.slider("Volatility (σ)",    0,   30,   18,             format="%d%%")   / 100
    with mc2:
        retirement_return = st.slider("Retirement Return", 0.0, 10.0,  4.0, step=0.5, format="%.1f%%") / 100
        salary_growth     = st.slider("Salary Growth",     0.0, 10.0,  1.5, step=0.5, format="%.1f%%") / 100

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
            SOCIAL_SECURITY_RATE      = st.number_input("SS Rate",          0.0, 0.20,      0.062,   step=0.001, format="%.3f")
            SOCIAL_SECURITY_WAGE_BASE = st.number_input("SS Wage Base ($)", 0,   500_000,   168_600, step=1_000)
        with f2:
            MEDICARE_RATE                 = st.number_input("Medicare Rate",       0.0, 0.10, 0.0145,  step=0.001, format="%.4f")
            ADDITIONAL_MEDICARE_RATE      = st.number_input("Add'l Medicare Rate", 0.0, 0.10, 0.009,   step=0.001, format="%.3f")
            ADDITIONAL_MEDICARE_THRESHOLD = st.number_input("Add'l Threshold ($)", 0, 1_000_000, 250_000, step=10_000)

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

st.title("Retirement Planning: Income & Savings Solver")
st.caption(
    "How much do you need to earn and save to achieve your target monthly spending in retirement?"
)

if not run:
    st.info("Configure inputs in the sidebar, then click **Solve Required Salary**.")
    st.stop()

# -----------------------------
# SOLVE + RESULTS
# -----------------------------

logger.info(
    "Inputs | monthly_retirement_spend=$%s retirement_years=%s target_survival_rate=%s "
    "social_security_monthly=$%s partner_income=$%s years=%s spending=$%s "
    "current_portfolio=$%s salary_growth=%s match_rate=%s inheritance=$%s inheritance_year=%s",
    f"{monthly_retirement_spend:,.0f}", retirement_years, f"{target_survival_rate:.0%}",
    f"{social_security_monthly:,.0f}", f"{partner_income:,.0f}", years, f"{spending:,.0f}",
    f"{current_portfolio:,.0f}", f"{salary_growth:.2%}", f"{match_rate:.2%}",
    f"{inheritance:,.0f}", inheritance_year,
)

with st.spinner("Running Monte Carlo simulation..."):

    # Generate one AR(1) process spanning both phases. Splitting it at the retirement
    # boundary preserves the market state across the transition: a crash at the end of
    # accumulation carries into early-retirement returns via the autoregressive dynamics.
    # Two independently seeded draws (the prior approach) would sever that link entirely.
    np.random.seed(42)
    accum_growth, ret_growth = generate_joint_return_paths(
        years, retirement_years, mean_return, retirement_return, return_std
    )

    # Step 1: Solve for the portfolio needed at retirement using Monte Carlo decumulation.
    # Uses the retirement slice of the joint paths so sequence-of-returns context is intact.
    target_portfolio, annual_draws = solve_required_retirement_portfolio(
        monthly_retirement_spend, retirement_years, retirement_return, return_std,
        social_security_monthly, ss_start_year, target_survival_rate, ret_growth
    )

    logger.info("target_portfolio (MC-derived)=$%s ss_start_year=%s annual_draws=%s",
                f"{target_portfolio:,.0f}", ss_start_year,
                [f"${d:,.0f}" for d in annual_draws])

    # Step 2: Solve for the required salary to accumulate target_portfolio
    required_salary = solve_required_salary(
        partner_income, years, spending, current_portfolio, target_portfolio,
        salary_growth, match_rate, target_success, inheritance, inheritance_year,
        accum_growth
    )

    # Step 3: Full accumulation simulation
    portfolios, accum_success, salaries, contributions, wealth_paths = simulate_salary_model(
        required_salary, partner_income, years, spending, current_portfolio,
        target_portfolio, accum_growth, salary_growth, match_rate,
        inheritance, inheritance_year
    )
    assert portfolios is not None and wealth_paths is not None and contributions is not None

    # Step 4: Decumulation — each sim's ending accumulation portfolio seeds its own drawdown.
    # ret_growth[i] is the continuation of the same AR(1) path that produced portfolios[i],
    # so a simulation that experienced a late-accumulation crash also experiences the
    # correlated early-retirement environment from that same path.
    decum_paths, survival_rate = simulate_decumulation_paths(
        portfolios, annual_draws, retirement_years, ret_growth
    )

years_axis  = list(range(years + 1))
contrib_x   = years_axis[1:]
retire_axis = list(range(years, years + retirement_years + 1))

# -----------------------------
# TOP METRICS
# -----------------------------

st.subheader("Results")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Required Starting Salary", f"${required_salary:,.0f}")
with m2:
    st.metric(
        "Target Portfolio at Retirement", f"${target_portfolio:,.0f}",
        help="Derived via Monte Carlo: the portfolio at retirement that achieves your "
             "target survival rate accounting for sequence-of-returns risk."
    )
with m3:
    st.metric(
        "Accumulation Success", f"{accum_success:.1%}",
        help="Share of simulations that reach the target portfolio by retirement."
    )
with m4:
    st.metric(
        "Portfolio Survival Rate", f"{survival_rate:.1%}",
        help="Share of simulations where the portfolio lasts the full retirement period."
    )

st.divider()

# -----------------------------
# UNIFIED TIMELINE
# -----------------------------

st.subheader("Retirement Journey")

accum_p10 = np.percentile(wealth_paths, 10, axis=0)
accum_p50 = np.percentile(wealth_paths, 50, axis=0)
accum_p90 = np.percentile(wealth_paths, 90, axis=0)

decum_p10 = np.percentile(decum_paths, 10, axis=0)
decum_p50 = np.percentile(decum_paths, 50, axis=0)
decum_p90 = np.percentile(decum_paths, 90, axis=0)

BLUE   = "#2196F3"
ORANGE = "#FF9800"

fig_timeline = go.Figure()

# Accumulation phase — blue
fig_timeline.add_trace(go.Scatter(
    x=years_axis, y=accum_p90, name="Accum 90th",
    line=dict(dash="dot", width=1, color=BLUE), opacity=0.45, showlegend=False,
))
fig_timeline.add_trace(go.Scatter(
    x=years_axis, y=accum_p50, name="Accumulation (median)",
    line=dict(width=2.5, color=BLUE),
))
fig_timeline.add_trace(go.Scatter(
    x=years_axis, y=accum_p10, name="Accum 10th",
    line=dict(dash="dot", width=1, color=BLUE), opacity=0.45, showlegend=False,
))

# Decumulation phase — orange
fig_timeline.add_trace(go.Scatter(
    x=retire_axis, y=decum_p90, name="Retire 90th",
    line=dict(dash="dot", width=1, color=ORANGE), opacity=0.45, showlegend=False,
))
fig_timeline.add_trace(go.Scatter(
    x=retire_axis, y=decum_p50, name="Retire 50th",
    line=dict(width=2.5, color=ORANGE),
))
fig_timeline.add_trace(go.Scatter(
    x=retire_axis, y=decum_p10, name="Retire 10th",
    line=dict(dash="dot", width=1, color=ORANGE), opacity=0.45, showlegend=False,
))

fig_timeline.add_vline(
    x=years, line_dash="dash", line_color="gray",
    annotation_text="Retirement", annotation_position="top right",
)
fig_timeline.add_hline(
    y=target_portfolio, line_dash="dot", line_color="red", opacity=0.5,
    annotation_text=f"Target ${target_portfolio/1e6:.1f}M", annotation_font_size=11,
)
fig_timeline.update_layout(
    xaxis_title="Year",
    yaxis_title="Portfolio Value",
    yaxis=dict(tickprefix="$", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=440,
)
st.plotly_chart(fig_timeline, use_container_width=True)

st.divider()

# -----------------------------
# SALARY PATH
# -----------------------------

fig_salary = go.Figure()
fig_salary.add_trace(go.Bar(x=contrib_x, y=salaries, name="Salary"))
fig_salary.update_layout(
    title="Salary Path",
    xaxis_title="Year", yaxis_title="Salary",
    xaxis=dict(dtick=1),
    yaxis=dict(tickprefix="$", tickformat=",.0f"),
    height=300,
)
st.plotly_chart(fig_salary, use_container_width=True)

st.divider()

# -----------------------------
# PORTFOLIO DISTRIBUTION + CONTRIBUTIONS
# -----------------------------

r1c1, r1c2 = st.columns(2, gap="medium")

with r1c1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=portfolios, nbinsx=50))
    fig.add_vline(x=target_portfolio, line_dash="dash", line_color="red",
                  annotation_text="Target", annotation_position="top right")
    fig.update_layout(
        title="Portfolio at Retirement (Distribution)",
        xaxis_title="Portfolio Value", yaxis_title="Count",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380,
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
        height=380,
    )
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# DRAWDOWN PATHS + FI PROBABILITY
# -----------------------------

r2c1, r2c2 = st.columns(2, gap="medium")

with r2c1:
    fig_decum = go.Figure()
    fig_decum.add_trace(go.Scatter(
        x=retire_axis, y=decum_p90, name="90th Percentile",
        line=dict(dash="dot", width=1), opacity=0.6,
    ))
    fig_decum.add_trace(go.Scatter(
        x=retire_axis, y=decum_p50, name="Median",
        line=dict(width=2.5),
    ))
    fig_decum.add_trace(go.Scatter(
        x=retire_axis, y=decum_p10, name="10th Percentile",
        line=dict(dash="dot", width=1), opacity=0.6,
    ))
    fig_decum.add_hline(y=0, line_color="red", opacity=0.3,
                        annotation_text="Depleted", annotation_font_size=11)
    fig_decum.update_layout(
        title="Retirement Drawdown Paths",
        xaxis_title="Year", yaxis_title="Portfolio Value",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=380,
    )
    st.plotly_chart(fig_decum, use_container_width=True)

with r2c2:
    prob_by_year = np.mean(wealth_paths >= target_portfolio, axis=0)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=years_axis, y=prob_by_year,
        mode="lines", name="P(FI)", fill="tozeroy", line=dict(width=2),
    ))
    fig3.update_layout(
        title="Probability of Reaching Target Portfolio",
        xaxis_title="Year", yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)
