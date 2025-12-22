# Question 1 — Stable vs Volatile Locations: Where Is It Safe to Bet?

## Why this question matters
When FilmViet releases a movie, picking the right locations is crucial. Some cinemas and cities deliver steady, reliable box office results—these are your “safe bets.” Others are unpredictable: they might flop, or they might surprise you with a big hit. By understanding which places are stable and which are risky, FilmViet can make smarter decisions about where to launch, how to allocate screens, and where to focus marketing.

---

## What we’re looking for
We want to know: which cities and cinemas are consistently strong for Indian films, and which ones are more of a gamble? The goal is to sort locations into:
- **Safer targets** (predictable, steady revenue)
- **Higher-risk targets** (volatile, but with potential for big wins)

---

## How I approached it
1. **Weekly box office**: I calculated the weekly gross for every cinema and city.
2. **Stability vs. risk**: For each, I measured the average weekly revenue (reward) and the volatility (risk, using the coefficient of variation).
3. **Risk-adjusted ranking**: I ranked locations by a “risk-adjusted score”—reward, but discounted if the place is volatile.
4. **Risk groups**: Each location is classified as Safer, Moderate, Higher-risk, or Highly volatile.
5. **Visuals**: I created charts to show the best and riskiest locations, and a heatmap to reveal week-by-week consistency.

---

## What the outputs show

- **Scatter plot**: See which locations are high-reward and stable (bottom-right), and which are high-reward but risky (top-right).
- **Bar charts**: The top 10 safest and riskiest cinemas and cities, ranked by risk-adjusted score.
- **Heatmap**: A “volatility fingerprint” showing which cinemas are steady week after week, and which are all over the place.

---

## How FilmViet can use this
- **Release order**: Start with the safest locations to guarantee a solid base, then expand to riskier places for upside.
- **Budgeting**: Spend more on marketing and screens in stable locations; experiment and monitor closely in volatile ones.
- **Portfolio**: Build a mix of reliable “core” locations and a few high-upside experiments.
  - **Bottom-left:** stable but small → reliable, lower scale
  - **Top-left:** volatile and small → lowest priority

### Output 2 — Bar chart: Top 10 Safer Cinemas (risk-adjusted)
- Ranks cinemas by `risk_adjusted_score`
- Longer bars = **strong and predictable weekly value**

### Output 3 — Bar chart: Top 10 Higher-risk Cinemas (risk-adjusted)
- Same metric, but only among **volatile cinemas**
- Shows the **best of the risky** cinemas (worth testing, but not the core)

### Output 4 — Bar chart: Top 10 Safer Cities (risk-adjusted)
- Identifies city-level **reliable anchors** for rollout
- Strong candidates for "release first" planning

### Output 5 — Bar chart: Top 10 Higher-risk Cities (risk-adjusted)
- Volatile cities that still look attractive after the volatility penalty
- Best for **targeted / experimental** releases

### Output 6 — Heatmap: Weekly pattern fingerprint
- **Rows:** cinemas, **columns:** weeks
- Reveals whether performance is:
  - **steady** (more consistent shading), or
  - **spiky/patchy** (large swings and gaps)
- Visually validates the "stable vs volatile" classification.

---

## How this helps FilmViet (business actions)
- **Release order**
  - Start with **safer cities/cinemas** to secure a predictable baseline.
  - Add **higher-risk locations** later as upside bets.
- **Budget and ops strategy**
  - Safer: standard marketing and stable screen planning.
  - Higher-risk: flexible screens, closer monitoring, shorter test windows.
- **Portfolio strategy**
  - Build a mix of **stable core** + **selective high-upside experiments**.
