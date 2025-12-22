# Question 3 — Seasonality: When and Where Are the Highs and Lows?

## Why this question matters
For FilmViet, it’s not just about where to release, but also when. Indian film demand isn’t steady all year—some weeks are reliably busy, others are quiet. By understanding these patterns, FilmViet can pick the best release windows and match the right films to the right times and places.

---

## What we’re looking for
We want to know: for each state, city, and top cinema, which calendar weeks are usually the busiest and which are the quietest? And how can we use that to plan smarter releases?

---

## How I approached it
1. **Calendar weeks**: I converted all dates to ISO week-of-year, so we can compare “Week 33” across different years.
2. **Aggregate by week**: I summed up box office, number of titles, and active cinemas for each week, at the state, city, and cinema level.
3. **Seasonality patterns**: For each place and week, I calculated average and median gross, and crowding (how many titles/cinemas).
4. **Highlight peaks and troughs**: I normalized the data to show which weeks are unusually busy or quiet.
5. **Find the extremes**: I listed the top 3 busiest and quietest weeks for each state.
6. **City and cinema deep dives**: I looked at which cities drive the peaks, and which are reliable all year.
7. **Opportunity analysis**: I combined demand and competition to find the best “bang for buck” weeks.

---

## What the outputs show

- **State heatmaps**: See at a glance which weeks are hot or cold for each state, and how crowded they are.
- **Peak/trough tables**: The best and worst weeks for each state, with crowding info.
- **Seasonality lines**: How strongly each state swings between busy and quiet.
- **City heatmaps and bump charts**: Which cities drive the peaks, and which are reliable anchors.
- **Opportunity quadrant**: Which weeks offer high demand with low competition.

---

## How FilmViet can use this
- **Release calendar**: Schedule big films in peak weeks, avoid troughs for tentpoles.
- **Location targeting**: In busy weeks, focus on states and cities that overperform.
- **Competition management**: Use the opportunity chart to find high-demand, low-competition windows.
- **Strategic planning**: Build a release plan that matches the right film to the right week and place, balancing demand and competition.

### 5. Identify busiest/quietest weeks directly
- For each state, extract **Top-K peak weeks** and **Top-K quiet weeks** based on `avg_gross`.

### 6. Deep dive into where peaks come from
- Select **key cities** (top by total average gross) and plot their week-of-year seasonality.
- Use a bump chart to show which cities consistently rank in the top by week.

### 7. Turn demand + competition into a decision tool
- Create z-scores for:
  - demand proxy: `gross_z` (z(avg_gross))
  - competition proxy: `titles_z` (z(avg_titles))
- Build an `opportunity_score = gross_z - 0.8 * titles_z`
- Plot a quadrant: **up = high demand**, **left = lower competition**.

---

## What each output represents (in order)

### Output 1 — State × Calendar Week heatmaps (3 panels)
- **Panel 1: Seasonality (z-scored avg gross)**
  - Red = busier-than-usual weeks for that state
  - Blue = quieter-than-usual weeks
- **Panel 2: Avg # active Indian titles** (competition proxy)
  - Higher = more Indian titles in market that week
- **Panel 3: Avg # active cinemas** (screen pressure proxy)
  - Higher = more cinemas screening Indian titles that week
- **Key takeaway:**
  - Shows when each state has predictable **peaks and troughs**, and whether those peaks are also crowded.

### Output 2 — Peak/trough tables (Top-K weeks per state)
- Lists the **busiest weeks** and **quietest weeks** for each state using `avg_gross`.
- Also includes `avg_titles` and `avg_cinemas` to show how crowded the week typically is.
- **Key takeaway:**
  - Produces a direct "shortlist of weeks" for planning releases.

### Output 3 — State seasonality line charts (top states)
- **Seasonality index** per state: `avg_gross / median_week`
- Index > 1 = busier-than-usual weeks; < 1 = quieter-than-usual.
- **Key takeaway:**
  - Shows the *strength* of seasonal swings (some states spike harder than others).

### Output 4 — Key Cities × Calendar Week heatmap (seasonality index)
- Rows = key cities; columns = ISO weeks; color centered at 1.0
- Red = city overperforms its own baseline that week
- Blue = underperforms
- **Key takeaway:**
  - Identifies *which cities drive the peak weeks* and which cities are weak during those weeks.

### Output 5 — Bump chart (city ranks by calendar week)
- Tracks how cities rank by avg gross across weeks.
- Cities that stay near the top are **reliable anchors**.
- Cities that jump up/down are **seasonal plays**.
- **Key takeaway:**
  - Helps pick a "core rollout city set" vs "seasonal add-on city set."

### Output 6 — Opportunity quadrant (demand vs competition)
- **Y-axis:** demand (`gross_z`) — up is better
- **X-axis:** competition (`titles_z`) — left is better
- Best release windows are in the **upper-left quadrant** (high demand, low competition).
- **Key takeaway:**
  - Highlights state-week combinations where FilmViet can maximise demand without heavy title crowding.

---

## How this helps FilmViet (business actions)

### Pick release windows
- Use peak weeks (high seasonality / high avg gross) for tentpoles.
- Use quiet weeks for smaller titles or longer-hold strategies.

### Pick locations per week
- In peak weeks, prioritise states/cities that **over-index** (strong red in city/state heatmaps).
- Use bump chart to anchor releases in consistently top cities.

### Balance demand vs competition
- Use the opportunity quadrant to target weeks that are **high demand but not overly crowded**.

### Planning strategy
- Build a release calendar that aligns:
  - (1) peak weeks,
  - (2) best-performing states/cities,
  - (3) manageable competition levels.

---

## Multiple Perspectives on Q3

### Perspective 1: Demand-Centric View
**Question:** When is demand naturally high?
- **Answer:** Peak weeks show sustained demand across states
- **Action:** Schedule major releases on peak weeks (Week 33–40 often strong in Australian summer)
- **Metric:** avg_gross per week

### Perspective 2: Competition-Aware View
**Question:** Are peak weeks also crowded with other Indian titles?
- **Answer:** Heatmap Panel 2 (avg_titles) shows competition levels
- **Action:** If peak weeks are crowded, consider Week 2–3 peaks to differentiate
- **Metric:** avg_titles per week (inverse: high titles = harder to stand out)

### Perspective 3: Regional Variation View
**Question:** Which states have the strongest seasonality signals?
- **Answer:** Line chart (Output 3) shows seasonality index by state
- **Action:** States with strong peaks (index > 1.5) are more predictable; plan accordingly
- **Metric:** seasonality_index = avg_gross / median_week per state

### Perspective 4: City-Level Micro-Seasonality View
**Question:** Which cities drive the peak weeks, and are they reliable all year?
- **Answer:** Heatmap (Output 4) shows city overperformance per week; bump chart (Output 5) shows consistency
- **Action:** Anchor releases in top-ranking, stable cities (e.g., always top-3); add seasonal cities as upside
- **Metric:** city rank by week (bump chart), seasonality index per city per week

### Perspective 5: Opportunity Cost View
**Question:** Which weeks offer the best "bang for buck" (high demand, low competition)?
- **Answer:** Upper-left quadrant of opportunity plot (Output 6) shows uncontested peak demand
- **Action:** Target these golden weeks for FilmViet releases
- **Metric:** opportunity_score = gross_z - 0.8 * titles_z

### Perspective 6: Cinema-Level Deep Dive View
**Question:** Which key cinemas (top performers) have distinct seasonality patterns?
- **Answer:** Cinema-week aggregations show revenue by cinema and week
- **Action:** In peak weeks, allocate more screens to over-indexing cinemas; reduce in troughs
- **Metric:** avg_gross per cinema per week, seasonality index per cinema
