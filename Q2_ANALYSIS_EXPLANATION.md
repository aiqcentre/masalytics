# Question 2 — Early Adopters vs Slow-burn: When Does the Audience Show Up?

## Why this question matters
Timing is everything. Some places rush out to see new Indian films in the first two weeks—these are your “early adopters.” Others take their time, with audiences building over several weeks. Knowing which is which helps FilmViet plan when and where to launch, how long to hold screens, and when to ramp up or ease off marketing.

---

## What we’re looking for
We want to find out: which states, cities, and cinemas are “fast starters” and which are “slow burners”? This helps decide where to launch first and where to wait.

---

## How I approached it
1. **Relative week**: For every film and location, I tracked which week was “week 1” (the first week it played there), so I could compare movies that started at different times.
2. **Early vs late revenue**: I split each location’s box office into “early” (weeks 1–2) and “late” (week 3+), and calculated the share of revenue that came early.
3. **Weighted averages**: Bigger films count more, so I weighted the averages by total gross.
4. **Timing types**: Locations are grouped into Early Adopters, Balanced, or Slow-burn, based on how much revenue comes in the first two weeks.
5. **Validation**: I checked how quickly each place reaches 95% of its total revenue.

---

## What the outputs show

- **Timing maps**: Bubble charts show which cities and cinemas are fast or slow, and how big their markets are.
- **Speed bar charts**: See how many cinemas and cities reach 95% of their revenue in 1, 2, 3, or more weeks, broken down by timing type.

---

## How FilmViet can use this
- **Release sequencing**: Launch in early adopter locations first for quick returns, then roll out to slow-burn markets.
- **Screen planning**: Hold screens for a shorter time in fast markets, and longer in slow ones.
- **Marketing**: Go big at launch in early adopter cities; spread out the spend in slow-burn places.
- **Portfolio**: Don’t put all your eggs in one basket—balance quick wins with long-tail markets.
     - Bottom 25% → **SLOW_BURN**

5. **Validate timing using "weeks_to_95"**
   - Build cumulative revenue share across rel_week per place.
   - Compute:
     - `weeks_to_95` = first rel_week where cumulative share ≥ 95%
   - Early adopters should hit 95% faster; slow-burn should take longer.

---

## What each output represents (in order)

### Output 1 — Timing Map (Cities): Early Share vs Total Gross
- **Each dot = a city**
- **X-axis:** `weighted_early_share` (Week 1–2 share of total)
  - Right = people watch quickly (early adopter)
  - Left  = demand builds later (slow-burn)
- **Y-axis:** `total_gross` (log scale)
  - Higher = bigger market overall
- **Color groups:** EARLY_ADOPTER / BALANCED / SLOW_BURN (based on Q25/Q75)
- **Key takeaway:**
  - **Top-right** = best "release first" cities (big + fast payoff)
  - **Top-left**  = big but slow-burn cities (worth releasing, but later/longer hold)

### Output 2 — Timing Map (Cinemas): Early Share vs Total Gross
- **Same logic as cities, but at cinema level**
- Shows which individual cinemas are early vs slow-burn adopters
- Helps FilmViet decide:
  - Which cinemas to include in Wave 1 rollout
  - Which to hold back for Wave 2 (extended run)
- **Top 10 cinemas labeled** to avoid clutter

### Output 3 — Cinema Speed Distribution: weeks_to_95 by Timing Class
- **Each bar = a timing class** (SLOW_BURN, BALANCED, EARLY_ADOPTER)
- **Stack segments** show how many cinemas reach 95% cumulative revenue by:
  - Week 1, Week 2, Week 3, Week 4, etc.
- **Key takeaway:**
  - **EARLY_ADOPTER** cinemas cluster at **weeks_to_95 = 1–2** (fast payoff)
  - **SLOW_BURN** cinemas shift to **weeks_to_95 = 3+** (need longer run)
  - Shows the business reality: early adopters recoup fast, slow-burn need patience

### Output 4 — City Speed Distribution: weeks_to_95 by Timing Class
- **Same as Output 3, but for cities**
- Aggregate cinema-level patterns into city markets
- Helps FilmViet understand:
  - City-level market velocity
  - Which cities are "quick turnovers" vs "long-tail markets"

---

## How this helps FilmViet (business actions)

### Release Sequencing
- **Wave 1 (Days 1–2):** Release in EARLY_ADOPTER cities/cinemas
  - Fast cashflow, quick ROI confirmation
  - Minimize screen holds in these locations
- **Wave 2 (Week 2–3):** Add BALANCED and SLOW_BURN markets
  - Let Wave 1 build word-of-mouth
  - Extended screen holds in slow-burn locations

### Screen Planning
- **Early adopters:** Plan for peak in Week 1–2
  - Shorter screen hold (2–3 weeks typical)
  - Maximize screens in Week 1
- **Slow-burn:** Plan for extended run (4–5 weeks+)
  - Lower peak, but sustained demand
  - Hold screens longer, monitor weekly trends

### Marketing Budget
- **Early adopter cities:** Front-load marketing, launch intensity high
  - "Frontload" strategy: capture demand spike
- **Slow-burn cities:** Steady marketing throughout run
  - "Drip" strategy: build awareness over time

### Portfolio Strategy
- Avoid releasing **only** in early adopters (leaves money on the table)
- Avoid releasing **all at once** in slow-burn markets (cash flow risk)
- **Balanced rollout:** 40% early adopters (Week 1) + 60% balanced/slow-burn (Week 2+)
  - Staggered revenue, managed risk, optimized cash flow
