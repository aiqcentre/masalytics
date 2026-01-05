# 🍿 Masalytics 

*Film Viet Australia — Competitive Analysis of Indian Films (2025)*

---

## 🎬 Purpose

**Masalytics** helps **Film Viet Australia** understand how Indian films perform in Australia — so we can make smarter decisions when scheduling and marketing Vietnamese releases.

Indian cinema is one of our biggest competitors for screens, marketing attention, and international audiences.
This project tracks **Indian film releases in 2025** to benchmark:

* 📈 **Box-office performance**
* 🗓️ **Release timing & competition windows**
* 🏢 **Distributor activity & reach in Australia**
* 📍 **Location-level demand** (state / cinema chains)

The goal: **protect screening slots, optimise release dates, and refine marketing spend** for Vietnamese titles.

---

## 💡 What We Answer

* Which Indian films perform best in Australia in 2025 — and why?
* Which **months & weekends** are biggest for Indian demand?
* Which **distributors control the strongest circuits**?
* Where do Indian films over-index (e.g. VIC vs NSW)?
* How much space is left for Vietnamese titles in key weeks?
* Which genres, stars, and language markets drive ticket sales?

---

## 🔧 Outputs

| Deliverable                             | Description                                                             |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `DataExplorationMain.py`                | Core ETL module — loads DB, flattens JSON, exports common variables     |
| `SalesOverview.py`                      | Box-office trends, top films, monthly/daily analysis                    |
| `LocationQuestions.py`                  | Q1/Q2/Q3 location stability & timing analysis                           |
| `TitlesDistributors.py`                 | Distributor market share, treemaps, Pareto charts                       |
| `extract_market_totals.py`              | Q2 market value breakdown by timing type                                |
| `Q2_Mapping_Step_by_Step.ipynb`         | Interactive SA4 geographic mapping notebook                             |
| `outputs_salesoverview/`                | Top 10 films, monthly trends visualizations                             |
| `outputs_locationquestions/`            | Q2 cinema/city summaries, SA4 timing maps                               |
| `outputs_titlesdistributors/`           | Distributor treemap, Pareto chart, state heatmap                        |

---

## 📊 Metrics Tracked

* Gross revenue (opening & total)
* Screens & sessions (where available)
* Release month / day-of-week behaviour
* Distributor footprint & consistency
* State / city performance
* Genre + language segmentation

---

## 👥 Roles

| Role      | Responsibilities                                                         |
| --------- | ------------------------------------------------------------------------ |
| **DS**    | Build sales benchmarks, timing models, clustering by region & demand     |
| **DA/BA** | Source data, clean datasets, extract competitor insights, create reports |

---

## 🗓️ Timeline

| Phase    | Target                                            |
| -------- | ------------------------------------------------- |
| Week 1–2 | Collect & clean India-film AU cinema data         |
| Week 3   | Identify distributor networks + region strengths  |
| Week 4   | Analyse timing patterns + Film Viet overlap weeks |
| Week 5   | Build dashboards + structured insights            |
| Week 6   | PDF report + screening strategy recommendations   |

---

## 🎯 Success Criteria

* Clear **competitor map** of Indian releases in Australia
* Actionable **release timing rules** for Vietnamese titles
* Location insights to **prioritise suburbs/cinemas**
* Reusable data pipeline for future quarters
