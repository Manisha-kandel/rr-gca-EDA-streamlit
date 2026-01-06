
# RR Grade Crossing Accident EDA (Streamlit)

An interactive **Streamlit dashboard** for exploratory data analysis (EDA) of **U.S. railroad–highway grade crossing accidents**, based on an app-ready dataset derived from FRA records.

The app is designed for **filter-driven, metric-aware exploration** of accident patterns across time, geography, and operational conditions.

---

## 📂 Dataset Overview

* **File**: `rr_grade_crossing_accident_data_app_ready.csv.gz`
* **Size**: ~5.7 MB
* **Rows**: 239,477
* **Columns**: 23
* **Time span**: 1975–2021
* **Source**: U.S. Federal Railroad Administration (FRA) grade crossing accident reports
Key fields include:

* Date/time attributes (Year, Month, Weekday, Hour24)
* Location (State Code, State Name, County, City)
* Casualties (Total Killed, Total Injured)
* Context (Public/Private crossing, Highway User, Weather, Visibility, Roadway Condition)
* Train attributes (Railroad, Train Speed)

All required preprocessing (typing fixes, NaT handling, weekday normalization) is handled inside the app.

---

## 🎛 Global Filters

All charts respond to the following controls:

* **Year range slider**
  Filter accidents by year (e.g., 1975–2021)

* **State multiselect**
  Select one or more states using
  `State Name (USPS code)` format (e.g., `California (CA)`)

* **Metric selector**

  * **Incidents** — count of accidents
  * **Killed** — total fatalities
  * **Injured** — total injuries

---

## 📊 Visualizations

### 1️⃣ Choropleth Map (Primary View)

* Displays U.S. states with **only selected states highlighted**
* Unselected states remain muted for geographic context
* State abbreviations (e.g., CA, DE) are shown directly on the map
* Metric-aware coloring (Incidents / Killed / Injured)
* Hover shows **state + metric value only** (no lat/lon noise)

**Use case:** spatial comparison and focused state-level analysis.

---

### 2️⃣ Weekday × Hour Heatmap

* Heatmap of the selected metric across:

  * Weekdays (Mon–Sun)
  * Hours of day (0–23)
* Square cells for visual consistency
* Yellow → Red color scale
* Integer values only

**Use case:** identifying temporal accident patterns.

---

### 3️⃣ Weekday Bar Chart

* Aggregated metric by weekday
* Automatically switches between:

  * incident counts
  * killed totals
  * injured totals

**Use case:** understanding day-of-week effects.

---

### 4️⃣ Top States (Horizontal Bar Chart)

* Ranks states by the selected metric
* User-selectable **Top N** (5 / 10 / 15 / 20)
* Horizontal bars
* Sorted descending (highest at top)

**Use case:** comparative ranking across states.

---

## 📄 Sample Data View

* Optional **“Show sample rows”** toggle at the bottom of the page
* Displays a preview of the **already filtered dataset**
* Adjustable row count (e.g., 25–200)

This keeps the main layout clean while still allowing data inspection.

---

## 🗂 Project Structure

```
rr-gca-EDA-streamlit/
│
├── app.py                      # Streamlit application
├── data/
│   └── rr_grade_crossing_accident_data_app_ready.csv.gz
├── notebooks/                  # Preprocessing / EDA notebooks
├── requirements.txt
└── README.md
```

---

## ▶️ Running the App

```bash
# create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run app
streamlit run app.py
```

---

## 🎯 Purpose

This dashboard is intended for:

* exploratory data analysis
* pattern discovery across time and geography
* supporting research and risk-focused discussions related to railroad grade crossing safety

It is **not** a predictive or causal model, but a transparent, filter-driven EDA tool.

