import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ----------------- SETUP -----------------
st.set_page_config(page_title="RR Grade Crossing Accident EDA", layout="wide")

DATA_PATH = "data/rr_grade_crossing_accident_data_app_ready.csv.gz"

# Numeric State Code (FIPS) -> USPS abbreviation
FIPS_TO_USPS = {
    1:"AL", 2:"AK", 4:"AZ", 5:"AR", 6:"CA", 8:"CO", 9:"CT", 10:"DE", 11:"DC", 12:"FL", 13:"GA",
    15:"HI", 16:"ID", 17:"IL", 18:"IN", 19:"IA", 20:"KS", 21:"KY", 22:"LA", 23:"ME", 24:"MD",
    25:"MA", 26:"MI", 27:"MN", 28:"MS", 29:"MO", 30:"MT", 31:"NE", 32:"NV", 33:"NH", 34:"NJ",
    35:"NM", 36:"NY", 37:"NC", 38:"ND", 39:"OH", 40:"OK", 41:"OR", 42:"PA", 44:"RI", 45:"SC",
    46:"SD", 47:"TN", 48:"TX", 49:"UT", 50:"VT", 51:"VA", 53:"WA", 54:"WV", 55:"WI", 56:"WY"
}

# State centroids for label placement on choropleth
CENTROIDS = pd.DataFrame({
    "state": ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA',
              'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM',
              'NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA',
              'WV','WI','WY'],
    "Latitude": [32.806671,61.370716,33.729759,34.969704,36.116203,39.059811,41.597782,39.318523,
                 38.897438,27.766279,33.040619,21.094318,44.240459,40.349457,39.849426,42.011539,
                 38.526600,37.668140,31.169546,44.693947,39.063946,42.230171,43.326618,45.694454,
                 32.741646,38.456085,46.921925,41.125370,38.313515,43.452492,40.298904,34.840515,
                 42.165726,35.630066,47.528912,40.388783,35.565342,44.572021,40.590752,41.680893,
                 33.856892,44.299782,35.747845,31.054487,40.150032,44.045876,37.769337,47.400902,
                 38.491226,44.268543,42.755966],
    "Longitude": [-86.791130,-152.404419,-111.431221,-92.373123,-119.681564,-105.311104,
                  -72.755371,-75.507141,-77.026817,-81.686783,-83.643074,-157.498337,
                  -114.478828,-88.986137,-86.258278,-93.210526,-96.726486,-84.670067,
                  -91.867805,-69.381927,-76.802101,-71.530106,-84.536095,-93.900192,
                  -89.678696,-92.288368,-110.454353,-98.268082,-117.055374,-71.563896,
                  -74.521011,-106.248482,-74.948051,-79.806419,-99.784012,-82.764915,
                  -96.928917,-122.070938,-77.209755,-71.511780,-80.945007,-99.438828,
                  -86.692345,-97.563461,-111.862434,-72.710686,-78.169968,-121.490494,
                  -80.954453,-89.616508,-107.302490]
})

# ----------------- FUNCTIONS -----------------

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    '''
    Load the app-ready gzipped CSV and apply basic typing/cleanup.

    - Converts Year, Hour24, State Code to numeric (nullable Int)
    - Normalizes Weekday to Mon–Sun ordered categorical
    - Adds 'State USPS' (2-letter) from numeric state codes (FIPS mapping)
    '''
    df = pd.read_csv(path, compression="gzip", low_memory=False)

    # typing cleanup
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Hour24"] = pd.to_numeric(df["Hour24"], errors="coerce").astype("Int64")
    df["State Code"] = pd.to_numeric(df["State Code"], errors="coerce").astype("Int64")

    # weekday order
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if "Weekday" in df.columns:
        mapping = {
            "Monday":"Mon","Tuesday":"Tue","Wednesday":"Wed","Thursday":"Thu",
            "Friday":"Fri","Saturday":"Sat","Sunday":"Sun"
        }
        df["Weekday"] = df["Weekday"].replace(mapping)
        df["Weekday"] = pd.Categorical(df["Weekday"], categories=weekday_order, ordered=True)

    # add USPS abbreviation (needed for dropdown + plotly choropleth)
    df["State USPS"] = df["State Code"].map(
        lambda x: FIPS_TO_USPS.get(int(x)) if pd.notna(x) else None
    )

    return df

def make_state_metric(df_all: pd.DataFrame, year_range, metric: str) -> pd.DataFrame:
    '''
    Docstring for make_state_metric
    
    Compute state-level totals for a given year range and metric.

    Returns a DataFrame with:
      - State USPS (2-letter)
      - value (int): Incidents (count) or sum of Killed/Injured
    '''
    y0, y1 = year_range
    base = df_all[df_all["Year"].between(y0, y1)].copy()

    if metric == "Incidents":
        out = base.groupby("State USPS").size().reset_index(name="value")
    elif metric == "Killed":
        out = base.groupby("State USPS")["Total Killed Form 57"].sum(min_count=1).reset_index(name="value")
    elif metric == "Injured":
        out = base.groupby("State USPS")["Total Injured Form 57"].sum(min_count=1).reset_index(name="value")
    else:
        raise ValueError("Unknown metric")

    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0).round().astype("int64")
    return out


def metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    '''
    Return a per-row numeric series for the selected metric.

    - Incidents -> 1 per row
    - Killed -> 'Total Killed Form 57'
    - Injured -> 'Total Injured Form 57'
    '''
    if metric == "Incidents":
        return pd.Series(1, index=df.index)
    if metric == "Killed":
        return pd.to_numeric(df["Total Killed Form 57"], errors="coerce").fillna(0)
    if metric == "Injured":
        return pd.to_numeric(df["Total Injured Form 57"], errors="coerce").fillna(0)
    return pd.Series(1, index=df.index)


def plot_weekday_hour_heatmap(df: pd.DataFrame, metric: str):
    '''
    Plot a Weekday × Hour24 heatmap for the selected metric.

    Heatmap cell values are:
      - Incidents: count of rows
      - Killed/Injured: sum over rows
    Weekdays are ordered Mon–Sun; hours are fixed 0–23.
    '''
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    idx = pd.Index(weekday_order, name="Weekday")
    cols = pd.Index(list(range(24)), name="Hour24")

    if df.empty:
        heat = pd.DataFrame(0, index=idx, columns=cols)
    else:
        tmp = df.dropna(subset=["Weekday", "Hour24"]).copy()
        tmp["Hour24"] = tmp["Hour24"].astype(int)

        if metric == "Incidents":
            tmp["_val"] = 1
        elif metric == "Killed":
            tmp["_val"] = pd.to_numeric(tmp["Total Killed Form 57"], errors="coerce").fillna(0)
        elif metric == "Injured":
            tmp["_val"] = pd.to_numeric(tmp["Total Injured Form 57"], errors="coerce").fillna(0)
        else:
            tmp["_val"] = 1

        heat = (
            tmp.groupby(["Weekday", "Hour24"])["_val"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=idx, columns=cols, fill_value=0)
        )

    heat = heat.round().astype(int)
    # plot
    fig_w = 12
    fig_h = fig_w * (7 / 24)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat.values, aspect="equal", cmap="YlOrRd", origin="upper")

    ax.set_title(f"{metric} by Weekday × Hour (0–23)")
    ax.set_xlabel("Hour24")
    ax.set_ylabel("Weekday")

    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(weekday_order)

    ax.set_xticks([x - 0.5 for x in range(1, 24)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 7)], minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(metric)

    st.pyplot(fig, clear_figure=True)


def choropleth_selected_highlight(df_all: pd.DataFrame, picked_usps: list[str], year_range, metric: str):
    '''
    Plot a USA choropleth for the selected metric.

    - All states are shown for geographic context.
    - If any states are selected, only those are colored (others blank).
    - Hover shows state + metric value; state abbreviations are overlaid as text.
    '''
    all_states = pd.DataFrame({"State USPS": list(FIPS_TO_USPS.values())})

    metric_df = make_state_metric(df_all, year_range, metric)
    m = all_states.merge(metric_df, on="State USPS", how="left")

    # if some selected: blank out all unselected states
    if picked_usps:
        m.loc[~m["State USPS"].isin(picked_usps), "value"] = None

    y0, y1 = year_range
    sel_txt = ", ".join(picked_usps) if picked_usps else "All States"
    title = f"{metric} by State: {sel_txt} | {y0} to {y1}"

    fig = px.choropleth(
        m,
        locations="State USPS",
        locationmode="USA-states",
        color="value",
        scope="usa",
        color_continuous_scale="Reds",
    )

    # hover shows only state + value
    fig.update_traces(
    hovertemplate=f"<b>%{{location}}</b><br>{metric}: %{{z:,.0f}}<extra></extra>"
    )
    fig.update_coloraxes(colorbar_tickformat=",d")


    # --- Static state labels (ALL states), no hover ---
    # CENTROIDS columns: state, Latitude, Longitude
    fig.add_trace(go.Scattergeo(
        locationmode="USA-states",
        lon=CENTROIDS["Longitude"],
        lat=CENTROIDS["Latitude"],
        text=CENTROIDS["state"],      # DE, CA, ...
        mode="text",
        textfont=dict(size=10, color="black"),
        hoverinfo="skip",             # <-- removes lat/lon hover completely
        showlegend=False
    ))

    fig.update_traces(marker_line_width=0.5, selector=dict(type="choropleth"))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=50, b=0))

    st.plotly_chart(fig, use_container_width=True)


# ----------------- APP -----------------
df = load_data(DATA_PATH)

# Sidebar controls
st.sidebar.header("Core Filters")

# year slider
if df["Year"].notna().any():
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
else:
    year_min, year_max = 2000, 2025

year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

# state multiselect: "State Name (DE)" labels
state_ref = df[["State Name", "State USPS"]].dropna().drop_duplicates().copy()
state_ref["label"] = state_ref["State Name"].astype(str) + " (" + state_ref["State USPS"].astype(str) + ")"
state_labels = sorted(state_ref["label"].unique())

picked_state_labels = st.sidebar.multiselect("State(s)", state_labels, default=[])

# extract "DE" from "(DE)"
picked_state_usps = [s.split("(")[-1].replace(")", "").strip() for s in picked_state_labels]

# dynamic title (state codes only are shown in the title to avoid messy interface)
code_text = ", ".join(picked_state_usps) if picked_state_usps else "All States"
st.title(f"RR Grade Crossing Accidents: {code_text} | {year_range[0]}–{year_range[1]}")

# filter df for non-map charts (year + selected states)
df_f = df[df["Year"].between(year_range[0], year_range[1])].copy()
if picked_state_usps:
    df_f = df_f[df_f["State USPS"].isin(picked_state_usps)]

st.caption(f"Loaded: {len(df):,} rows. After filters: {len(df_f):,} rows.")

# chart chooser on the sidebar
st.sidebar.divider()
demo_chart = st.sidebar.selectbox(
    "Which main chart?",
    ["Choropleth", "Week-Hour Heatmap", "Weekdays", "Top States"]
)

# global metric (applies to all charts)
metric = st.sidebar.selectbox("Metric", ["Incidents", "Killed", "Injured"])

# top-N control only for Top States
top_n = None
if demo_chart == "Top States":
    top_n = st.sidebar.selectbox("Top N states", [5, 10, 15, 20])

st.divider()

# --- Main chart area ---
if demo_chart == "Choropleth":
    st.subheader("Choropleth")
    choropleth_selected_highlight(df, picked_state_usps, year_range, metric)

elif demo_chart == "Week-Hour Heatmap":
    st.subheader("Week-Hour Heatmap")
    plot_weekday_hour_heatmap(df_f, metric)

elif demo_chart == "Weekdays":
    st.subheader("Weekdays")
    tmp = df_f.dropna(subset=["Weekday"]).copy()
    tmp["_val"] = metric_series(tmp, metric)
    wk = tmp.groupby("Weekday")["_val"].sum().reindex(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    st.bar_chart(wk)

elif demo_chart == "Top States":
    st.subheader(f"Top {top_n} States")
    base = df[df["Year"].between(year_range[0], year_range[1])].copy()
    if picked_state_usps:
        base = base[base["State USPS"].isin(picked_state_usps)]

    base["_val"] = metric_series(base, metric)

    top = (
        base.groupby("State Name")["_val"]
        .sum()
        .sort_values(ascending=False)
        .head(int(top_n))
    )
    top = top.round().astype(int)


    # Horizontal bar: #1 at top
    top_plot = top.sort_values(ascending=True)  # smallest bottom, largest top

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_plot))))
    ax.barh(top_plot.index, top_plot.values)
    ax.set_xlabel(metric)
    ax.set_ylabel("State")
    ax.set_title(f"Top {top_n} States by {metric} | {year_range[0]}–{year_range[1]}")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


st.divider()
st.subheader("Sample of selected data")

show_sample = st.checkbox("Show sample rows", value=False)

if show_sample:
    n_rows = st.number_input("Rows to show", min_value=5, max_value=200, value=25, step=5)
    st.dataframe(df_f.head(int(n_rows)), use_container_width=True)