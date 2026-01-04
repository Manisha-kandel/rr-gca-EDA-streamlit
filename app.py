import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="RR Grade Crossing Accident EDA", layout="wide")

DATA_PATH = "data/rr_grade_crossing_accident_data_app_ready.csv.gz"

#these are copied from the .ipynb notebook from bootcamp project, will be updated to function next time to do end-to-end work
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", low_memory=False)

    # basic typing cleanup
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    if "Hour24" in df.columns:
        df["Hour24"] = pd.to_numeric(df["Hour24"], errors="coerce").astype("Int64")

    # weekday order
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if "Weekday" in df.columns:
        mapping = {
            "Monday":"Mon","Tuesday":"Tue","Wednesday":"Wed","Thursday":"Thu",
            "Friday":"Fri","Saturday":"Sat","Sunday":"Sun"
        }
        df["Weekday"] = df["Weekday"].replace(mapping)
        df["Weekday"] = pd.Categorical(df["Weekday"], categories=weekday_order, ordered=True)

    return df

def filter_data(df: pd.DataFrame, year_range, states_selected):
    y0, y1 = year_range
    out = df.copy()
    if "Year" in out.columns:
        out = out[out["Year"].between(y0, y1)]
    if states_selected and ("State Name" in out.columns):
        out = out[out["State Name"].isin(states_selected)]
    return out

def plot_weekday_hour_heatmap(df: pd.DataFrame):
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    idx = pd.Index(weekday_order, name="Weekday")
    cols = pd.Index(list(range(24)), name="Hour24")

    if df.empty:
        heat = pd.DataFrame(0, index=idx, columns=cols)
    else:
        tmp = df.dropna(subset=["Weekday", "Hour24"]).copy()
        tmp["Hour24"] = tmp["Hour24"].astype(int)

        heat = (
            tmp.groupby(["Weekday", "Hour24"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=idx, columns=cols, fill_value=0)
        )

    fig_w = 12
    fig_h = fig_w * (7/24)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat.values, aspect="equal", cmap="YlOrRd", origin="upper")

    ax.set_title("Incidents by Weekday × Hour (0–23)")
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
    cbar.set_label("Count")

    st.pyplot(fig, clear_figure=True)


# -----------------------------------------------------------------------#

# -----------------------------------------------------------------------#

#let's try some streamlit feature for practice

#title
st.title("Grade Crossing Accidents in USA")

df = load_data(DATA_PATH)

chart  = st.selectbox(
    "Choose a chart",
    ["Heatmap", "Bar diagram", "Chloropeth"]
)

#write
st.write("You picked: ", chart)

states = sorted(df["State Name"].dropna().unique())
picked_states = st.multiselect("States", states, default = states[:3])
st.write(str(len(picked_states)) + " states were selected for Exploratory Data Analysis")

#sidebar
st.sidebar.header('Filters')

#slider
if "Year" in df.columns and df['Year'].notna().any():
    year_min = int(df['Year'].min())
    year_max = int(df['Year'].max())
else:
    year_min, year_max = 2000, 2020

year_range = st.sidebar.slider("Select year range of interest", year_min, year_max, (year_min, year_max))
st.sidebar.write("The selected range of years is: " + str(year_range) )

