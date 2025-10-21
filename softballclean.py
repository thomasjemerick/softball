import pandas as pd

df = pd.read_csv("mlb_pitch_data_2020_2024.csv", low_memory=False)

df["in_play"] = df["in_play"].astype(str).str.upper().eq("TRUE") | df["in_play"].eq(1)
bbe = df[df["in_play"] & df["launch_speed"].notna() & df["launch_angle"].notna()].copy()

bbe["event_clean"] = bbe["event_type"].fillna(bbe["event"]).str.lower()

event_map = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
}
bbe["total_bases"] = bbe["event_clean"].map(event_map).fillna(0).astype(int)

labels = {0:"Out", 1:"1B", 2:"2B", 3:"3B", 4:"HR"}
bbe["outcome"] = bbe["total_bases"].map(labels)

clean = bbe[["game_date","batter_id","batter_name",
             "pitcher_id","pitcher_name",
             "launch_speed","launch_angle",
             "event_clean","total_bases","outcome"]]

print(clean.head())


import numpy as np
import pandas as pd

# --- 1) Build EV–LA probability surface (with Laplace smoothing) ---

def build_surface(bbe, ev_bins=None, la_bins=None, alpha=1.0, woba_weights=None):
    """
    Returns dict with per-bin probabilities p0..p4, plus xBA, xSLG, xwOBA, and metadata (bins).
    alpha: Laplace smoothing (1.0 is safe); woba_weights: dict {1: w1B, 2: w2B, 3: w3B, 4: wHR}
    """
    if ev_bins is None: ev_bins = np.linspace(40, 120, 41)   # 2 mph bins
    if la_bins is None: la_bins = np.linspace(-20, 60, 41)   # 2° bins
    if woba_weights is None:
        # Example weights (update to your season if needed)
        woba_weights = {1: 0.87, 2: 1.24, 3: 1.56, 4: 2.01}

    ev = bbe["launch_speed"].to_numpy()
    la = bbe["launch_angle"].to_numpy()
    tb = bbe["total_bases"].astype(int).to_numpy()  # 0..4

    x_idx = np.digitize(ev, ev_bins) - 1
    y_idx = np.digitize(la, la_bins) - 1
    valid = (x_idx >= 0) & (x_idx < len(ev_bins)-1) & (y_idx >= 0) & (y_idx < len(la_bins)-1)
    x_idx, y_idx, tb = x_idx[valid], y_idx[valid], tb[valid]

    # counts[y, x, outcome]
    counts = np.zeros((len(la_bins)-1, len(ev_bins)-1, 5), dtype=np.int32)
    np.add.at(counts, (y_idx, x_idx, tb), 1)

    # Laplace smoothing across outcomes
    smoothed = counts + alpha
    denom = smoothed.sum(axis=2, keepdims=True)  # sum over outcomes
    probs = smoothed / denom

    p0, p1, p2, p3, p4 = [probs[:, :, k] for k in range(5)]
    xBA  = p1 + p2 + p3 + p4
    xSLG = 1*p1 + 2*p2 + 3*p3 + 4*p4
    xwOBA = (woba_weights[1]*p1 + woba_weights[2]*p2 +
             woba_weights[3]*p3 + woba_weights[4]*p4)

    surface = {
        "ev_bins": ev_bins, "la_bins": la_bins,
        "p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4,
        "xBA": xBA, "xSLG": xSLG, "xwOBA": xwOBA,
        "woba_weights": woba_weights
    }
    return surface

# --- 2) Attach per-ball expected stats via bin lookup ---

def attach_expected(bbe, surface):
    ev_bins, la_bins = surface["ev_bins"], surface["la_bins"]

    x_idx = np.digitize(bbe["launch_speed"].to_numpy(), ev_bins) - 1
    y_idx = np.digitize(bbe["launch_angle"].to_numpy(), la_bins) - 1
    valid = (x_idx >= 0) & (x_idx < len(ev_bins)-1) & (y_idx >= 0) & (y_idx < len(la_bins)-1)

    # default zeros
    bbe = bbe.copy()
    for col in ["p_out","p_1B","p_2B","p_3B","p_HR","xBA_evla","xSLG_evla","xwOBA_evla"]:
        bbe[col] = np.nan

    # fill only valid rows
    bbe.loc[valid, "p_out"] = surface["p0"][y_idx[valid], x_idx[valid]]
    bbe.loc[valid, "p_1B"]  = surface["p1"][y_idx[valid], x_idx[valid]]
    bbe.loc[valid, "p_2B"]  = surface["p2"][y_idx[valid], x_idx[valid]]
    bbe.loc[valid, "p_3B"]  = surface["p3"][y_idx[valid], x_idx[valid]]
    bbe.loc[valid, "p_HR"]  = surface["p4"][y_idx[valid], x_idx[valid]]
    bbe.loc[valid, "xBA_evla"]   = surface["xBA"][y_idx[valid],  x_idx[valid]]
    bbe.loc[valid, "xSLG_evla"]  = surface["xSLG"][y_idx[valid], x_idx[valid]]
    bbe.loc[valid, "xwOBA_evla"] = surface["xwOBA"][y_idx[valid],x_idx[valid]]

    return bbe

# --- 3) Aggregate to players (hitters & pitchers) ---

def aggregate_players(bbe_expected, who="batter"):
    """
    who: 'batter' or 'pitcher'
    Returns per-player table with actual vs expected over batted balls.
    """
    if who == "batter":
        id_col, name_col = "batter_id", "batter_name"
    else:
        id_col, name_col = "pitcher_id", "pitcher_name"

    df = bbe_expected.copy()
    df["is_hit"] = (df["total_bases"] > 0).astype(int)

    agg = df.groupby([id_col, name_col], dropna=False).agg(
        BBE=("total_bases", "size"),
        # actuals over BBE (not PA): BA_on_contact and SLG_on_contact
        ACT_BA=("is_hit", "mean"),
        ACT_SLG=("total_bases", "mean"),
        EXP_xBA=("xBA_evla", "mean"),
        EXP_xSLG=("xSLG_evla", "mean"),
        EXP_xwOBA=("xwOBA_evla", "mean"),
    ).reset_index()

    # simple residuals (actual - expected)
    agg["BA_diff"]  = agg["ACT_BA"]  - agg["EXP_xBA"]
    agg["SLG_diff"] = agg["ACT_SLG"] - agg["EXP_xSLG"]
    return agg.sort_values("BBE", ascending=False)

# --- Run it ---

surface = build_surface(bbe, alpha=1.0)                 # build probability surface
bbe_expected = attach_expected(bbe, surface)            # per-ball expected stats
batters = aggregate_players(bbe_expected, "batter")     # per-batter table
pitchers = aggregate_players(bbe_expected, "pitcher")   # per-pitcher table

# Save outputs for tomorrow's deck
bbe_expected[[
    "game_date","batter_name","pitcher_name","launch_speed","launch_angle",
    "total_bases","xBA_evla","xSLG_evla","xwOBA_evla"
]].to_csv("per_ball_expected.csv", index=False)

batters.to_csv("batters_expected.csv", index=False)
pitchers.to_csv("pitchers_expected.csv", index=False)
