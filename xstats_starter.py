#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Starter EDA + visualization script for MLB batted-ball expected stats (xBA/xSLG-style)
# Focus: exit velocity (launch_speed) & launch angle (launch_angle)
#
# Usage:
#   python3 xstats_starter.py --csv /path/to/mlb_pitch_data_2020_2024.csv --outdir ./outputs
#

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------

def to_bool(x):
    """Parse different truthy/falsey formats into bool."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true","t","1","yes","y"}: return True
        if s in {"false","f","0","no","n"}: return False
    return False

def map_event_to_bases(event_type, is_out_flag):
    """Map event string to total bases (0-4)."""
    if pd.isna(event_type): return 0
    e = str(event_type).strip().lower()
    if e == "single": return 1
    if e == "double": return 2
    if e == "triple": return 3
    if e in {"home_run","home run","homerun","hr"}: return 4
    return 0 if is_out_flag else 0

def map_bases_to_label(tb):
    return {0:"Out",1:"1B",2:"2B",3:"3B",4:"HR"}.get(int(tb),"Other")

def binned_stat_2d(x,y,values,xbins,ybins,stat="mean"):
    """Compute 2D binned statistic (mean)."""
    x_idx = np.digitize(x, xbins) - 1
    y_idx = np.digitize(y, ybins) - 1
    valid = (x_idx>=0)&(x_idx<len(xbins)-1)&(y_idx>=0)&(y_idx<len(ybins)-1)
    x_idx,y_idx,values = x_idx[valid], y_idx[valid], values[valid]
    grid = np.full((len(ybins)-1,len(xbins)-1),np.nan)
    counts = np.zeros_like(grid,dtype=int)
    for i,j,v in zip(y_idx,x_idx,values):
        if np.isnan(v): continue
        if np.isnan(grid[i,j]):
            grid[i,j] = v; counts[i,j] = 1
        else:
            grid[i,j] += v; counts[i,j] += 1
    if stat=="mean":
        with np.errstate(invalid="ignore"):
            grid = grid / np.where(counts==0,1,counts)
    return grid,counts

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="./outputs")
    parser.add_argument("--ev_col", default="launch_speed")
    parser.add_argument("--la_col", default="launch_angle")
    parser.add_argument("--inplay_col", default="in_play")
    parser.add_argument("--event_col", default="event_type")
    parser.add_argument("--isout_col", default="is_out")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    if args.event_col not in df.columns and "event" in df.columns:
        args.event_col = "event"

    if args.inplay_col in df.columns:
        df[args.inplay_col] = df[args.inplay_col].apply(to_bool)
    else:
        df[args.inplay_col] = True
    if args.isout_col in df.columns:
        df[args.isout_col] = df[args.isout_col].apply(to_bool)
    else:
        df[args.isout_col] = False

    keep = df[args.inplay_col].fillna(False) & df[args.ev_col].notna() & df[args.la_col].notna()
    bbe = df.loc[keep,[args.ev_col,args.la_col,args.event_col,args.isout_col]].copy()

    bbe["total_bases"] = [map_event_to_bases(ev,is_out) for ev,is_out in zip(bbe[args.event_col],bbe[args.isout_col])]
    bbe["is_hit"] = bbe["total_bases"].gt(0).astype(int)
    bbe["outcome"] = bbe["total_bases"].apply(map_bases_to_label)

    print("Batted balls kept:",len(bbe))
    print(bbe["outcome"].value_counts())

    # Scatter
    fig,ax = plt.subplots(figsize=(8,6))
    for label,sub in bbe.groupby("outcome"):
        ax.scatter(sub[args.ev_col],sub[args.la_col],s=6,alpha=0.4,label=label)
    ax.set_xlabel("Exit Velocity (mph)"); ax.set_ylabel("Launch Angle (deg)")
    ax.legend(markerscale=2,fontsize=8); ax.set_title("EV vs LA by Outcome")
    fig.tight_layout(); fig.savefig(outdir/"scatter_ev_la.png",dpi=150)

    # Heatmaps
    ev_bins = np.linspace(40,120,41); la_bins = np.linspace(-20,60,41)
    xba_grid,_ = binned_stat_2d(bbe[args.ev_col].to_numpy(),bbe[args.la_col].to_numpy(),bbe["is_hit"].astype(float).to_numpy(),ev_bins,la_bins)
    xslg_grid,_ = binned_stat_2d(bbe[args.ev_col].to_numpy(),bbe[args.la_col].to_numpy(),bbe["total_bases"].astype(float).to_numpy(),ev_bins,la_bins)

    for grid,title,fname in [(xba_grid,"xBA (hit prob)","heatmap_xba.png"),(xslg_grid,"xSLG (expected TB)","heatmap_xslg.png")]:
        fig,ax = plt.subplots(figsize=(7,6))
        im=ax.imshow(grid,origin="lower",extent=(ev_bins[0],ev_bins[-1],la_bins[0],la_bins[-1]),aspect="auto")
        ax.set_xlabel("Exit Velocity (mph)"); ax.set_ylabel("Launch Angle (deg)")
        ax.set_title(title); fig.colorbar(im,ax=ax)
        fig.tight_layout(); fig.savefig(outdir/fname,dpi=150)

if __name__=="__main__":
    main()
