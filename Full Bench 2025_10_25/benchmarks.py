import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

# ---------------------------------------------------------------------
#                         Get Criterion Data 
# ---------------------------------------------------------------------

def find_estimate_files(base_dir: str) -> List[Path]:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Criterion directory not found: {base_dir}")
    return list(base.rglob("estimates.json"))

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def infer_benchmark_name(est_path: Path, base_dir: str) -> str:
    bench_json = est_path.parent.parent / "benchmark.json"
    if bench_json.exists():
        data = load_json(bench_json)
        if data:
            full_id = data.get("full_id")
            if isinstance(full_id, str) and full_id.strip():
                return full_id.strip()
            group_id = data.get("group_id", "")
            function_id = data.get("function_id", "")
            value_str = data.get("value_str", "")
            parts = [p for p in [group_id, function_id, value_str] if p]
            if parts:
                return "/".join(parts)
    try:
        rel = est_path.parent.relative_to(Path(base_dir))
        parts = list(rel.parts)
        if parts and parts[-1] in ("new", "base"):
            parts = parts[:-1]
        return "/".join(parts)
    except Exception:
        return est_path.parent.name

def parse_dataset_size(benchmark_name: str) -> Optional[int]:
    m = re.search(r"/(\d+)(?:$|[^0-9])", benchmark_name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def parse_algorithm(benchmark_name: str) -> str:
    parts = benchmark_name.split("/")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]

def parse_threads(benchmark_name: str) -> int:
    m = re.search(r"_t(\d+)", benchmark_name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return 1
    return 1

def infer_time_unit_ms(bench_json_path: Path) -> float:
    unit = "ns"
    data = load_json(bench_json_path)
    if data:
        unit = data.get("unit", unit) or unit
    unit = unit.lower()
    if unit in ("ns", "nanosecond", "nanoseconds"):
        return 1e-6
    if unit in ("us", "microsecond", "microseconds"):
        return 1e-3
    if unit in ("ms", "millisecond", "milliseconds"):
        return 1.0
    if unit in ("s", "sec", "second", "seconds"):
        return 1e3
    return 1e-6

def extract_throughput_elements(bench_json_path: Path) -> Optional[float]:
    data = load_json(bench_json_path)
    if not data:
        return None
    thr = data.get("throughput")
    if isinstance(thr, dict):
        if str(thr.get("type","")).lower() == "elements":
            val = thr.get("value")
            try:
                return float(val) if val is not None else None
            except Exception:
                return None
    return None

def extract_mean_from_estimates(estimates_path: Path) -> Optional[float]:
    data = load_json(estimates_path)
    if not data:
        return None
    mean = data.get("mean", {})
    pe = mean.get("point_estimate")
    try:
        return float(pe)
    except Exception:
        return None

def build_rows(base_dir: str) -> pd.DataFrame:
    est_files = find_estimate_files(base_dir)
    latest_by_name: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    for est in est_files:
        bench_name = infer_benchmark_name(est, base_dir)
        bench_dir = est.parent
        bench_json = bench_dir.parent / "benchmark.json"

        time_unit_to_ms = infer_time_unit_ms(bench_json)
        mean_est = extract_mean_from_estimates(est)
        if mean_est is None:
            continue
        mean_ms = mean_est * time_unit_to_ms

        # Prefer Criterion throughput; fallback to parsed dataset size
        elements_per_iter = extract_throughput_elements(bench_json)
        ds_size = parse_dataset_size(bench_name)
        if elements_per_iter is None and ds_size is not None:
            elements_per_iter = float(ds_size)

        throughput_melems_s = None
        if elements_per_iter is not None and mean_ms and mean_ms > 0:
            eps = elements_per_iter / (mean_ms / 1000.0)  # elements per second
            throughput_melems_s = eps / 1e6

        algo = parse_algorithm(bench_name)
        threads = parse_threads(bench_name)
        mtime = est.stat().st_mtime

        rec = {
            "Benchmark_Name": bench_name,
            "Dataset_Size": ds_size if ds_size is not None else None,
            "Algorithm": algo,
            "Threads": threads,
            "Time_Mean_ms": float(mean_ms),
            "Throughput_Melem_s": float(throughput_melems_s) if throughput_melems_s is not None else float("nan"),
            "mtime": mtime,
        }

        prev = latest_by_name.get(bench_name)
        if prev is None or mtime > prev[0]:
            latest_by_name[bench_name] = (mtime, rec)

    rows = [v for (_, v) in latest_by_name.values()]
    df = pd.DataFrame(rows)
    df = df.sort_values(["Algorithm", "Dataset_Size", "Threads", "Benchmark_Name"], na_position="last").reset_index(drop=True)

    # Enforce dtypes
    df["Dataset_Size"] = pd.to_numeric(df["Dataset_Size"], errors="coerce").astype("Int64")
    df["Threads"] = pd.to_numeric(df["Threads"], errors="coerce").astype("Int64")
    df["Time_Mean_ms"] = pd.to_numeric(df["Time_Mean_ms"], errors="coerce")
    df["Throughput_Melem_s"] = pd.to_numeric(df["Throughput_Melem_s"], errors="coerce")

    df = df[~df["Benchmark_Name"].str.endswith("/change")].reset_index(drop=True)

    df["Benchmark_Group"] = df["Benchmark_Name"].str.split("/").str[0]

    return df

def get_last_benchmark(base_dir, run_gap=pd.Timedelta("90min"), verbose=True):
    """
    Return only the most recent benchmark for each bench group from the most recent 
    Criterion run, detected by a time-gap clustering over the `mtime` column (epoch seconds).
    """
    df2 = build_rows(base_dir).copy()   # use the function arg, not a global
    if "mtime" not in df2.columns:
        raise KeyError("Expected a 'mtime' column (epoch seconds). Make sure build_rows() keeps it.")
    
    # Normalize and sort by datetime
    df2["_mtime_dt"] = pd.to_datetime(pd.to_numeric(df2["mtime"], errors="coerce"), unit="s", utc=True)
    df2 = df2.sort_values("_mtime_dt").reset_index(drop=True)
    
    if df2.empty:
        return df2
    
    # Split into runs: whenever time gap > run_gap, start a new run id
    run_id = (df2["_mtime_dt"].diff().gt(run_gap)).cumsum()
    latest_run = int(run_id.iloc[-1])
    
    # Keep only the most recent run cluster
    #df_last = df2.loc[run_id == latest_run].copy()
    df_last = df2.copy()
    
    # Get the most recent benchmark for each bench group
    # Assumes there's a column identifying the bench group (e.g., 'group', 'benchmark', or 'name')
    # Adjust the groupby column name as needed for your data structure
    group_col = None
    for col in ['group', 'benchmark', 'name', 'bench_group', 'function']:
        if col in df_last.columns:
            group_col = col
            break
    
    if group_col is not None:
        # Keep only the most recent entry for each group
        df_last = df_last.sort_values("_mtime_dt").groupby(group_col, as_index=False).last()
    
    if verbose:
        start = df_last["_mtime_dt"].min()
        end   = df_last["_mtime_dt"].max()
        print(f"Latest run window: {start} → {end} (rows={len(df_last)})")
    
    # If you don't want the helper datetime column in the result, drop it:
    # df_last = df_last.drop(columns=["_mtime_dt"])
    return df_last
    
    
# ---------------------------------------------------------------------
#                         Serial Benche Plots
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.express as px

def plot_bench(df, benchmark_group: str, title: str, bench_type: str = "throughput",
               min_size: float | None = None,
               max_size: float | None = None,
               rescale_y: bool = False,
               log_y: bool = False,
               height: int = 600):
    """
    Plot a Criterion benchmark group (insertion/lookup/...) with nice styling.
    Args:
      df: DataFrame with columns Benchmark_Name, Dataset_Size, Algorithm, Threads,
          Time_Mean_ms, Throughput_Melem_s (and optionally others).
      benchmark_group: e.g. "lookup", "insertion", "removal".
      title: chart title.
      bench_type: "throughput" (Melem/s) or "time" (ms).
      min_size, max_size: optional Dataset_Size filters (e.g., min_size=1e6 to focus on >=1M).
      rescale_y: if True, y-axis range is set from the filtered data ±10% padding.
      log_y: if True, plot y axis on log scale.
      height: figure height in pixels (default: 600).
    """
    if bench_type not in {"throughput", "time"}:
        raise ValueError("bench_type must be 'throughput' or 'time'")
    y_col = "Throughput_Melem_s" if bench_type == "throughput" else "Time_Mean_ms"
    y_label = "Throughput (Melem/s)" if bench_type == "throughput" else "Time (ms)"
    
    # Add log scale notation if enabled
    if log_y:
        y_label += " (log scale)"
    
    # base filter
    data = df[df["Benchmark_Name"].str.startswith(f"{benchmark_group}/")].copy()
    data = data[~data["Benchmark_Name"].str.endswith("/change")]
    data["Dataset_Size"] = pd.to_numeric(data["Dataset_Size"], errors="coerce")
    data = data.dropna(subset=["Dataset_Size", y_col])
    
    # single-thread view by default for fair comparison
    data = data[data["Threads"].fillna(1).astype(int) == 1]
    
    # optional size window
    if min_size is not None:
        data = data[data["Dataset_Size"] >= float(min_size)]
    if max_size is not None:
        data = data[data["Dataset_Size"] <= float(max_size)]
    
    # legend order and color mapping
    order = ["std_hashmap", "ahashmap", "simple_chained_hash_map", "anchor_hca", "direct_anchor_hca"]
    present = list(data["Algorithm"].dropna().unique())
    algo_order = [a for a in order if a in present] + [a for a in present if a not in order]
    data["Algorithm"] = pd.Categorical(data["Algorithm"], categories=algo_order, ordered=True)
    
    # Define consistent color map for algorithms
    color_map = {
        "std_hashmap": "#636EFA",        # blue
        "ahashmap": "#EF553B",           # red
        "simple_chained_hash_map": "#00CC96",  # green/teal
        "anchor_hca": "#AB63FA",         # purple
        "direct_anchor_hca": "#FFA15A"   # orange
    }
    
    # figure
    fig = px.line(
        data.sort_values(["Algorithm", "Dataset_Size"]),
        x="Dataset_Size",
        y=y_col,
        color="Algorithm",
        markers=True,
        template="plotly_white",
        category_orders={"Algorithm": algo_order},
        color_discrete_map=color_map,  # Added color map
        title=title,
        labels={"Dataset_Size": "Dataset Size", y_col: y_label},
        height=height,
    )
    
    fig.update_traces(
        mode="lines+markers",
        line={"width": 3, "shape": "spline", "smoothing": 0.8},
        marker={"size": 7}
    )
    
    # axes
    fig.update_layout(
        hovermode="x unified",
        font={"size": 14},
        margin=dict(l=70, r=20, t=60, b=55),
        legend_title_text="",
    )
    
    fig.update_xaxes(
        type="log",
        tickvals=[1e4, 1e5, 1e6, 1e7, 2.5e7],
        ticktext=["10K", "100K", "1M", "10M", "25M"],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)"
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        type="log" if log_y else "linear"
    )
    
    # Add explicit tick formatting for log scale
    if log_y:
        # Use explicit tick values that work well for your data range
        fig.update_yaxes(
            tickvals=[1, 2, 5, 10, 20, 50, 100],
            ticktext=["1", "2", "5", "10", "20", "50", "100"]
        )
    
    # rescale y to filtered subset (only for linear scale)
    if rescale_y and not log_y and not data.empty:
        y = data[y_col].astype(float).to_numpy()
        lo, hi = np.nanmin(y), np.nanmax(y)
        pad = (hi - lo) * 0.10 if hi > lo else max(0.25, hi * 0.1)
        fig.update_yaxes(range=[lo - pad, hi + pad])
    
    return fig
    
# ---------------------------------------------------------------------
#                         Parallel Bench Plots
# ---------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import plotly.express as px

def _prep_parallel(df, benchmark_group: str, y_col: str,
                   min_size: float | None = None, max_size: float | None = None):
    """Common filter/cleanup for parallel benches."""
    data = df.copy()
    data = data[data["Benchmark_Name"].str.startswith(f"{benchmark_group}/")]
    data = data[~data["Benchmark_Name"].str.endswith("/change")]
    data = data[data["Benchmark_Name"].str.contains("_par_", na=False)]  # only parallel variants
    data["Dataset_Size"] = pd.to_numeric(data["Dataset_Size"], errors="coerce")
    data = data.dropna(subset=["Dataset_Size", y_col, "Threads"])
    if min_size is not None:
        data = data[data["Dataset_Size"] >= float(min_size)]
    if max_size is not None:
        data = data[data["Dataset_Size"] <= float(max_size)]
    # Derive an algorithm 'family' (strip _par/_tN tail) for nicer legends
    data["Algorithm_Family"] = data["Algorithm"].str.replace(r"_par(?:_separated)?_t\d+$", "", regex=True)
    # Make threads categorical for consistent ordering
    data["Threads"] = pd.to_numeric(data["Threads"], errors="coerce").astype("Int64")
    return data

def plot_parallel_facets_by_threads(df, benchmark_group: str, title: str,
                                    bench_type: str = "throughput",
                                    min_size: float | None = None, max_size: float | None = None):
    """
    View #1: Compare algorithms at each Thread count across dataset sizes.
    Facets = Threads, X = Dataset_Size (log), Y = throughput|time, color = Algorithm_Family.
    """
    y_col = "Throughput_Melem_s" if bench_type == "throughput" else "Time_Mean_ms"
    y_label = "Throughput (Melem/s)" if bench_type == "throughput" else "Time (ms)"
    data = _prep_parallel(df, benchmark_group, y_col, min_size, max_size)
    if data.empty:
        raise ValueError("No matching parallel rows after filtering.")

    # Order families (only those present)
    pref = ["chained_hash_map", "concurrent_direct_anchor_hca", "anchor_hca", "direct_anchor_hca", "simple_chained_hash_map"]
    fams = list(dict.fromkeys([f for f in pref if f in data["Algorithm_Family"].unique()] +
                              [f for f in data["Algorithm_Family"].unique() if f not in pref]))
    data["Algorithm_Family"] = pd.Categorical(data["Algorithm_Family"], fams, ordered=True)

    fig = px.line(
        data.sort_values(["Threads", "Algorithm_Family", "Dataset_Size"]),
        x="Dataset_Size", y=y_col, color="Algorithm_Family", facet_col="Threads",
        facet_col_wrap=0,  # one row of facets
        markers=True, template="plotly_white",
        title=title, labels={"Dataset_Size": "Dataset Size", y_col: y_label, "Algorithm_Family": ""}
    )
    fig.update_traces(line={"width": 3, "shape": "spline", "smoothing": 0.8}, marker={"size": 7})
    fig.update_xaxes(type="log",
                     tickvals=[1e4,1e5,1e6,1e7,2.5e7],
                     ticktext=["10K","100K","1M","10M","25M"],
                     showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_layout(hovermode="x unified", font={"size":14}, margin=dict(l=70,r=20,t=60,b=55))
    return fig

def plot_parallel_strong_scaling(df, benchmark_group: str, title: str,
                                 bench_type: str = "throughput",
                                 sizes: list[float] | None = None,
                                 min_size: float | None = None, max_size: float | None = None):
    """
    View #2: Strong scaling. For each Dataset_Size (facets), X=Threads, Y=throughput|time,
    color = Algorithm_Family. Shows how adding threads scales at fixed size.
    """
    y_col = "Throughput_Melem_s" if bench_type == "throughput" else "Time_Mean_ms"
    y_label = "Throughput (Melem/s)" if bench_type == "throughput" else "Time (ms)"
    data = _prep_parallel(df, benchmark_group, y_col, min_size, max_size)
    if sizes:
        sizes = [float(s) for s in sizes]
        data = data[data["Dataset_Size"].isin(sizes)]
    if data.empty:
        raise ValueError("No matching parallel rows after filtering.")

    # Nice labels for facet titles
    def _size_label(v):
        v = float(v)
        if v >= 1e7 and abs(v-2.5e7)<1: return "25M"
        if v >= 1e7: return "10M"
        if v >= 1e6: return f"{int(v/1e6)}M"
        if v >= 1e5: return f"{int(v/1e3)}K"
        return str(int(v))
    data["SizeLabel"] = data["Dataset_Size"].map(_size_label)

    pref = ["chained_hash_map", "concurrent_direct_anchor_hca", "anchor_hca", "direct_anchor_hca", "simple_chained_hash_map"]
    fams = list(dict.fromkeys([f for f in pref if f in data["Algorithm_Family"].unique()] +
                              [f for f in data["Algorithm_Family"].unique() if f not in pref]))
    data["Algorithm_Family"] = pd.Categorical(data["Algorithm_Family"], fams, ordered=True)

    fig = px.line(
        data.sort_values(["SizeLabel","Algorithm_Family","Threads"]),
        x="Threads", y=y_col, color="Algorithm_Family", facet_col="SizeLabel",
        markers=True, template="plotly_white",
        title=title, labels={"Threads":"Threads", y_col:y_label, "Algorithm_Family":""}
    )
    fig.update_traces(line={"width":3, "shape":"spline", "smoothing":0.8}, marker={"size":7})
    fig.update_xaxes(type="linear", dtick=2, showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_layout(hovermode="x unified", font={"size":14}, margin=dict(l=70,r=20,t=60,b=55))
    return fig

def plot_parallel_speedup(df, benchmark_group: str, title: str,
                          bench_type: str = "throughput",
                          baseline="auto",  # "auto" → 1-thread of same family if available; else min-threads present
                          sizes: list[float] | None = None,
                          min_size: float | None = None, max_size: float | None = None,
                          show_efficiency: bool = False):
    """
    View #3: Speedup/Efficiency vs Threads per Dataset_Size, color by Algorithm_Family.
    Speedup is normalized to a baseline of the same family+size.
    """
    y_raw = "Throughput_Melem_s" if bench_type == "throughput" else "Time_Mean_ms"
    data = _prep_parallel(df, benchmark_group, y_raw, min_size, max_size)
    if data.empty:
        raise ValueError("No matching parallel rows after filtering.")
    if sizes:
        sizes = [float(s) for s in sizes]
        data = data[data["Dataset_Size"].isin(sizes)]

    # Try to find 1-thread baselines from non-par rows of same family
    base = df.copy()
    base = base[base["Benchmark_Name"].str.startswith(f"{benchmark_group}/")]
    base = base[~base["Benchmark_Name"].str.endswith("/change")]
    base["Dataset_Size"] = pd.to_numeric(base["Dataset_Size"], errors="coerce")
    base = base.dropna(subset=["Dataset_Size", y_raw, "Threads"])
    base["Algorithm_Family"] = base["Algorithm"].str.replace(r"_par(?:_separated)?_t\d+$", "", regex=True)

    # Build baseline map: (family, size) -> y_value at 1 thread if available
    basemap = {}
    g = base.groupby(["Algorithm_Family", "Dataset_Size"])
    for (fam, size), sub in g:
        # prefer Threads == 1
        one = sub[sub["Threads"] == 1]
        if not one.empty:
            basemap[(fam, float(size))] = float(one.iloc[0][y_raw])

    # If missing, fall back to min-thread value within parallel data for that family+size
    if baseline == "auto":
        for (fam, size), sub in data.groupby(["Algorithm_Family","Dataset_Size"]):
            key = (fam, float(size))
            if key not in basemap:
                idx = sub["Threads"].astype(float).idxmin()
                basemap[key] = float(sub.loc[idx, y_raw])

    if not basemap:
        raise ValueError("No baselines found for speedup; ensure you have either 1-thread non-par rows or parallel rows to seed baseline.")

    # Compute speedup/efficiency
    data["Baseline"] = data.apply(lambda r: basemap.get((r["Algorithm_Family"], float(r["Dataset_Size"])), np.nan), axis=1)
    if bench_type == "throughput":
        data["Speedup"] = data[y_raw] / data["Baseline"]
    else:
        # for time, lower is better: speedup = baseline_time / time
        data["Speedup"] = data["Baseline"] / data[y_raw]
    data["Efficiency"] = data["Speedup"] / data["Threads"].astype(float)

    metric = "Efficiency" if show_efficiency else "Speedup"
    y_label = metric if not show_efficiency else "Efficiency (Speedup / Threads)"

    # Facet by dataset size for clarity
    def _size_label(v):
        v = float(v)
        if v >= 1e7 and abs(v-2.5e7)<1: return "25M"
        if v >= 1e7: return "10M"
        if v >= 1e6: return f"{int(v/1e6)}M"
        if v >= 1e5: return f"{int(v/1e3)}K"
        return str(int(v))
    data["SizeLabel"] = data["Dataset_Size"].map(_size_label)

    fig = px.line(
        data.sort_values(["SizeLabel","Algorithm_Family","Threads"]),
        x="Threads", y=metric, color="Algorithm_Family", facet_col="SizeLabel",
        markers=True, template="plotly_white",
        title=title, labels={"Threads":"Threads", metric:y_label, "Algorithm_Family":""}
    )
    fig.update_traces(line={"width":3, "shape":"spline", "smoothing":0.8}, marker={"size":7})
    fig.update_xaxes(type="linear", dtick=2, showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_layout(hovermode="x unified", font={"size":14}, margin=dict(l=70,r=20,t=60,b=55))
    return fig


def plot_migration_speedup(bench_df):
    """
    Create a horizontal bar chart showing migration speedup vs rebuild performance.
    
    Parameters:
    -----------
    bench_df : pd.DataFrame
        Benchmark dataframe containing migration_vs_rebuild results
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import pandas as pd
    import plotly.graph_objects as go
    
    # Filter for migration_vs_rebuild benchmarks (100K and 1M)
    migration_data = bench_df[
        bench_df['Benchmark_Group'].isin(['migration_vs_rebuild_100000', 'migration_vs_rebuild_1000000'])
    ].copy()
    
    # Extract dataset size
    migration_data['dataset_size'] = migration_data['Benchmark_Group'].str.extract(r'(\d+)')[0]
    migration_data.loc[migration_data['dataset_size'].isna(), 'dataset_size'] = '100000'
    migration_data['dataset_size'] = migration_data['dataset_size'].astype(int)
    migration_data['size_label'] = migration_data['dataset_size'].apply(
        lambda x: f"{x//1000}K" if x < 1000000 else f"{x//1000000}M"
    )
    
    # Separate migrations and rebuilds
    migrations = migration_data[migration_data['Algorithm'].str.contains('migration')]
    rebuilds = migration_data[migration_data['Algorithm'].str.contains('rebuild')]
    
    # Calculate speedups: rebuild_time / migration_time
    speedup_results = []
    
    for size in migration_data['dataset_size'].unique():
        size_label = f"{size//1000}K" if size < 1000000 else f"{size//1000000}M"
        
        # Get migration times
        simple_mig = migrations[
            (migrations['dataset_size'] == size) & 
            (migrations['Algorithm'] == 'simple_migration')
        ]['Time_Mean_ms'].values
        chained_mig = migrations[
            (migrations['dataset_size'] == size) & 
            (migrations['Algorithm'] == 'chained_migration')
        ]['Time_Mean_ms'].values
        
        if len(simple_mig) == 0 or len(chained_mig) == 0:
            continue
        
        simple_mig_time = simple_mig[0]
        chained_mig_time = chained_mig[0]
        
        # Calculate speedups for all rebuild variants
        size_rebuilds = rebuilds[rebuilds['dataset_size'] == size]
        
        for _, row in size_rebuilds.iterrows():
            rebuild_algo = row['Algorithm'].replace('_rebuild', '').replace('_', ' ').title()
            rebuild_time = row['Time_Mean_ms']
            
            # Use appropriate migration baseline
            if 'simple' in row['Algorithm'].lower():
                baseline = 'simple_migration'
                mig_time = simple_mig_time
            elif 'chained' in row['Algorithm'].lower():
                baseline = 'chained_migration'
                mig_time = chained_mig_time
            else:
                # For std_hashmap and ahashmap, compare against fastest migration (chained)
                baseline = 'chained_migration'
                mig_time = chained_mig_time
            
            speedup = rebuild_time / mig_time
            
            speedup_results.append({
                'dataset_size': size,
                'size_label': size_label,
                'comparison': f"{rebuild_algo} vs {baseline.replace('_', ' ').title()}",
                'rebuild_algo': rebuild_algo,
                'speedup': speedup
            })
    
    speedup_df = pd.DataFrame(speedup_results)
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for different algorithms
    colors = {
        'Simple': '#e74c3c',
        'Chained': '#c0392b',
        'Std Hashmap': '#e67e22',
        'Ahashmap': '#d35400'
    }
    
    # Add bars for each dataset size
    for size_label in speedup_df['size_label'].unique():
        size_data = speedup_df[speedup_df['size_label'] == size_label].sort_values('speedup')
        
        fig.add_trace(go.Bar(
            name=size_label,
            x=size_data['speedup'],
            y=size_data['comparison'],
            orientation='h',
            marker_color=[colors.get(algo, '#95a5a6') for algo in size_data['rebuild_algo']],
            text=[f"{x:.2f}×" for x in size_data['speedup']],
            textposition='outside',
            textfont=dict(size=12),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Migration Speedup: How Much Faster vs Rebuild',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Speedup Factor (Rebuild Time / Migration Time)',
        yaxis_title='',
        width=1000,
        height=500,
        template='plotly_white',
        barmode='group',
        legend=dict(
            title='Dataset Size',
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=250, r=100, t=80, b=60)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        range=[0, max(speedup_df['speedup']) * 1.15]
    )
    
    # Add vertical line at 1.0 (break-even point) with floating annotation
    fig.add_vline(
        x=1.0, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Break-even",
        annotation_position="top",
        annotation=dict(
            yanchor="bottom",
            y=1.02,
            showarrow=False,
            font=dict(size=11, color="gray")
        )
    )
    
    return fig
    
def plot_migration_capacity_scaling(bench_df):
    """
    Create a line plot showing migration time vs dataset size for 2× and 4× capacity increases.
    
    Parameters:
    -----------
    bench_df : pd.DataFrame
        Benchmark dataframe containing migration capacity scaling results
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import pandas as pd
    import plotly.graph_objects as go
    
    # Filter for capacity scaling benchmarks
    capacity_data = bench_df[
        bench_df['Benchmark_Group'].isin(['migration_simple_chained_hash_map', 'migration_chained_hash_map'])
    ].copy()
    
    # Extract dataset size from Dataset_Size column (appears to be the multiplier, not actual size)
    # Need to extract from Benchmark_Name instead
    capacity_data['elements'] = capacity_data['Benchmark_Name'].str.extract(r'/(\d+)$')[0].astype(int)
    capacity_data['capacity_mult'] = capacity_data['Algorithm'].str.extract(r'(\d+)x')[0] + '×'
    capacity_data['variant'] = capacity_data['Benchmark_Group'].apply(
        lambda x: 'SimpleChainedHashMap' if 'simple' in x else 'ChainedHashMap'
    )
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for each line (all distinct)
    line_configs = {
        ('SimpleChainedHashMap', '2×'): {'color': '#16a085', 'dash': 'solid'},   # teal solid
        ('SimpleChainedHashMap', '4×'): {'color': '#e74c3c', 'dash': 'dash'},    # red dash
        ('ChainedHashMap', '2×'): {'color': '#27ae60', 'dash': 'solid'},         # green solid
        ('ChainedHashMap', '4×'): {'color': '#f39c12', 'dash': 'dash'}           # orange dash
    }
    
    # Add traces for each combination
    for variant in ['SimpleChainedHashMap', 'ChainedHashMap']:
        for capacity_mult in ['2×', '4×']:
            data = capacity_data[
                (capacity_data['variant'] == variant) & 
                (capacity_data['capacity_mult'] == capacity_mult)
            ].sort_values('elements')
            
            if len(data) > 0:
                config = line_configs[(variant, capacity_mult)]
                fig.add_trace(go.Scatter(
                    x=data['elements'],
                    y=data['Time_Mean_ms'],
                    mode='lines+markers',
                    name=f"{variant} {capacity_mult}",
                    line=dict(
                        color=config['color'],
                        dash=config['dash'],
                        width=2
                    ),
                    marker=dict(
                        size=8,
                        symbol='circle'
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Elements: %{x:,}<br>' +
                                  'Time: %{y:.2f} ms<br>' +
                                  '<extra></extra>'
                ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Migration Time vs Dataset Size: 2× vs 4× Capacity',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Dataset Size (elements)',
        yaxis_title='Migration Time (ms)',
        width=1100,
        height=600,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=70, r=150, t=80, b=60)
    )
    
    # Use log scales for both axes
    fig.update_xaxes(
        type="log",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        tickvals=[10000, 100000, 1000000, 10000000],
        ticktext=["10K", "100K", "1M", "10M"]
    )
    
    fig.update_yaxes(
        type="log",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)"
    )
    
    return fig