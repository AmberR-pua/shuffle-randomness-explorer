import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

# put below line into terminal to run, which get let you get UI,
# and can do the simulations in a more explicit way
# streamlit run app.py

# Pulls in the simulation engine functions.
from shuffle_main import (
    all_card_extreme_probability_rows,
    benchmark_shuffle_step,
    build_shuffle_config,
    deterministic_deck_path,
    find_return_to_original,
    position_probability_rows,
    results_to_frame,
    run_trials,
    run_trials_live,
    schedule_overhand,
    schedule_pile,
    schedule_riffle,
)

# Sets browser tab title, a wide layout and add the main title and subtitle.
st.set_page_config(page_title="Shuffle Mixing Explorer", layout="wide")
st.title("Card Shuffling Randomness Explorer")
st.caption(
    "Compare shuffle models or parameter settings, watch metrics update, and inspect how a tracked card spreads across positions."
)

# provide info for metrics for user to check and help them decide which metric they want to look at
METRIC_INFO = {
    "ks_pos": {
        "label": "KS distance of tracked-card position",
        "benchmark_label": "Expected benchmark: 0.0 (perfectly uniform tracked-card positions)",
        "direction": "Lower is better.",
        "plain": "This measures how far the tracked card's position distribution is from a uniform distribution.",
    },
    "pos_entropy_frac": {
        "label": "Position entropy fraction",
        "benchmark_label": "Expected benchmark: 1.0 (maximum spread across all positions)",
        "direction": "Higher is better.",
        "plain": "This measures how evenly the tracked card spreads over all positions.",
    },
    "inv_mean": {
        "label": "Mean inversion count",
        "benchmark_label": "Expected benchmark: n(n-1)/4 for a random permutation",
        "direction": "Closer to the benchmark is better.",
        "plain": "This is a global disorder score. More inversions means the deck is less like the original sorted order.",
    },
    "runs_mean": {
        "label": "Increasing sequences (rising runs)",
        "benchmark_label": "Expected benchmark: (n+1)/2 for a random permutation",
        "direction": "Closer to the benchmark is better for random mixing.",
        "plain": "A rising run is a consecutive increasing block. "
                 "Perfect structure gives very few runs, while random decks give about half the deck size in runs on average.",
    },
    "pos_mean": {
        "label": "Mean tracked-card position",
        "benchmark_label": "Expected benchmark: (n-1)/2 (middle of the deck on average)",
        "direction": "Closer to the benchmark is better.",
        "plain": "If the tracked card is mixed fairly, its average position should be around the middle of the deck.",
    },
}

BENCHMARK_COLUMN_MAP = {
    "ks_pos": "ks_pos_benchmark",
    "pos_entropy_frac": "pos_entropy_frac_benchmark",
    "inv_mean": "inv_uniform_mean",
    "runs_mean": "runs_mean_benchmark",
    "pos_mean": "pos_mean_benchmark",
}

# default method appear cyclically
DEFAULT_METHODS = ["riffle", "pile", "overhand", "riffle", "pile", "overhand", "riffle", "pile", "overhand", "riffle", "pile", "overhand"]
PREFIXES = [chr(ord("A") + i) for i in range(12)]
PAGES = ["Overview", "Tracked card", "Advanced diagnostics", "Perfect riffle", "Downloads"]


# descrption of the overall project and metrics and other terminology
with st.expander("What do we mean by randomness here?", expanded=True):
    st.markdown(
        """
**Randomness:** for each card of the deck, it should distribute uniformly to each position through the process of shuffling

**Tracked card.** The tracked card is one card with label from `0` to `n-1`. If the shuffle is mixing well, 
that card should become almost equally likely to appear in any position.

**Metrics used in this site.**
- **KS distance of tracked-card position:**   
Measures how far the tracked card’s position distribution is from perfectly uniform.  
  It looks for the *largest imbalance* — for example, if the card appears too often in certain positions.   
  smaller is better; `0` means a perfectly uniform tracked-card distribution.
- **Position entropy fraction:**   
Measures how evenly the tracked card is spread across all positions.  
   Instead of focusing on worst cases, this looks at the *overall spread*.  
closer to `1` is better; `1` means the tracked card is spread as evenly as possible.
- **Mean inversion count:**   
Measures how much the entire deck has been scrambled compared to the original order.  
  It counts how many pairs of cards are “out of order.”  
compares the deck to a random permutation. The benchmark is `n(n-1)/4`.
- **Increasing sequences (rising runs):**   
Counts how many ordered chunks remain in the deck(counts increasing blocks in the deck).   
  If large blocks of cards stay in order, the shuffle is not mixing well.  
  More runs means the original structure has been broken up more effectively.  
A random permutation has expected value `(n+1)/2`.
- **Mean tracked-card position:**   
Tracks where the chosen card tends to end up on average.   
for a fair shuffle, it should not favor the top or bottom, the expected average position is `(n-1)/2`.
- **Notice about inversion count and Increasing sequences:**   
Rising sequences and inversion count are different because they measure different scales of randomness:   
inversion count captures global disorder across all card pairs, while rising sequences detect local structure by identifying ordered chunks. A shuffle can appear random in one metric but not the other.

**Expected lines.** Each metric plot includes a benchmark line so you can compare the observed curve to the theoretical target.

**Cheat mode.** Honest models try to randomize the deck. Cheat modes intentionally keep some structure, such as fixing cards near the top or bottom.

**Perfect riffle.** This is a deterministic left-right-left-right interleave. It is useful for showing that some shuffles can look structured and can even cycle back to the original order.
        """
    )


# Run simulation with caching to avoid recomputing expensive Monte Carlo trials when inputs have not changed.
@st.cache_data(show_spinner=False)
def run_cached_result(
        method: str,
        trials: int,
        seed: int,
        tracked_card: int,
        deck_size: int,
        steps_list: list,
        riffle_cut_p: float,
        p_overhand: float,
        piles_k: int,
        pile_random_pickup: bool,
        cheat_mode: str,
        cheat_cards: int,
        max_workers: int,
        perfect_riffle: bool,
        perfect_riffle_start: str,
):
    cfg = build_shuffle_config(
        method=method,
        riffle_cut_p=riffle_cut_p,
        p_overhand=p_overhand,
        piles_k=piles_k,
        pile_random_pickup=pile_random_pickup,
        cheat_mode=cheat_mode,
        cheat_cards=cheat_cards,
        tracked_card=tracked_card,
        perfect_riffle=perfect_riffle,
        perfect_riffle_start=perfect_riffle_start,
    )
    results = run_trials(
        cfg=cfg,
        steps_list=steps_list,
        trials=trials,
        seed=seed,
        tracked_card=tracked_card,
        deck_size=deck_size,
        max_workers=max_workers,
    )
    return cfg, results


def make_metric_figure(df: pd.DataFrame, metric: str, title: str, compare_col: str = "label") -> go.Figure:
    """
    This draw line charts for one metric over steps, which can compare several configurations,
    add error bars, and overlay the theoretical benchmark line.
    """
    fig = go.Figure()
    if df.empty:
        return fig

    error_col = {"pos_mean": "pos_se", "inv_mean": "inv_se", "runs_mean": "runs_se"}.get(metric)

    for label, grp in df.groupby(compare_col):
        grp = grp.sort_values("steps")
        trace_kwargs = dict(x=grp["steps"], y=grp[metric], mode="lines+markers", name=str(label))
        if error_col and error_col in grp.columns:
            trace_kwargs["error_y"] = dict(type="data", array=1.96 * grp[error_col], visible=True)
        fig.add_trace(go.Scatter(**trace_kwargs))

    # Add benchmark dashed line if available
    benchmark_col = BENCHMARK_COLUMN_MAP.get(metric)
    if benchmark_col and benchmark_col in df.columns:
        bench = df.sort_values("steps").drop_duplicates("steps")[["steps", benchmark_col]].dropna()
        if not bench.empty:
            fig.add_trace(
                go.Scatter(
                    x=bench["steps"],
                    y=bench[benchmark_col],
                    mode="lines",
                    name="Expected benchmark",
                    line=dict(dash="dash"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Shuffles / steps",
        yaxis_title=METRIC_INFO.get(metric, {}).get("label", metric),
        uirevision=f"metric-{metric}",
    )
    return fig


def make_single_label_metric_figure(df: pd.DataFrame, metric: str, title: str, label: str) -> go.Figure:
    sub = df[df["label"] == label].copy()
    return make_metric_figure(sub, metric, title)


def make_multi_metric_overlay_figure(
        df: pd.DataFrame,
        metrics: list[str],
        title: str,
        compare_col: str = "label",
        normalize: bool = False,
        label_filter: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    if df.empty or not metrics:
        return fig

    plot_df = df.copy()
    if label_filter is not None:
        plot_df = plot_df[plot_df[compare_col] == label_filter].copy()
    if plot_df.empty:
        return fig

    for metric in metrics:
        if metric not in plot_df.columns:
            continue

        metric_df = plot_df[[compare_col, "steps", metric]].dropna().copy()
        if metric_df.empty:
            continue

        if normalize:
            vals = metric_df[metric].astype(float)
            lo, hi = vals.min(), vals.max()
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                metric_df["_plot_y"] = (vals - lo) / (hi - lo)
            else:
                metric_df["_plot_y"] = vals
            y_col = "_plot_y"
        else:
            y_col = metric

        for label, grp in metric_df.groupby(compare_col):
            grp = grp.sort_values("steps")
            fig.add_trace(
                go.Scatter(
                    x=grp["steps"],
                    y=grp[y_col],
                    mode="lines+markers",
                    name=f"{label} — {METRIC_INFO[metric]['label']}",
                    hovertemplate=(
                        f"Configuration: {label}<br>"
                        f"Metric: {METRIC_INFO[metric]['label']}<br>"
                        "Steps: %{x}<br>"
                        "Value: %{y}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Shuffles / steps",
        yaxis_title="Normalized value" if normalize else "Metric value",
        uirevision=f"multi-metric-{'-'.join(metrics)}-norm-{int(normalize)}",
    )
    return fig


def make_metric_vs_metric_figure(
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        title: str,
        compare_col: str = "label",
        label_filter: str | None = None,
        show_step_labels: bool = True,
) -> go.Figure:
    fig = go.Figure()
    if df.empty or x_metric not in df.columns or y_metric not in df.columns:
        return fig

    plot_df = df.copy()
    if label_filter is not None:
        plot_df = plot_df[plot_df[compare_col] == label_filter].copy()

    plot_df = plot_df[[compare_col, "steps", x_metric, y_metric]].dropna().copy()
    if plot_df.empty:
        return fig

    for label, grp in plot_df.groupby(compare_col):
        grp = grp.sort_values("steps")
        fig.add_trace(
            go.Scatter(
                x=grp[x_metric],
                y=grp[y_metric],
                mode="lines+markers+text" if show_step_labels else "lines+markers",
                text=grp["steps"].astype(str) if show_step_labels else None,
                textposition="top center",
                name=str(label),
                hovertemplate=(
                    f"Configuration: {label}<br>"
                    "Step: %{text}<br>"
                    f"{METRIC_INFO[x_metric]['label']}: %{{x}}<br>"
                    f"{METRIC_INFO[y_metric]['label']}: %{{y}}<extra></extra>"
                ) if show_step_labels else (
                    f"Configuration: {label}<br>"
                    f"{METRIC_INFO[x_metric]['label']}: %{{x}}<br>"
                    f"{METRIC_INFO[y_metric]['label']}: %{{y}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=METRIC_INFO[x_metric]["label"],
        yaxis_title=METRIC_INFO[y_metric]["label"],
        uirevision=f"metric-vs-metric-{x_metric}-{y_metric}",
    )
    return fig


def make_hist_figure(pos_df: pd.DataFrame, chosen_step: int, title: str, deck_size: int) -> go.Figure:
    """
    This shows the full probability distribution of the tracked card at one selected step

    If a deck is perfectly mixed in the tracked-card sense, each position should have probability about 1/n
    """
    # Select exactly one shuffle count.
    filt = pos_df[pos_df["steps"] == chosen_step].copy()
    fig = px.bar(filt, x="position", y="probability", color="label", barmode="group", title=title)
    fig.add_hline(y=1 / max(1, int(deck_size)), line_dash="dash", annotation_text="Uniform benchmark")
    fig.update_layout(yaxis_title="Estimated probability", uirevision=f"hist-{chosen_step}")
    return fig


def make_top_bottom_bar(named_results, chosen_step: int, deck_size: int) -> go.Figure:
    """
    For every card, show its highest-probability position and lowest-probability position at a chosen step.
    If mixing were perfect, these highs and lows should not vary much from 1/n.
    """
    rows = []
    for label, result in named_results.items():
        step_result = result.get(chosen_step)
        if not step_result:
            continue
        extreme_rows = pd.DataFrame(all_card_extreme_probability_rows({chosen_step: step_result}, int(deck_size)))
        if extreme_rows.empty:
            continue
        extreme_rows["label"] = label
        rows.append(extreme_rows)

    merged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if merged.empty:
        return go.Figure()

    merged["bar_slot"] = merged.groupby(["label", "card"]).cumcount()
    merged["x_label"] = merged["card"].astype(str) + "-" + merged["bar_slot"].map({0: "H", 1: "L"})

    fig = px.bar(
        merged,
        x="x_label",
        y="probability",
        color="extreme_type",
        facet_row="label",
        barmode="group",
        hover_data=["card", "position", "probability", "label"],
        category_orders={"extreme_type": ["Highest position probability", "Lowest position probability"]},
        title=(
            f"For each card: highest and lowest position probabilities at step {chosen_step} "
            f"(2 bars per card, so {2 * int(deck_size)} bars per configuration)"
        ),
    )
    fig.add_hline(y=1 / max(1, int(deck_size)), line_dash="dash")
    fig.update_layout(
        xaxis_title="Card (H = highest-probability position, L = lowest-probability position)",
        yaxis_title="Probability",
        uirevision=f"tb-{chosen_step}",
        bargap=0.2,
    )
    return fig


def make_top_bottom_bar_for_config(label: str, result, deck_size: int):
    """
    Same diagnostic as above, but focused on one chosen configuration.
    """
    if not result:
        return go.Figure(), None

    available_steps = sorted(result.keys())
    if not available_steps:
        return go.Figure(), None

    chosen_step = max(available_steps)
    step_result = result.get(chosen_step)
    if not step_result:
        return go.Figure(), chosen_step

    merged = pd.DataFrame(all_card_extreme_probability_rows({chosen_step: step_result}, int(deck_size)))
    if merged.empty:
        return go.Figure(), chosen_step

    merged["label"] = label
    merged["bar_slot"] = merged.groupby(["label", "card"]).cumcount()
    merged["x_label"] = merged["card"].astype(str) + "-" + merged["bar_slot"].map({0: "H", 1: "L"})

    fig = px.bar(
        merged,
        x="x_label",
        y="probability",
        color="extreme_type",
        barmode="group",
        hover_data=["card", "position", "probability", "label"],
        category_orders={"extreme_type": ["Highest position probability", "Lowest position probability"]},
        title=(
            f"For each card in {label}: highest and lowest position probabilities after the final shuffle step {chosen_step} "
            f"(2 bars per card, so {2 * int(deck_size)} bars total)"
        ),
    )
    fig.add_hline(y=1 / max(1, int(deck_size)), line_dash="dash")
    fig.update_layout(
        xaxis_title="Card (H = highest-probability position, L = lowest-probability position)",
        yaxis_title="Probability",
        uirevision=f"tb-final-{label}-{chosen_step}",
        bargap=0.2,
    )
    return fig, chosen_step


def make_3d_figure(pos_df: pd.DataFrame) -> go.Figure:
    """Visualize how tracked-card probability mass spreads across positions over time."""
    fig = px.scatter_3d(
        pos_df,
        x="steps",
        y="position",
        z="probability",
        color="label",
        opacity=0.6,
        title="3D view: step × position × probability",
    )
    fig.update_layout(uirevision="3d")
    return fig


def make_perfect_riffle_runs_figure(path_df: pd.DataFrame, label: str) -> go.Figure:
    """Plot rising runs under repeated perfect riffles, which help reveal strong hidden order."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path_df["steps"], y=path_df["runs"], mode="lines+markers", name=label))
    fig.add_trace(
        go.Scatter(
            x=path_df["steps"],
            y=[1] * len(path_df),
            mode="lines",
            name="Original-order benchmark (1 run)",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Increasing sequences under repeated perfect riffles: {label}",
        xaxis_title="Number of perfect riffles",
        yaxis_title="Increasing sequences (rising runs)",
        uirevision=f"perfect-{label}",
    )
    return fig


def combine_results_to_frames(named_results, deck_size: int):
    """
    This takes the nested simulation output and flattens it into two plotting tables:
    one for summary metrics over time;another for tracked-card position probabilities."""
    metric_rows, pos_rows = [], []
    for label, result in named_results.items():
        frame = pd.DataFrame(results_to_frame(result))
        if not frame.empty:
            frame["label"] = label
            metric_rows.append(frame)
        pframe = pd.DataFrame(position_probability_rows(result, deck_size))
        if not pframe.empty:
            pframe["label"] = label
            pos_rows.append(pframe)
    metrics_df = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    pos_df = pd.concat(pos_rows, ignore_index=True) if pos_rows else pd.DataFrame()
    return metrics_df, pos_df


def next_render_nonce() -> int:
    """This nonce gives each redraw a slightly different key so the chart refreshes properly."""
    st.session_state["render_nonce"] = int(st.session_state.get("render_nonce", 0)) + 1
    return int(st.session_state["render_nonce"])


def ui_params_from_prefix(prefix: str, method: str):
    """eads the current sidebar widget values for one configuration prefix"""
    perfect_riffle = bool(st.session_state.get(f"{prefix}_perfect_riffle", False)) if method == "riffle" else False
    perfect_riffle_start = st.session_state.get(f"{prefix}_perfect_start", "left")
    if method == "riffle":
        riffle_cut_p = float(st.session_state.get(f"{prefix}_riffle_cut_p", 0.5))
        p_overhand = 0.5
        piles_k = 7
        pile_random_pickup = True
    elif method == "overhand":
        riffle_cut_p = 0.5
        p_overhand = float(st.session_state.get(f"{prefix}_overhand_p", 0.5))
        piles_k = 7
        pile_random_pickup = True
    else:
        riffle_cut_p = 0.5
        p_overhand = 0.5
        piles_k = int(st.session_state.get(f"{prefix}_pile_k", 7))
        pile_random_pickup = bool(st.session_state.get(f"{prefix}_random_pickup", True))
    cheat_mode = st.session_state.get(f"{prefix}_cheat_mode", "none")
    cheat_cards = 0 if cheat_mode == "none" else int(st.session_state.get(f"{prefix}_cheat_cards", 2))
    return {
        "riffle_cut_p": riffle_cut_p,
        "p_overhand": p_overhand,
        "piles_k": piles_k,
        "pile_random_pickup": pile_random_pickup,
        "cheat_mode": cheat_mode,
        "cheat_cards": cheat_cards,
        "perfect_riffle": perfect_riffle,
        "perfect_riffle_start": perfect_riffle_start,
    }


def cfg_download_label(configs_meta, fallback: str) -> str:
    """Makes download filenames clearer."""
    if len(configs_meta) != 1:
        return fallback
    return configs_meta[0]["cfg"].name


def metric_download_name(cfg_name: str, compare_count: int) -> str:
    return "shuffle_metrics.csv" if compare_count > 1 else f"{cfg_name}_results.csv"


def position_download_name(cfg_name: str, compare_count: int) -> str:
    return "shuffle_position_probabilities.csv" if compare_count > 1 else f"{cfg_name}_position_probabilities.csv"


def _default_schedule_settings_for_method(method: str, perfect_riffle: bool = False):
    """ Return schedule settings for a method."""
    if method == "riffle":
        return {"max_shuffles": 20 if perfect_riffle else 12}
    if method == "overhand":
        return {"long_sched": False}
    return {"pile_max_steps": 200, "pile_stride": 10}


def export_single_config_from_session(meta: dict, idx: int):
    """Build and return an exportable JSON-like dictionary for one configuration."""
    prefix = meta["prefix"]
    default_method = DEFAULT_METHODS[idx % len(DEFAULT_METHODS)]
    method = st.session_state.get(f"{prefix}_method", default_method)

    perfect_riffle = bool(st.session_state.get(f"{prefix}_perfect_riffle", False)) if method == "riffle" else False

    if method == "riffle":
        schedule_settings = {
            "max_shuffles": int(st.session_state.get(f"{prefix}_max_shuffles", 20 if perfect_riffle else 12))
        }
    elif method == "overhand":
        schedule_settings = {
            "long_sched": bool(st.session_state.get(f"{prefix}_long_sched", False))
        }
    else:
        schedule_settings = {
            "pile_max_steps": int(st.session_state.get(f"{prefix}_pile_max_steps", 200)),
            "pile_stride": int(st.session_state.get(f"{prefix}_pile_stride", 10)),
        }

    shuffle_params = {
        "riffle_cut_p": float(st.session_state.get(f"{prefix}_riffle_cut_p", 0.5)),
        "p_overhand": float(st.session_state.get(f"{prefix}_overhand_p", 0.5)),
        "piles_k": int(st.session_state.get(f"{prefix}_pile_k", 7)),
        "pile_random_pickup": bool(st.session_state.get(f"{prefix}_random_pickup", True)),
        "cheat_mode": st.session_state.get(f"{prefix}_cheat_mode", "none"),
        "cheat_cards": int(st.session_state.get(f"{prefix}_cheat_cards", 2)),
        "perfect_riffle": perfect_riffle,
        "perfect_riffle_start": st.session_state.get(f"{prefix}_perfect_start", "left"),
    }

    return {
        "prefix": prefix,
        "display_name": st.session_state.get(f"{prefix}_display_name", meta.get("display_name", prefix)),
        "method": method,
        "schedule_settings": schedule_settings,
        "steps_list": list(meta.get("steps_list", [])),
        "shuffle_params": shuffle_params,
    }


# The two functions make the app reproducible.
# One exports the current experiment setup to JSON, and the other loads such a JSON file back into the app
def build_config_export_payload(sidebar_state):
    configs_payload = [
        export_single_config_from_session(meta, idx)
        for idx, meta in enumerate(sidebar_state["configs"])
    ]

    payload = {
        "app": "Card Shuffling Randomness Explorer",
        "version": 1,
        "common_settings": {
            "deck_size": int(sidebar_state["deck_size"]),
            "tracked_card": int(sidebar_state["tracked_card"]),
            "trials": int(sidebar_state["trials"]),
            "seed": int(sidebar_state["seed"]),
            "batch_size": int(sidebar_state["batch_size"]),
            "max_workers": int(sidebar_state["max_workers"]),
            "refresh_every": int(sidebar_state["refresh_every"]),
            "use_cached_final": bool(sidebar_state["use_cached_final"]),
            "compare_enabled": bool(sidebar_state["compare_enabled"]),
            "config_count": int(len(sidebar_state["configs"])),
        },
        "display_settings": {
            "metric_choice": sidebar_state["metric_choice"],
            "page_mode": sidebar_state["page_mode"],
            "chart_layout_mode": sidebar_state["chart_layout_mode"],
            "metric_view_mode": sidebar_state.get("metric_view_mode", "Single metric"),
            "metric_overlay_choices": list(sidebar_state.get("metric_overlay_choices") or []),
            "normalize_overlay_metrics": bool(sidebar_state.get("normalize_overlay_metrics", True)),
            "metric_overlay_focus": sidebar_state.get("metric_overlay_focus", "All visible configurations"),
            "metric_compare_x": sidebar_state.get("metric_compare_x", "ks_pos"),
            "metric_compare_y": sidebar_state.get("metric_compare_y", "pos_entropy_frac"),
            "show_metric_compare_step_labels": bool(sidebar_state.get("show_metric_compare_step_labels", True)),
            "show_advanced_graphs": bool(sidebar_state["show_advanced_graphs"]),
            "show_runtime_table": bool(sidebar_state["show_runtime_table"]),
            "tracked_label_choice": sidebar_state.get("tracked_label_choice"),
            "advanced_diag_label_choice": sidebar_state.get("advanced_diag_label_choice"),
            "visible_labels": list(sidebar_state.get("visible_labels", [])),
        },
        "shuffle_configurations": configs_payload,
    }

    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def apply_loaded_configuration(payload: dict):
    common = payload.get("common_settings", {}) or {}
    display = payload.get("display_settings", {}) or {}
    configs = payload.get("shuffle_configurations", []) or []

    config_count = max(1, min(len(PREFIXES), int(common.get("config_count", len(configs) or 1))))
    compare_enabled = bool(common.get("compare_enabled", config_count > 1))

    st.session_state["deck_size"] = int(common.get("deck_size", st.session_state.get("deck_size", 52)))
    st.session_state["tracked_card"] = int(common.get("tracked_card", st.session_state.get("tracked_card", 0)))
    st.session_state["trials"] = int(common.get("trials", st.session_state.get("trials", 600)))
    st.session_state["seed"] = int(common.get("seed", st.session_state.get("seed", 2026)))
    st.session_state["batch_size"] = int(common.get("batch_size", st.session_state.get("batch_size", 100)))
    st.session_state["max_workers"] = int(common.get("max_workers", st.session_state.get("max_workers", 2)))
    st.session_state["refresh_every"] = int(common.get("refresh_every", st.session_state.get("refresh_every", 2)))
    st.session_state["use_cached_final"] = bool(common.get("use_cached_final", st.session_state.get("use_cached_final", False)))
    st.session_state["compare_enabled"] = compare_enabled
    st.session_state["config_count"] = config_count

    if "metric_choice" in display:
        st.session_state["metric_choice"] = display["metric_choice"]
    if "page_mode" in display:
        st.session_state["page_mode"] = display["page_mode"]
    if "chart_layout_mode" in display:
        st.session_state["chart_layout_mode"] = display["chart_layout_mode"]
    if "metric_view_mode" in display:
        st.session_state["metric_view_mode"] = display["metric_view_mode"]
    if "metric_overlay_choices" in display:
        st.session_state["metric_overlay_choices"] = list(display["metric_overlay_choices"])
    if "normalize_overlay_metrics" in display:
        st.session_state["normalize_overlay_metrics"] = bool(display["normalize_overlay_metrics"])
    if "metric_overlay_focus" in display:
        st.session_state["metric_overlay_focus"] = display["metric_overlay_focus"]
    if "metric_compare_x" in display:
        st.session_state["metric_compare_x"] = display["metric_compare_x"]
    if "metric_compare_y" in display:
        st.session_state["metric_compare_y"] = display["metric_compare_y"]
    if "show_metric_compare_step_labels" in display:
        st.session_state["show_metric_compare_step_labels"] = bool(display["show_metric_compare_step_labels"])
    if "show_advanced_graphs" in display:
        st.session_state["show_advanced_graphs"] = bool(display["show_advanced_graphs"])
    if "show_runtime_table" in display:
        st.session_state["show_runtime_table"] = bool(display["show_runtime_table"])
    if "tracked_label_choice" in display:
        st.session_state["tracked_label_choice"] = display["tracked_label_choice"]
    if "advanced_diag_label_choice" in display:
        st.session_state["advanced_diag_label_choice"] = display["advanced_diag_label_choice"]
    if "visible_labels" in display:
        st.session_state["visible_labels"] = list(display["visible_labels"])

    for idx, cfg_data in enumerate(configs[: len(PREFIXES)]):
        prefix = PREFIXES[idx]
        method = cfg_data.get("method", DEFAULT_METHODS[idx % len(DEFAULT_METHODS)])
        params = cfg_data.get("shuffle_params", {}) or {}
        schedule = cfg_data.get("schedule_settings", {}) or {}

        perfect_riffle = bool(params.get("perfect_riffle", False))
        schedule_defaults = _default_schedule_settings_for_method(method, perfect_riffle=perfect_riffle)

        st.session_state[f"{prefix}_display_name"] = cfg_data.get("display_name", prefix)
        st.session_state[f"{prefix}_method"] = method

        st.session_state[f"{prefix}_perfect_riffle"] = perfect_riffle
        st.session_state[f"{prefix}_perfect_start"] = params.get("perfect_riffle_start", "left")
        st.session_state[f"{prefix}_riffle_cut_p"] = float(params.get("riffle_cut_p", 0.5))

        st.session_state[f"{prefix}_overhand_p"] = float(params.get("p_overhand", 0.5))

        default_pile_k = min(7, int(st.session_state["deck_size"]))
        st.session_state[f"{prefix}_pile_k"] = int(params.get("piles_k", default_pile_k))
        st.session_state[f"{prefix}_random_pickup"] = bool(params.get("pile_random_pickup", True))

        st.session_state[f"{prefix}_cheat_mode"] = params.get("cheat_mode", "none")
        st.session_state[f"{prefix}_cheat_cards"] = int(params.get("cheat_cards", 2))

        if method == "riffle":
            st.session_state[f"{prefix}_max_shuffles"] = int(
                schedule.get("max_shuffles", schedule_defaults["max_shuffles"])
            )
        elif method == "overhand":
            st.session_state[f"{prefix}_long_sched"] = bool(
                schedule.get("long_sched", schedule_defaults["long_sched"])
            )
        else:
            st.session_state[f"{prefix}_pile_max_steps"] = int(
                schedule.get("pile_max_steps", schedule_defaults["pile_max_steps"])
            )
            st.session_state[f"{prefix}_pile_stride"] = int(
                schedule.get("pile_stride", schedule_defaults["pile_stride"])
            )


def ensure_unique_config_labels(configs_meta):
    """This function prevents label collisions in plots and tables."""
    counts = {}
    for meta in configs_meta:
        name = str(meta.get("display_name", meta.get("prefix", "Config"))).strip() or str(meta.get("prefix", "Config"))
        counts[name] = counts.get(name, 0) + 1

    seen = {}
    normalized = []
    for meta in configs_meta:
        base_name = str(meta.get("display_name", meta.get("prefix", "Config"))).strip() or str(meta.get("prefix", "Config"))
        seen[base_name] = seen.get(base_name, 0) + 1
        final_name = base_name
        if counts[base_name] > 1:
            final_name = f"{base_name} ({meta['prefix']})"
        normalized.append({**meta, "display_name": final_name})
    return normalized


def render_schedule_controls(container, prefix: str, method: str, perfect_riffle: bool = False):
    """Creates the sidebar controls that determine which shuffle steps are simulated."""
    if method == "riffle":
        max_shuffles = container.slider(
            f"{prefix} max shuffles", 1, 60, 12 if not perfect_riffle else 20, 1, key=f"{prefix}_max_shuffles"
        )
        return schedule_riffle(max_shuffles)
    if method == "overhand":
        long_sched = container.checkbox(f"{prefix} long overhand schedule", value=False, key=f"{prefix}_long_sched")
        return schedule_overhand(long=long_sched)
    max_steps = container.slider(f"{prefix} pile max steps", 10, 500, 200, 1, key=f"{prefix}_pile_max_steps")
    stride = container.slider(f"{prefix} pile stride", 1, 25, 10, 1, key=f"{prefix}_pile_stride")
    return schedule_pile(max_steps=max_steps, stride=stride)


def render_config_controls(container, prefix: str, default_method: str, deck_size: int, tracked_card: int):
    """This is the sidebar builder for one experiment."""
    display_name = container.text_input(
        f"{prefix} display label",
        value=str(st.session_state.get(f"{prefix}_display_name", prefix)),
        key=f"{prefix}_display_name",
        help="Optional label shown in plots and tables.",
    )
    method = container.selectbox(
        f"{prefix} method",
        ["riffle", "overhand", "pile"],
        key=f"{prefix}_method",
        index=["riffle", "overhand", "pile"].index(default_method),
    )
    riffle_cut_p = 0.5
    p_overhand = 0.5
    piles_k = 7
    pile_random_pickup = True
    perfect_riffle = False
    perfect_riffle_start = "left"

    if method == "riffle":
        perfect_riffle = container.checkbox(
            f"{prefix} use perfect riffle (deterministic faro-style)", value=False, key=f"{prefix}_perfect_riffle"
        )
        if perfect_riffle:
            perfect_riffle_start = container.selectbox(
                f"{prefix} perfect riffle start side",
                ["left", "right"],
                key=f"{prefix}_perfect_start",
                help="left starts with the left half, right starts with the right half.",
            )
        else:
            riffle_cut_p = container.slider(
                f"{prefix} riffle cut_p", 0.1, 0.9, 0.5, 0.05, key=f"{prefix}_riffle_cut_p"
            )
    elif method == "overhand":
        p_overhand = container.slider(f"{prefix} overhand p", 0.05, 0.95, 0.5, 0.05, key=f"{prefix}_overhand_p")
    else:
        piles_k = container.slider(
            f"{prefix} pile k", 2, int(deck_size), min(7, int(deck_size)), 1, key=f"{prefix}_pile_k"
        )
        pile_random_pickup = container.checkbox(f"{prefix} random pickup", value=True, key=f"{prefix}_random_pickup")

    cheat_mode = container.selectbox(
        f"{prefix} cheat mode",
        ["none", "keep_top", "keep_bottom", "return_tracked_to_top"],
        key=f"{prefix}_cheat_mode",
        help=(
            "keep_top keeps the original top cards fixed near the top; keep_bottom keeps the original bottom cards fixed near the bottom; "
            "return_tracked_to_top forces the selected tracked card back to the top after each shuffle."
        ),
    )
    cheat_cards = 0
    if cheat_mode != "none":
        cheat_cards = container.slider(
            f"{prefix} cheat cards", 1, min(10, int(deck_size)), 2, 1, key=f"{prefix}_cheat_cards"
        )

    steps_list = render_schedule_controls(container, prefix, method, perfect_riffle=perfect_riffle)
    cfg = build_shuffle_config(
        method=method,
        riffle_cut_p=riffle_cut_p,
        p_overhand=p_overhand,
        piles_k=piles_k,
        pile_random_pickup=pile_random_pickup,
        cheat_mode=cheat_mode,
        cheat_cards=cheat_cards,
        tracked_card=int(tracked_card),
        perfect_riffle=perfect_riffle,
        perfect_riffle_start=perfect_riffle_start,
    )
    return cfg, steps_list, (display_name.strip() or prefix)

# This function build sidebar the user can modify for experiment
def build_sidebar():
    st.sidebar.header("Experiment Controls")

    restore_message = st.session_state.pop("config_restore_message", None)
    if restore_message:
        st.sidebar.success(restore_message)

    restore_box = st.sidebar.expander("Load saved configuration", expanded=False)
    uploaded_config = restore_box.file_uploader(
        "Upload configuration JSON",
        type=["json"],
        key="uploaded_shuffle_config_json",
        help="Upload a configuration JSON previously exported from the Downloads page.",
    )

    if uploaded_config is not None:
        restore_box.caption("Click the button below to apply the uploaded configuration to the sidebar controls.")
        if restore_box.button(
                "Apply uploaded configuration",
                width="stretch",
                key="apply_uploaded_shuffle_config",
        ):
            try:
                payload = json.loads(uploaded_config.getvalue().decode("utf-8"))
                apply_loaded_configuration(payload)
                st.session_state["config_restore_message"] = "Configuration loaded from JSON."
                st.rerun()
            except Exception as exc:
                restore_box.error(f"Could not load configuration JSON: {exc}")

    common_box = st.sidebar.expander("Common experiment settings", expanded=True)
    deck_size = common_box.number_input("Deck size (n)", min_value=5, max_value=300, value=52, step=1, key="deck_size")
    tracked_card = common_box.number_input(
        "Tracked card (0-index)", min_value=0, max_value=int(deck_size - 1), value=0, step=1, key="tracked_card"
    )
    trials = common_box.slider("Trials", 50, 5000, 600, step=50, key="trials")
    seed = common_box.number_input("Seed", min_value=0, max_value=10_000_000, value=2026, step=1, key="seed")
    batch_size = common_box.slider("Live update batch size", 10, 500, 100, step=10, key="batch_size")
    max_workers = common_box.slider("Worker threads", 1, 8, 2, step=1, key="max_workers")
    refresh_every = common_box.slider("Redraw every N live batches", 1, 10, 2, step=1, key="refresh_every")
    use_cached_final = common_box.checkbox("Use cached final run (no live updates)", value=False, key="use_cached_final")
    compare_enabled = common_box.checkbox(
        "Compare multiple configurations",
        value=bool(st.session_state.get("compare_enabled", True)),
        key="compare_enabled",
        help="Turn this on to compare several configurations. When it is off, the app runs a single configuration and hides comparison-only controls.",
    )

    if compare_enabled:
        comparison_box = st.sidebar.expander("Comparison setup", expanded=True)
        config_count = comparison_box.number_input(
            "Number of shuffle configurations to compare",
            min_value=1,
            max_value=12,
            value=max(2, int(st.session_state.get("config_count", 2))),
            step=1,
            key="config_count",
            help="Use as many configurations as you need. When the comparison gets crowded, filter which ones are visible in charts.",
        )
    else:
        config_count = 1

    configs = []
    for idx in range(int(config_count)):
        prefix = PREFIXES[idx]
        config_box = st.sidebar.expander(f"Configuration {prefix}", expanded=(idx < 2))
        cfg, steps_list, display_name = render_config_controls(
            config_box,
            prefix,
            DEFAULT_METHODS[idx % len(DEFAULT_METHODS)],
            int(deck_size),
            int(tracked_card),
        )
        configs.append({"prefix": prefix, "display_name": display_name, "cfg": cfg, "steps_list": steps_list})

    configs = ensure_unique_config_labels(configs)

    stored_results = st.session_state.get("latest_results")
    stored_configs = ensure_unique_config_labels(st.session_state.get("last_configs_meta", []))
    available_steps = []
    if stored_results and stored_configs:
        latest_named_results = make_named_results(stored_configs, stored_results)
        _metrics_df, stored_pos_df = combine_results_to_frames(latest_named_results, int(deck_size))
        if not stored_pos_df.empty:
            available_steps = sorted(stored_pos_df["steps"].unique())

    visible_box = st.sidebar.expander("Visible configurations in charts", expanded=bool(compare_enabled))
    visible_options = [cfg_meta["display_name"] for cfg_meta in configs]
    previous_labels = [cfg_meta["display_name"] for cfg_meta in (stored_configs or configs)]
    default_visible = st.session_state.get("visible_labels", previous_labels)
    visible_labels = visible_box.multiselect(
        "Choose which configurations to show",
        options=visible_options,
        default=[label for label in default_visible if label in visible_options] or visible_options,
        key="visible_labels_sidebar",
        help="You can run many configurations but only show a subset in the charts.",
        disabled=not compare_enabled,
    )
    if not compare_enabled:
        visible_labels = visible_options

    display_box = st.sidebar.expander("Display options", expanded=True)
    # metric_options = list(METRIC_INFO.keys())
    # metric_choice = display_box.selectbox(
    #     "Main metric",
    #     metric_options,
    #     index=0,
    #     key="metric_choice",
    #     format_func=lambda key: f"{METRIC_INFO[key]['label']} — {METRIC_INFO[key]['benchmark_label']}",
    # )
    metric_options = list(METRIC_INFO.keys())
    current_metric = st.session_state.get("metric_choice", "ks_pos")
    if current_metric not in metric_options:
        current_metric = metric_options[0]

    metric_choice = display_box.selectbox(
        "Main metric",
        metric_options,
        index=metric_options.index(current_metric),
        key="metric_choice",
        format_func=lambda key: f"{METRIC_INFO[key]['label']} — {METRIC_INFO[key]['benchmark_label']}",
    )

    metric_view_mode_options = ["Single metric", "Multi-metric overlay", "Metric vs metric"]
    previous_metric_view_mode = st.session_state.get("metric_view_mode", "Single metric")
    if previous_metric_view_mode not in metric_view_mode_options:
        previous_metric_view_mode = "Single metric"

    metric_view_mode = display_box.radio(
        "Metric view mode",
        metric_view_mode_options,
        index=metric_view_mode_options.index(previous_metric_view_mode),
        key="metric_view_mode",
        help="Single metric uses the main metric. Multi-metric overlay shows several metrics against steps. Metric vs metric compares two metrics directly.",
    )

    default_overlay_metrics = st.session_state.get("metric_overlay_choices", [metric_choice, "pos_entropy_frac"])
    default_overlay_metrics = [m for m in default_overlay_metrics if m in metric_options]
    if not default_overlay_metrics:
        default_overlay_metrics = [metric_choice]

    metric_overlay_choices = display_box.multiselect(
        "Metrics to overlay",
        metric_options,
        default=default_overlay_metrics,
        key="metric_overlay_choices",
        format_func=lambda key: METRIC_INFO[key]["label"],
    )

    normalize_overlay_metrics = display_box.checkbox(
        "Normalize overlayed metrics",
        value=bool(st.session_state.get("normalize_overlay_metrics", True)),
        key="normalize_overlay_metrics",
        help="Recommended when overlaying metrics with different scales.",
    )

    overlay_focus_options = ["All visible configurations"] + [meta["display_name"] for meta in configs]
    previous_overlay_focus = st.session_state.get("metric_overlay_focus", "All visible configurations")
    if previous_overlay_focus not in overlay_focus_options:
        previous_overlay_focus = "All visible configurations"

    metric_overlay_focus = display_box.selectbox(
        "Overlay focus",
        overlay_focus_options,
        index=overlay_focus_options.index(previous_overlay_focus),
        key="metric_overlay_focus",
        help="Show multi-metric overlay for all visible configurations or focus on one configuration.",
    )

    previous_x_metric = st.session_state.get("metric_compare_x", "ks_pos")
    if previous_x_metric not in metric_options:
        previous_x_metric = metric_options[0]

    metric_compare_x = display_box.selectbox(
        "Metric comparison X-axis",
        metric_options,
        index=metric_options.index(previous_x_metric),
        key="metric_compare_x",
        format_func=lambda key: METRIC_INFO[key]["label"],
    )

    compare_y_options = [m for m in metric_options if m != metric_compare_x]
    previous_y_metric = st.session_state.get("metric_compare_y", "pos_entropy_frac")
    if previous_y_metric not in compare_y_options:
        previous_y_metric = compare_y_options[0]

    metric_compare_y = display_box.selectbox(
        "Metric comparison Y-axis",
        compare_y_options,
        index=compare_y_options.index(previous_y_metric),
        key="metric_compare_y",
        format_func=lambda key: METRIC_INFO[key]["label"],
    )

    show_metric_compare_step_labels = display_box.checkbox(
        "Show step labels in metric-vs-metric plot",
        value=bool(st.session_state.get("show_metric_compare_step_labels", True)),
        key="show_metric_compare_step_labels",
    )

    page_mode = display_box.radio(
        "Results page",
        PAGES,
        index=0,
        key="page_mode",
        help="Choose which result section to display.",
    )
    chart_layout_options = ["Overlay", "Side by side"] if compare_enabled else ["Overlay"]
    previous_layout = st.session_state.get("chart_layout_mode", "Overlay")
    chart_layout_mode = display_box.radio(
        "Comparison chart layout",
        chart_layout_options,
        index=chart_layout_options.index(previous_layout) if previous_layout in chart_layout_options else 0,
        key="chart_layout_mode",
        help="Overlay puts configurations on the same chart. Side by side shows one chart per configuration.",
    )
    show_advanced_graphs = display_box.checkbox(
        "Show advanced diagnostics by default",
        value=False,
        key="show_advanced_graphs",
    )
    show_runtime_table = display_box.checkbox(
        "Show runtime benchmark table",
        value=True,
        key="show_runtime_table",
    )

    tracked_label_options = ["All visible configurations"]
    if compare_enabled:
        tracked_label_options.append("All configurations")
    tracked_label_options.extend([meta["display_name"] for meta in configs])

    tracked_step = st.session_state.get("tracked_step_sidebar") if available_steps else None

    default_tracked_label = st.session_state.get("tracked_label_choice", "All visible configurations")
    if default_tracked_label not in tracked_label_options:
        default_tracked_label = "All visible configurations"
    tracked_label_choice = display_box.selectbox(
        "Tracked-card configuration to show",
        tracked_label_options,
        index=tracked_label_options.index(default_tracked_label),
        key="tracked_label_choice",
        help="Choose one configuration for the tracked-card histogram, or keep all visible configurations together.",
    )

    advanced_diag_options = [meta["display_name"] for meta in configs]
    default_advanced_diag = st.session_state.get(
        "advanced_diag_label_choice",
        advanced_diag_options[0] if advanced_diag_options else None,
    )
    if advanced_diag_options:
        if default_advanced_diag not in advanced_diag_options:
            default_advanced_diag = advanced_diag_options[0]
        advanced_diag_label_choice = display_box.selectbox(
            "Advanced-diagnostics configuration to show",
            advanced_diag_options,
            index=advanced_diag_options.index(default_advanced_diag),
            key="advanced_diag_label_choice",
            help="Choose which configuration to inspect in the highest/lowest probability chart. This chart always uses that configuration's last completed shuffle step.",
        )
    else:
        advanced_diag_label_choice = None

    metric_guide_box = st.sidebar.expander("Metric guide", expanded=False)
    for info in METRIC_INFO.values():
        metric_guide_box.markdown(f"**{info['label']}**")
        metric_guide_box.caption(f"{info['direction']} {info['benchmark_label']}")
        metric_guide_box.write(info["plain"])

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("Run simulation", width="stretch", key="run_simulation_button")

    return {
        "deck_size": int(deck_size),
        "tracked_card": int(tracked_card),
        "trials": int(trials),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "max_workers": int(max_workers),
        "refresh_every": int(refresh_every),
        "use_cached_final": bool(use_cached_final),
        "compare_enabled": bool(compare_enabled),
        "metric_choice": metric_choice,
        "page_mode": page_mode,
        "metric_view_mode": metric_view_mode,
        "metric_overlay_choices": list(metric_overlay_choices or []),
        "normalize_overlay_metrics": bool(normalize_overlay_metrics),
        "metric_overlay_focus": metric_overlay_focus,
        "metric_compare_x": metric_compare_x,
        "metric_compare_y": metric_compare_y,
        "show_metric_compare_step_labels": bool(show_metric_compare_step_labels),
        "chart_layout_mode": chart_layout_mode,
        "show_advanced_graphs": bool(show_advanced_graphs),
        "show_runtime_table": bool(show_runtime_table),
        "tracked_step": tracked_step,
        "tracked_label_choice": tracked_label_choice,
        "advanced_diag_label_choice": advanced_diag_label_choice,
        "configs": configs,
        "visible_labels": visible_labels,
        "run_btn": run_btn,
    }

def make_runtime_df(configs_meta, deck_size: int) -> pd.DataFrame:
    """time cost comparison between methods"""
    rows = []
    for meta in configs_meta:
        rows.append(
            {
                "label": meta["display_name"],
                "config": meta["cfg"].name,
                "sec_per_shuffle": benchmark_shuffle_step(meta["cfg"], steps=500, repeats=30, deck_size=int(deck_size)),
            }
        )
    return pd.DataFrame(rows)


def make_named_results(configs_meta, latest_results):
    named = {}
    for meta in configs_meta:
        prefix = meta["prefix"]
        if prefix in latest_results:
            named[meta["display_name"]] = latest_results[prefix]
    return named


def choose_tracked_step(source_pos_df: pd.DataFrame, chosen_label: str):
    """ Keeps the tracked-card inspection step stable and friendly to users
        even when the chosen configuration changes."""
    available_steps = sorted(source_pos_df["steps"].unique()) if not source_pos_df.empty else []
    if not available_steps:
        return None, []

    state_key = f"tracked_step_for::{chosen_label}"
    default_step = available_steps[len(available_steps) // 2]
    current_value = st.session_state.get(state_key, st.session_state.get("tracked_step_sidebar", default_step))
    if current_value not in available_steps:
        current_value = default_step

    chosen_step = st.select_slider(
        "Tracked-card step to inspect",
        options=available_steps,
        value=current_value,
        key=state_key,
        help="Choose the shuffle step used for the tracked-card distribution. The available steps come from the selected configuration view.",
    )
    st.session_state["tracked_step_sidebar"] = chosen_step
    return chosen_step, available_steps

# identify which choice the user select and load the corresponding page with different result and data
def render_results(sidebar_state, target_placeholder=None):
    render_target = target_placeholder.container() if target_placeholder is not None else st.container()
    with render_target:
        stored_results = st.session_state.get("latest_results")
        stored_configs = st.session_state.get("last_configs_meta")
        if not stored_results or not stored_configs:
            st.info("Choose configurations in the sidebar, then click Run simulation.")
            return

        st.info("Showing the most recent completed simulation. Change display options freely without rerunning.")

        # visible_labels = sidebar_state["visible_labels"] or [meta["display_name"] for meta in stored_configs]
        visible_labels = st.session_state.get(
            "visible_labels",
            sidebar_state["visible_labels"] or [meta["display_name"] for meta in stored_configs],
            )
        latest_named_results = make_named_results(stored_configs, stored_results)
        filtered_named_results = {k: v for k, v in latest_named_results.items() if k in visible_labels}

        metrics_df, pos_df = combine_results_to_frames(filtered_named_results, sidebar_state["deck_size"])
        _all_metrics_df, all_pos_df = combine_results_to_frames(latest_named_results, sidebar_state["deck_size"])
        runtime_df = st.session_state.get("runtime_df", pd.DataFrame())

        if metrics_df.empty and pos_df.empty:
            st.warning("No results are currently visible. Choose at least one configuration in the sidebar filter.")
            return

        available_steps = sorted(pos_df["steps"].unique()) if not pos_df.empty else []
        if available_steps:
            default_step = available_steps[len(available_steps) // 2]
            chosen_step = sidebar_state.get("tracked_step")
            if chosen_step not in available_steps:
                chosen_step = st.session_state.get("tracked_step_sidebar", default_step)
            if chosen_step not in available_steps:
                chosen_step = default_step
        else:
            chosen_step = None

        # page_mode = sidebar_state["page_mode"]
        # metric_choice = sidebar_state["metric_choice"]
        # chart_layout_mode = sidebar_state["chart_layout_mode"]
        page_mode = st.session_state.get("page_mode", sidebar_state["page_mode"])
        metric_choice = st.session_state.get("metric_choice", sidebar_state["metric_choice"])
        chart_layout_mode = st.session_state.get("chart_layout_mode", sidebar_state["chart_layout_mode"])
        metric_view_mode = st.session_state.get(
            "metric_view_mode",
            sidebar_state.get("metric_view_mode", "Single metric"),
        )
        metric_overlay_choices = st.session_state.get(
            "metric_overlay_choices",
            sidebar_state.get("metric_overlay_choices", [metric_choice]),
        )
        normalize_overlay_metrics = bool(
            st.session_state.get(
                "normalize_overlay_metrics",
                sidebar_state.get("normalize_overlay_metrics", True),
            )
        )
        metric_overlay_focus = st.session_state.get(
            "metric_overlay_focus",
            sidebar_state.get("metric_overlay_focus", "All visible configurations"),
        )
        metric_compare_x = st.session_state.get(
            "metric_compare_x",
            sidebar_state.get("metric_compare_x", "ks_pos"),
        )
        metric_compare_y = st.session_state.get(
            "metric_compare_y",
            sidebar_state.get("metric_compare_y", "pos_entropy_frac"),
        )
        show_metric_compare_step_labels = bool(
            st.session_state.get(
                "show_metric_compare_step_labels",
                sidebar_state.get("show_metric_compare_step_labels", True),
            )
        )
        render_nonce = next_render_nonce()

        if page_mode == "Overview":
            st.subheader("Overview")
            if not metrics_df.empty:
                labels = list(metrics_df["label"].dropna().unique())
                if metric_view_mode == "Single metric":
                    st.caption(
                        f"{METRIC_INFO[metric_choice]['plain']} {METRIC_INFO[metric_choice]['direction']} {METRIC_INFO[metric_choice]['benchmark_label']}"
                    )
                    if chart_layout_mode == "Side by side" and len(labels) > 1:
                        cols = st.columns(min(len(labels), 3))
                        for idx, label in enumerate(labels):
                            with cols[idx % len(cols)]:
                                st.plotly_chart(
                                    make_single_label_metric_figure(metrics_df, metric_choice, f"{label}: {METRIC_INFO[metric_choice]['label']} over time", label),
                                    width="stretch",
                                    key=f"metric_chart_{metric_choice}_{label}_{render_nonce}",
                                )
                    else:
                        st.plotly_chart(
                            make_metric_figure(metrics_df, metric_choice, f"{METRIC_INFO[metric_choice]['label']} over time"),
                            width="stretch",
                            key=f"metric_chart_{metric_choice}_{render_nonce}",
                        )

                    secondary_metric = "runs_mean" if metric_choice != "runs_mean" else "inv_mean"
                    st.caption(f"Secondary view: {METRIC_INFO[secondary_metric]['plain']} {METRIC_INFO[secondary_metric]['direction']}")
                    if chart_layout_mode == "Side by side" and len(labels) > 1:
                        cols = st.columns(min(len(labels), 3))
                        for idx, label in enumerate(labels):
                            with cols[idx % len(cols)]:
                                st.plotly_chart(
                                    make_single_label_metric_figure(metrics_df, secondary_metric, f"{label}: {METRIC_INFO[secondary_metric]['label']} over time", label),
                                    width="stretch",
                                    key=f"secondary_chart_{secondary_metric}_{label}_{render_nonce}",
                                )
                    else:
                        st.plotly_chart(
                            make_metric_figure(metrics_df, secondary_metric, f"{METRIC_INFO[secondary_metric]['label']} over time"),
                            width="stretch",
                            key=f"secondary_chart_{secondary_metric}_{render_nonce}",
                        )
                elif metric_view_mode == "Multi-metric overlay":
                    valid_metrics = [m for m in metric_overlay_choices if m in METRIC_INFO]
                    if not valid_metrics:
                        st.warning("Choose at least one metric to overlay.")
                    else:
                        if metric_overlay_focus == "All visible configurations":
                            st.caption("Overlaying multiple metrics across the currently visible configurations.")
                            st.plotly_chart(
                                make_multi_metric_overlay_figure(
                                    metrics_df,
                                    valid_metrics,
                                    title="Multi-metric overlay over time",
                                    normalize=normalize_overlay_metrics,
                                ),
                                width="stretch",
                                key=f"multi_metric_overlay_all_{render_nonce}",
                            )
                        else:
                            st.caption(f"Overlaying multiple metrics for configuration: {metric_overlay_focus}")
                            st.plotly_chart(
                                make_multi_metric_overlay_figure(
                                    metrics_df,
                                    valid_metrics,
                                    title=f"{metric_overlay_focus}: multi-metric overlay over time",
                                    normalize=normalize_overlay_metrics,
                                    label_filter=metric_overlay_focus,
                                ),
                                width="stretch",
                                key=f"multi_metric_overlay_{metric_overlay_focus}_{render_nonce}",
                            )
                elif metric_view_mode == "Metric vs metric":
                    if metric_compare_x == metric_compare_y:
                        st.warning("Choose two different metrics for the comparison plot.")
                    else:
                        st.caption(
                            f"Comparing {METRIC_INFO[metric_compare_x]['label']} "
                            f"against {METRIC_INFO[metric_compare_y]['label']}."
                        )
                        st.plotly_chart(
                            make_metric_vs_metric_figure(
                                metrics_df,
                                metric_compare_x,
                                metric_compare_y,
                                title=(
                                    f"{METRIC_INFO[metric_compare_x]['label']} vs "
                                    f"{METRIC_INFO[metric_compare_y]['label']}"
                                ),
                                show_step_labels=show_metric_compare_step_labels,
                            ),
                            width="stretch",
                            key=f"metric_vs_metric_{metric_compare_x}_{metric_compare_y}_{render_nonce}",
                        )
                else:
                    st.warning(f"Unknown metric view mode: {metric_view_mode}")
                with st.expander("Show metric table", expanded=False):
                    st.dataframe(metrics_df.sort_values(["label", "steps"]), width="stretch")
            else:
                st.info("No metric data available for the selected visible configurations.")

        elif page_mode == "Tracked card":
            st.subheader("Tracked card")
            tracked_label_choice = sidebar_state.get("tracked_label_choice", "All visible configurations")

            if tracked_label_choice == "All visible configurations":
                source_pos_df = pos_df.copy()
            elif tracked_label_choice == "All configurations":
                source_pos_df = all_pos_df.copy()
            else:
                source_pos_df = all_pos_df[all_pos_df["label"] == tracked_label_choice].copy()

            if not source_pos_df.empty:
                chosen_tracked_step, available_tracked_steps = choose_tracked_step(source_pos_df, tracked_label_choice)
            else:
                chosen_tracked_step, available_tracked_steps = None, []

            if not source_pos_df.empty and chosen_tracked_step is not None:
                tracked_view_df = source_pos_df.copy()

                if tracked_view_df.empty:
                    st.info("The selected tracked-card configuration has no available data for the current run.")
                else:
                    if tracked_label_choice == "All visible configurations":
                        st.caption(
                            "This histogram shows where the tracked card lands after the selected number of shuffles for the configurations currently visible in charts. A flatter distribution means better mixing."
                        )
                        hist_title = f"Tracked-card position distribution at step {chosen_tracked_step}"
                    elif tracked_label_choice == "All configurations":
                        st.caption(
                            "This histogram shows where the tracked card lands after the selected number of shuffles across all simulated configurations, even if some are hidden from the other charts."
                        )
                        hist_title = f"Tracked-card position distribution for all configurations at step {chosen_tracked_step}"
                    else:
                        max_step_note = available_tracked_steps[-1] if available_tracked_steps else chosen_tracked_step
                        st.caption(
                            f"This histogram is focused on configuration {tracked_label_choice}. Choose any available step up to {max_step_note} to inspect how that configuration spreads the tracked card over time."
                        )
                        hist_title = f"Tracked-card position distribution for {tracked_label_choice} at step {chosen_tracked_step}"
                    st.plotly_chart(
                        make_hist_figure(tracked_view_df, chosen_tracked_step, hist_title, sidebar_state["deck_size"]),
                        width="stretch",
                        key=f"hist_chart_{tracked_label_choice}_{chosen_tracked_step}_{render_nonce}",
                    )
            else:
                st.info("Tracked-card position results will appear here once data is available.")

        elif page_mode == "Advanced diagnostics":
            st.subheader("Advanced diagnostics")
            show_advanced_now = sidebar_state["show_advanced_graphs"]
            advanced_diag_label = sidebar_state.get("advanced_diag_label_choice")
            selected_result = latest_named_results.get(advanced_diag_label) if advanced_diag_label else None
            if show_advanced_now and selected_result:
                final_bar_fig, final_step = make_top_bottom_bar_for_config(
                    advanced_diag_label,
                    selected_result,
                    sidebar_state["deck_size"],
                )
                st.caption(
                    "These diagnostics help inspect remaining structure in the deck beyond the main mixing metrics. "
                    "The highest/lowest chart always uses the selected configuration's final completed shuffle step."
                )
                if final_step is not None:
                    st.caption(
                        f"Showing {advanced_diag_label} at its final available step, {final_step}. "
                        f"With deck size {sidebar_state['deck_size']}, the chart contains {2 * int(sidebar_state['deck_size'])} bars total."
                    )
                st.plotly_chart(
                    final_bar_fig,
                    width="stretch",
                    key=f"bar_chart_final_{advanced_diag_label}_{final_step}_{render_nonce}",
                )
                if not pos_df.empty:
                    st.caption("The 3D view shows how the tracked-card probability mass moves across positions over shuffle steps for the visible configurations.")
                    st.plotly_chart(
                        make_3d_figure(pos_df),
                        width="stretch",
                        key=f"plot3d_chart_{render_nonce}",
                    )
            elif not show_advanced_now:
                st.info("Advanced diagnostics are hidden. Turn them on in the sidebar display options.")
            else:
                st.info("Advanced diagnostics will appear after the simulation returns tracked-card probabilities.")

        elif page_mode == "Perfect riffle":
            st.subheader("Perfect riffle")
            perfect_cards = []
            for meta in stored_configs:
                cfg = meta["cfg"]
                steps = meta["steps_list"]
                label = meta["display_name"]
                if label in visible_labels and cfg.method == "riffle" and bool(cfg.params.get("perfect_riffle", False)):
                    max_steps = max(steps) if steps else 20
                    path_df = pd.DataFrame(deterministic_deck_path(cfg, steps=max_steps, deck_size=sidebar_state["deck_size"]))
                    cycle = find_return_to_original(cfg, deck_size=sidebar_state["deck_size"], max_steps=max(500, max_steps))
                    perfect_cards.append((label, path_df, cycle))
            if perfect_cards:
                st.caption("Perfect riffle is deterministic, so this section shows how structure evolves and whether the deck cycles back to its original order.")
                for label, path_df, cycle in perfect_cards:
                    st.plotly_chart(
                        make_perfect_riffle_runs_figure(path_df, label),
                        width="stretch",
                        key=f"perfect_chart_{label}_{render_nonce}",
                    )
                    if cycle is not None:
                        st.success(f"{label}: this perfect riffle returns to the original deck order after {cycle} shuffles for deck size {sidebar_state['deck_size']}.")
                    else:
                        st.warning(f"{label}: no full return to the original order was found within the current search limit.")
                    st.dataframe(path_df[["steps", "runs", "inversions", "returned_to_original"]], width="stretch")
            else:
                st.info("Enable perfect riffle for at least one visible riffle configuration to see deterministic-cycle diagnostics here.")

        elif page_mode == "Downloads":
            st.subheader("Downloads")
            export_metrics, export_pos = combine_results_to_frames(latest_named_results, sidebar_state["deck_size"])
            single_cfg_name = cfg_download_label(stored_configs, "shuffle")
            config_export_bytes = build_config_export_payload(sidebar_state)
            st.download_button(
                "Download metric results as CSV",
                data=export_metrics.to_csv(index=False).encode("utf-8"),
                file_name=metric_download_name(single_cfg_name, len(stored_configs)),
                mime="text/csv",
                key=f"download_metrics_csv_{render_nonce}",
            )
            st.download_button(
                "Download tracked-card position probabilities as CSV",
                data=export_pos.to_csv(index=False).encode("utf-8"),
                file_name=position_download_name(single_cfg_name, len(stored_configs)),
                mime="text/csv",
                key=f"download_positions_csv_{render_nonce}",
            )
            st.download_button(
                "Download full configuration as JSON",
                data=config_export_bytes,
                file_name=f"{single_cfg_name}_configuration.json" if len(stored_configs) == 1 else "shuffle_configurations.json",
                mime="application/json",
                key=f"download_config_json_{render_nonce}",
            )
            if sidebar_state["show_runtime_table"] and not runtime_df.empty:
                st.subheader("Runtime benchmark")
                st.dataframe(runtime_df, width="stretch")
            st.markdown(
                """

    **The cheat code enforces the intended behavior after each shuffle:**
    - `keep_top` restores the original top cards to the top,
    - `keep_bottom` restores the original bottom cards to the bottom,
    - `return_tracked_to_top` moves the selected tracked card back to the first position.

    This is why the histogram now changes in a visible way when cheat mode is enabled.

    **About fully redrawing.** With the current Streamlit + Plotly pattern, the simplest reliable approach is still to redraw the figure object compare to add points. To make that lighter, we reduce the redraw frequency with `Redraw every N live batches`, so it can update less often while still seeing the curve grow over time.
                """
            )

def run_simulation(sidebar_state, results_placeholder=None):
    """This is the execution of the app."""
    status_placeholder = st.empty()
    latest_results = {}
    configs_meta = sidebar_state["configs"]

    if sidebar_state["use_cached_final"]:
        status_placeholder.info("Running cached final-result mode (no live updates).")
        updated_configs = []
        for idx, meta in enumerate(configs_meta):
            prefix = meta["prefix"]
            method = meta["cfg"].method
            ui = ui_params_from_prefix(prefix, method)
            cached_cfg, cached_result = run_cached_result(
                method=method,
                trials=sidebar_state["trials"],
                seed=sidebar_state["seed"] + idx,
                tracked_card=sidebar_state["tracked_card"],
                deck_size=sidebar_state["deck_size"],
                steps_list=list(meta["steps_list"]),
                riffle_cut_p=ui["riffle_cut_p"],
                p_overhand=ui["p_overhand"],
                piles_k=ui["piles_k"],
                pile_random_pickup=ui["pile_random_pickup"],
                cheat_mode=ui["cheat_mode"],
                cheat_cards=ui["cheat_cards"],
                max_workers=sidebar_state["max_workers"],
                perfect_riffle=ui["perfect_riffle"],
                perfect_riffle_start=ui["perfect_riffle_start"],
            )
            latest_results[prefix] = cached_result
            updated_configs.append({**meta, "cfg": cached_cfg})
        configs_meta = updated_configs
    else:
        for idx, meta in enumerate(configs_meta):
            cfg = meta["cfg"]
            progress_counter = 0
            for latest_result in run_trials_live(
                    cfg=cfg,
                    steps_list=meta["steps_list"],
                    trials=sidebar_state["trials"],
                    seed=sidebar_state["seed"] + idx,
                    tracked_card=sidebar_state["tracked_card"],
                    deck_size=sidebar_state["deck_size"],
                    batch_size=sidebar_state["batch_size"],
                    max_workers=sidebar_state["max_workers"],
            ):
                latest_results[meta["prefix"]] = latest_result
                progress_counter += 1
                current_trials = max(v[next(iter(v))]["trials_done"] for v in latest_results.values()) if latest_results else 0
                status_placeholder.info(
                    f"Running {meta['display_name']}: {cfg.name}. Completed about {current_trials}/{sidebar_state['trials']} trials for the latest updated configuration."
                )
                should_redraw = (progress_counter % sidebar_state["refresh_every"] == 0) or (current_trials >= sidebar_state["trials"])
                if should_redraw:
                    st.session_state["latest_results"] = latest_results.copy()
                    st.session_state["last_configs_meta"] = configs_meta
                    st.session_state["runtime_df"] = pd.DataFrame()

    runtime_df = make_runtime_df(configs_meta, sidebar_state["deck_size"])
    st.session_state["latest_results"] = latest_results
    st.session_state["last_configs_meta"] = configs_meta
    st.session_state["runtime_df"] = runtime_df


# sidebar_state = build_sidebar()
# results_placeholder = st.empty()
#
# if sidebar_state["run_btn"]:
#     run_simulation(sidebar_state, results_placeholder)
#
# render_results(sidebar_state, results_placeholder)
sidebar_state = build_sidebar()

if sidebar_state["run_btn"]:
    results_placeholder = st.empty()
    run_simulation(sidebar_state, results_placeholder)

render_results(sidebar_state)
