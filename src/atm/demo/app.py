"""Streamlit replay of frozen LightGBM checkpoints and conformal decisions."""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from atm.demo.replay import (apply_policy, get_trace, load_replay_tables, load_steps, model_bundle,
                             predict_prefix, replay_frame, replay_runs)

ROOT = Path.cwd()


@st.cache_data
def catalog():
    return load_replay_tables(ROOT)


@st.cache_data
def steps_for(run_id: str):
    return load_steps(ROOT, run_id)


@st.cache_resource
def trained(fold: int, checkpoint: int):
    return model_bundle(ROOT, fold, checkpoint)


def main() -> None:
    st.set_page_config(page_title="Held-out run replay", layout="wide")
    st.title("Held-out run replay")
    st.caption("Inspect a saved LightGBM failure score and the calibrated stopping policy.")
    try:
        runs, scores, gates = catalog()
    except FileNotFoundError:
        st.error("Run make all to build the held-out scores and saved models.")
        return
    examples = replay_runs(runs, scores, gates)
    options = examples + sorted(set(scores["run_id"]) - set(examples))
    if not options:
        st.warning("No held-out runs are available.")
        return
    selected = st.selectbox("Run", options=options, format_func=lambda key: key.replace("::", " · ", 1))
    record = runs.loc[runs["run_id"].eq(selected)].iloc[0]
    controls = st.columns([3, 1])
    alpha = controls[1].radio("False-stop target", [.10, .05], format_func=lambda x: f"{x:.0%}")
    frame = replay_frame(scores, gates, selected, alpha)
    fold = int(frame.iloc[0]["fold"])
    n_steps = int(record["n_steps"])
    position = controls[0].slider("Observed steps", 1, n_steps, min(5, n_steps), key="position-" + selected)
    trace_table = steps_for(selected)
    trace = get_trace(trace_table)
    visible = frame.loc[frame["checkpoint"].le(position)].copy()
    visible["score"] = [predict_prefix(trained(fold, int(t)), trace, int(t)) for t in visible["checkpoint"]]
    stop = apply_policy(visible, n_steps)
    col = st.columns(3)
    if visible.empty:
        col[0].metric("Failure probability", "First score at step 5")
        col[1].metric("Threshold", "—")
    else:
        latest = visible.iloc[-1]
        col[0].metric("Failure probability", f"{latest.score:.1%}", help=f"Score from step {int(latest.checkpoint)}")
        col[1].metric("Threshold", f"{latest.threshold:.1%}" if np.isfinite(latest.threshold) else "Stop disabled")
    col[2].metric("Policy", f"Stop at step {stop}" if stop else "Continue" if position < n_steps else "Completed")
    if position == n_steps:
        st.write(f"Recorded outcome · {'Unresolved' if record.failed else 'Resolved'} at step {n_steps}")
    if stop is not None:
        st.write(f"The policy crosses the threshold at step {stop}. The original run continues through step {n_steps}.")
    domain = [1, max(n_steps, 2)]
    layers = [
        alt.Chart(visible).mark_line(point=True, color="#0072B2").encode(
            x=alt.X("checkpoint:Q", title="Observed steps", scale=alt.Scale(domain=domain)),
            y=alt.Y("score:Q", title="Failure probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=["checkpoint", alt.Tooltip("score:Q", format=".3f")],
        ),
        alt.Chart(visible.loc[np.isfinite(visible["threshold"])]).mark_line(point=True, strokeDash=[5,4], color="#D55E00").encode(
            x="checkpoint:Q", y="threshold:Q", tooltip=["checkpoint", alt.Tooltip("threshold:Q", format=".3f")],
        ),
    ]
    if stop is not None:
        layers.append(alt.Chart(pd.DataFrame({"step": [stop]})).mark_rule(color="#7A3B69").encode(x="step:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=300), use_container_width=True)
    st.caption("Solid blue · saved checkpoint model. Dashed orange · split-conformal threshold. Between checkpoints the last score stays visible.")
    with st.expander("Observed action tokens", expanded=True):
        st.dataframe(trace_table.loc[trace_table.step_idx.lt(position), ["step_idx","token","outcome"]].assign(
            Step=lambda frame: frame.step_idx + 1
        )[["Step","token","outcome"]].rename(columns={"token":"Action token","outcome":"Outcome"}), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
