"""Render the final scientific figures from normalized and evaluated artifacts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter

from atm.demo.replay import apply_policy, load_replay_tables, load_steps, replay_frame, replay_runs
from atm.final_results import CHECKPOINTS, FRAMEWORKS, MODELS, read

BLUE, ORANGE, GREEN, PINK, GRAY, INK = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7A8594", "#172436"
PALETTE = (PINK, GREEN, BLUE, ORANGE)
DESTINATION = Path("figures")
ROOT = Path.cwd()


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#CFDAE3", "grid.alpha": .6,
        "grid.linewidth": .5, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(DESTINATION / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 2.0))
    ax.set(xlim=(0, 10.2), ylim=(0, 2))
    ax.axis("off")
    nodes = [(0.12,"Agent logs","Task and outcome"),(1.81,"Normalize","Action tokens"),
             (3.50,"Mine + cluster","Training tasks"),(5.19,"Prefix features","Observed steps"),
             (6.88,"Predict failure","Four models"),(8.57,"Calibrated stop","Held-out tasks")]
    for i, (x, label, detail) in enumerate(nodes):
        color = BLUE if i >= 3 else GREEN
        box = FancyBboxPatch((x,.59), 1.46, .96, boxstyle="round,pad=.045,rounding_size=.11",
                             linewidth=1.2, facecolor=color + "13", edgecolor=color)
        ax.add_patch(box)
        ax.text(x+.73, 1.19, label, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x+.73, .92, detail, ha="center", va="center", color="#435873", fontsize=8)
        if i < len(nodes)-1:
            ax.annotate("", (x+1.68,1.08), (x+1.48,1.08),
                        arrowprops={"arrowstyle":"->", "lw":1.1, "color":"#516986"})
    save(fig, "01_pipeline")


def data_and_length() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_parquet("data/processed/runs.parquet")
    difficulty = pd.read_parquet("data/processed/difficulty.parquet")
    data = runs.merge(difficulty[["instance_id","difficulty"]], on="instance_id", validate="many_to_one")
    fig, axes = plt.subplots(1,3,figsize=(10.1,3.0),layout="constrained")
    by = data.groupby(["framework","failed"]).size().unstack(1).reindex(FRAMEWORKS)
    positions = np.arange(len(by))
    axes[0].bar(positions,by[0],label="Resolved",color=BLUE)
    axes[0].bar(positions,by[1],bottom=by[0],label="Unresolved",color=ORANGE)
    for x, (_, row) in enumerate(by.iterrows()):
        axes[0].text(x,row.sum()+12,f"{row[1]/row.sum():.0%} failed",ha="center",va="bottom",fontsize=8)
    axes[0].set(xticks=positions,xticklabels=["OpenHands","SWE-agent","mini-SWE"],ylabel="Runs",
                ylim=(0,by.sum(axis=1).max()*1.2))
    axes[0].legend(loc="upper right",fontsize=8)
    for label,state,color in [("Resolved",0,BLUE),("Unresolved",1,ORANGE)]:
        counts = np.sort(data.loc[data["failed"] == state,"n_steps"])
        axes[1].step(counts,np.arange(1,len(counts)+1)/len(counts),where="post",color=color,label=label)
    axes[1].set(xscale="log",xlabel="Recorded steps",ylabel="Cumulative share")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    difficulty_order = ["hard","medium","easy"]
    diff = data.groupby("difficulty")["failed"].agg(["mean","size"]).reindex(difficulty_order)
    axes[2].bar(difficulty_order,diff["mean"],color=[ORANGE,GRAY,GREEN])
    for x, (_,row) in enumerate(diff.iterrows()):
        axes[2].text(x,row["mean"]+.03,f"{int(row['size']):,} runs",ha="center",fontsize=8)
    axes[2].set(ylabel="Failure rate",ylim=(0,1.08))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    save(fig,"02_dataset")
    return runs,data


def length_confound(data: pd.DataFrame) -> None:
    bounds = [0, 10, 20, 40, 60, 100, np.inf]
    groups = ["≤10","11–20","21–40","41–60","61–100",">100"]
    bins = pd.cut(data["n_steps"], bounds, labels=groups)
    fig,ax = plt.subplots(figsize=(7.2,3.2),layout="constrained")
    labels = [("Overall",data,INK), *[(name.title(),data.loc[data["difficulty"].eq(name)],color)
              for name,color in [("hard",ORANGE),("medium",GRAY),("easy",GREEN)]]]
    correl = read("length_confound").set_index("stratum")
    for name, subset, color in labels:
        rates = subset.groupby(bins,observed=False)["failed"].mean()
        ax.plot(groups,rates,marker="o",label=f"{name} (r={correl.loc[name.lower(),'pearson_r']:+.2f})",color=color)
    ax.set(xlabel="Recorded run length (steps)",ylabel="Failure rate",ylim=(0,1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=2,fontsize=9)
    save(fig,"03_length_confound")


def patterns() -> None:
    primary = read("patterns_primary")
    robust = set(read("patterns_robust")["pattern"])
    if len(primary):
        data = primary.copy().assign(robust=lambda x: x["pattern"].isin(robust))
    else:
        data = read("patterns_top").drop_duplicates("pattern").head(12).assign(robust=False)
    data = data.loc[np.isfinite(data[["odds_ratio","odds_ci_lower","odds_ci_upper"]]).all(axis=1)]
    data = data.sort_values("odds_ratio")
    fig,ax = plt.subplots(figsize=(8.4,max(3.1, .255*max(len(data),5)+1.1)),layout="constrained")
    for i,row in enumerate(data.itertuples()):
        center, low, high = np.log([row.odds_ratio,row.odds_ci_lower,row.odds_ci_upper])
        color = BLUE if row.robust else GREEN if row.pattern in set(primary["pattern"]) else GRAY
        ax.barh(i,center,color=color,alpha=.18,height=.7)
        ax.errorbar(center,i,xerr=[[center-low],[high-center]],fmt="o",color=color,capsize=2)
    ax.set(yticks=range(len(data)),yticklabels=data["pattern"],xlabel="Log odds ratio for failure (unadjusted 95% interval)")
    ax.tick_params(axis="y",labelsize=8)
    ax.axvline(0,color="#7A8996",linewidth=.8)
    ax.legend(handles=[Patch(color=BLUE,label="Fisher + CMH + task-clustered"),
                       Patch(color=GREEN,label="Fisher + CMH")],fontsize=8,loc="lower right")
    save(fig,"04_patterns")


def archetype_map() -> None:
    table = read("archetypes")
    strips = []
    if len(table):
        for row in table.head(3).itertuples():
            key = json.loads(row.example_run_ids)[0]
            steps = load_steps(ROOT,key)
            strips.append((row.archetype_id,tuple(steps["action"].head(24))))
    fig = plt.figure(figsize=(9.3,4.7),layout="constrained")
    grid = fig.add_gridspec(2,2,height_ratios=[2,1])
    sizes, heat, examples = fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])
    pos = np.arange(len(table))
    sizes.barh(pos,table["size"],color=BLUE)
    sizes.set(yticks=pos,yticklabels=table["archetype_id"],xlabel="Failed runs")
    if len(table):
        values = np.array([[json.loads(row.framework_mix).get(name,0) for name in FRAMEWORKS] for row in table.itertuples()])
        heat.grid(False)
        heat.imshow(values,aspect="auto",cmap="GnBu")
        heat.set(xticks=range(3),xticklabels=["OpenHands","SWE-agent","mini-SWE"],yticks=pos,yticklabels=table["archetype_id"])
        for (i,j),v in np.ndenumerate(values):
            heat.text(j,i,str(v),ha="center",va="center",color="white" if v>.57*values.max() else INK)
    examples.axis("off")
    examples.set(xlim=(-2.5,25.0),ylim=(-.9,max(1,len(strips))+0.1))
    examples.set_title("Observed action sequences from representative runs",fontsize=10,loc="left")
    palette = {"view":BLUE,"edit":ORANGE,"run_script":GREEN,"search":"#56B4E9","create":PINK,
               "run_test":"#B38B09","submit":"#323D5B"}
    for i,(name,sequence) in enumerate(strips):
        y=len(strips)-i-1
        examples.text(-.2,y+.22,name,ha="right",va="center",fontsize=8)
        for j,token in enumerate(sequence):
            examples.add_patch(Rectangle((j,y),.91,.44,facecolor=palette.get(token,GRAY)))
    used = sorted({token for _,seq in strips for token in seq})
    if used:
        examples.legend(handles=[Patch(color=palette.get(x,GRAY),label=x.replace("_"," ")) for x in used],
                        loc="lower left",bbox_to_anchor=(.06,-.06),ncol=6,fontsize=7)
    save(fig,"05_archetypes")


def scaling() -> None:
    frame = read("lsh_window_scaling")
    fig,axes=plt.subplots(1,3,figsize=(10.2,3),layout="constrained")
    exact=frame.loc[frame["exact_status"].eq("completed")]
    def slope(population, field):
        return np.polyfit(np.log(population.windows), np.log(population[field]), 1)[0]
    axes[0].loglog(frame.windows,frame.lsh_candidates,"o-",color=BLUE,
                   label=f"LSH candidates · slope {slope(frame, 'lsh_candidates'):.2f}")
    axes[0].loglog(frame.windows,frame.all_pairs,"s--",color=ORANGE,
                   label=f"All pairs · slope {slope(frame, 'all_pairs'):.2f}")
    axes[0].set(ylabel="Window pairs",title="Pair workload")
    axes[0].legend(fontsize=7)
    axes[1].loglog(frame.windows,frame.lsh_seconds,"o-",color=BLUE,
                   label=f"Python LSH · slope {slope(frame, 'lsh_seconds'):.2f}")
    axes[1].loglog(exact.windows,exact.exact_seconds,"s--",color=ORANGE,
                   label=f"Compiled exact · slope {slope(exact, 'exact_seconds'):.2f}")
    axes[1].set(ylabel="Seconds",title="Wall time")
    axes[1].legend(fontsize=7)
    axes[2].semilogx(exact.windows,exact.recall,"o-",color=GREEN)
    axes[2].set(ylabel="Exact-pair recall",ylim=(.78,1.015),title="Retrieval quality")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    ticks=sorted(set([int(frame.windows.min()),int(frame.windows.max())] + ([int(frame.windows.iloc[1])] if len(frame)>2 else [])))
    for ax in axes:
        ax.set(xlabel="Windows",xticks=ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
        ax.xaxis.set_minor_locator(NullLocator())
    save(fig,"06_lsh_pairs")


def cohort_curves() -> None:
    metrics = read("fixed_cohort_metrics")
    fig,axes=plt.subplots(1,2,figsize=(8,3.05),layout="constrained")
    for ax,metric,label in zip(axes,("auroc","pr_auc"),("AUROC","PR-AUC"),strict=True):
        for (key,name),color in zip(MODELS.items(),PALETTE,strict=True):
            rows=metrics.loc[metrics["model"].eq(key)].sort_values("checkpoint")
            t=rows["checkpoint"];mean=rows[metric+"_mean"];spread=rows[metric+"_std"]
            ax.plot(t,mean,color=color,marker="o",label=name)
            ax.fill_between(t,mean-spread,mean+spread,color=color,alpha=.11,linewidth=0)
        ax.set(xlabel="Observed steps",ylabel=label,xticks=CHECKPOINTS)
    axes[0].legend(fontsize=8)
    save(fig,"07_fixed_cohort")

    d=read("difficulty_signal")
    fig,ax=plt.subplots(figsize=(6.5,2.85),layout="constrained")
    width=.22
    for index,(col,name,color) in enumerate([("overall_auc","All pairs",GRAY),("within_bin_auc","Within difficulty bin",BLUE),
                                             ("within_task_auc","Within same task",GREEN)]):
        x=np.arange(len(d))+(index-1)*width
        ax.bar(x,d[col]-.5,bottom=.5,width=width,color=color,label=name)
        for a,b in zip(x,d[col],strict=True):
            ax.text(a,b+.005,f"{b:.3f}",ha="center",fontsize=8)
    ax.set(xticks=np.arange(len(d)),xticklabels=d["checkpoint"],xlabel="Observed steps",
           ylabel="LightGBM AUROC",ylim=(.5,.77))
    ax.legend(ncol=3,fontsize=8)
    save(fig,"07b_same_task_pairs")


def stop_policies() -> None:
    rows=read("early_stop").query("model == 'lightgbm'")
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout="constrained")
    for alpha,color in [(.05,BLUE),(.10,ORANGE)]:
        group=rows.loc[rows["alpha"].eq(alpha)]
        one=group.query("policy == 'single'").sort_values("checkpoint")
        many=group.query("policy == 'multi'").iloc[0]
        for ax,key,name in [(axes[0],"compute_saved","Character cost saved"),(axes[1],"false_stop_rate","False stops among successes")]:
            ax.plot(one.checkpoint,one[key],"o-",color=color,label=f"α = {alpha:.0%}")
            ax.scatter([47],[many[key]],color=color,marker="D",zorder=3)
            ax.set(xlabel="Single step or multi-step policy",ylabel=name,
                   xticks=[*CHECKPOINTS,47],xticklabels=[*CHECKPOINTS,"All"])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].axhline(alpha,color=color,lw=.6,ls=":")
        axes[1].errorbar(one.checkpoint,one.false_stop_rate,
                         yerr=[one.false_stop_rate-one.false_stop_ci_low,one.false_stop_ci_high-one.false_stop_rate],
                         fmt="none",ecolor=color,alpha=.45,capsize=2)
        axes[1].errorbar([47],[many.false_stop_rate],yerr=[[many.false_stop_rate-many.false_stop_ci_low],
                         [many.false_stop_ci_high-many.false_stop_rate]],fmt="none",ecolor=color,capsize=2)
    axes[0].legend()
    save(fig,"08_compute_saved")


def ablation() -> None:
    rows=read("ablation").query("checkpoint == 20 and dropped not in ['none','archetypes']")
    revised=read("archetype_ablation_refit").query("checkpoint == 20").assign(dropped="archetypes (refit)")
    rows=pd.concat([rows,revised],ignore_index=True).sort_values("delta_auroc_mean")
    fig,ax=plt.subplots(figsize=(6.8,3),layout="constrained")
    x=np.arange(len(rows))
    ax.barh(x,rows.delta_auroc_mean,color=[ORANGE if val<0 else BLUE for val in rows.delta_auroc_mean])
    ax.axvline(0,color=INK,lw=.8)
    ax.set(yticks=x,yticklabels=[v.replace("_"," ") for v in rows.dropped],
           xlabel="Δ AUROC at t=20 (dropped group minus full model)")
    save(fig,"09_ablation")


def transfer() -> None:
    rows=read("transfer_metrics").query("model == 'lightgbm' and checkpoint in [10,20]")
    matrix_=rows.pivot(index="held_framework",columns="checkpoint",values="auroc_mean").reindex(FRAMEWORKS)
    fig,ax=plt.subplots(figsize=(6.6,3.0),layout="constrained")
    ax.grid(False)
    norm=Normalize(vmin=.48,vmax=.72)
    im=ax.imshow(matrix_,cmap="Blues",norm=norm,aspect="auto")
    ax.set(xticks=[0,1],xticklabels=["t=10","t=20"],yticks=range(3),
           yticklabels=[f"Hold out {name}" for name in FRAMEWORKS])
    for (i,j),val in np.ndenumerate(matrix_.to_numpy()):
        ax.text(j,i,f"{val:.3f}",ha="center",va="center",color="white" if val>.63 else INK,weight="bold")
    fig.colorbar(im,ax=ax,label="AUROC")
    save(fig,"10_transfer")


def replay() -> None:
    runs,scores,gates=load_replay_tables(ROOT)
    examples=replay_runs(runs,scores,gates)
    fig,axes=plt.subplots(1,len(examples),figsize=(9.5,2.9),sharey=True,layout="constrained")
    for ax,run_id in zip(np.atleast_1d(axes),examples,strict=True):
        run=runs.loc[runs["run_id"].eq(run_id)].iloc[0]
        frame=replay_frame(scores,gates,run_id,.10)
        cut=frame.loc[np.isfinite(frame["threshold"])]
        ax.plot(frame.checkpoint,frame.score,"o-",color=BLUE)
        ax.plot(cut.checkpoint,cut.threshold,"s--",color=ORANGE)
        stopped=apply_policy(frame,int(run.n_steps))
        ax.set(xlabel="Observed steps",ylim=(0,1.04),xticks=[5,20,40])
        ax.set_title(f'{"Unresolved" if run.failed else "Resolved"} · {run.n_steps} steps',fontsize=9)
        if stopped is not None:
            ax.axvline(stopped,color=PINK,ls=":",lw=1.6)
            ax.text(.05,.1,f"Policy stop at {stopped}",transform=ax.transAxes,fontsize=8)
        else:
            ax.text(.05,.1,"Policy continues",transform=ax.transAxes,fontsize=8)
    axes=np.atleast_1d(axes)
    axes[0].set_ylabel("Failure probability")
    axes[-1].legend(handles=[Line2D([0],[0],color=BLUE,label="LightGBM"),Line2D([0],[0],color=ORANGE,ls="--",label="Threshold")],
                    loc="upper right",fontsize=8)
    save(fig,"11_replay")


def main() -> None:
    style()
    DESTINATION.mkdir(exist_ok=True)
    pipeline()
    _, data = data_and_length()
    length_confound(data)
    patterns()
    archetype_map()
    scaling()
    cohort_curves()
    stop_policies()
    ablation()
    transfer()
    replay()


if __name__ == "__main__":
    main()
