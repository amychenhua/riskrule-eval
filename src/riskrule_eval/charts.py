"""报告图表（matplotlib，输出 PNG）。

系统里找不到中文字体时，图上的文字自动改用英文，避免出现方块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#8a8983"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

_CJK_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "PingFang SC",
    "SimHei",
    "Heiti SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]


def _find_cjk() -> Optional[str]:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in _CJK_CANDIDATES:
        if c in names:
            return c
    return None


CJK_FONT = _find_cjk()
HAS_CJK = CJK_FONT is not None

plt.rcParams.update(
    {
        "font.sans-serif": ([CJK_FONT] if CJK_FONT else []) + ["DejaVu Sans", "Arial"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 9.5,
    }
)


def T(cn: str, en: str) -> str:
    return cn if HAS_CJK else en


def _style(ax, grid_axis: str = "y") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _money(v: float) -> str:
    a = abs(v)
    if HAS_CJK:
        if a >= 1e8:
            return f"{v / 1e8:.2f}亿"
        if a >= 1e4:
            return f"{v / 1e4:.1f}万"
        return f"{v:,.0f}"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def lift_bars(single: pd.DataFrame, path: Path) -> Optional[Path]:
    if single.empty:
        return None
    d = single.iloc[::-1]
    labels = (d["规则ID"] + " " + d["规则名称"]) if HAS_CJK else d["规则ID"]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.4))
    ax.barh(labels, d["提升度"].fillna(0), color=BLUE, height=0.55)
    ax.axvline(1.0, color=NEUTRAL, linestyle="--", linewidth=1)
    for y, v in enumerate(d["提升度"].fillna(0)):
        ax.text(v, y, f" {v:.2f}", va="center", color=INK2)
    ax.set_title(T("单规则提升度（虚线为总体水平 1.0）", "Lift by rule (dashed = base rate)"))
    _style(ax, "x")
    return _save(fig, path)


def marginal_bars(marg: pd.DataFrame, path: Path) -> Optional[Path]:
    if marg.empty:
        return None
    d = marg.iloc[::-1]
    labels = (d["规则ID"] + " " + d["规则名称"]) if HAS_CJK else d["规则ID"]
    colors = [BLUE if v >= 0 else ORANGE for v in d["边际贡献"]]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.4))
    ax.barh(labels, d["边际贡献"], color=colors, height=0.55)
    ax.axvline(0, color=NEUTRAL, linewidth=1)
    for y, v in enumerate(d["边际贡献"]):
        ax.text(v, y, f" {_money(v)} " if v >= 0 else f" {_money(v)} ", va="center", ha="left" if v >= 0 else "right", color=INK2)
    name = marg.attrs.get("strategy", "")
    ax.set_title(T(f"边际贡献：从「{name}」中剔除该规则后净收益减少多少", "Marginal net benefit (leave-one-out)"))
    ax.margins(x=0.18)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v)))
    _style(ax, "x")
    return _save(fig, path)


def net_curve(scan: pd.DataFrame, path: Path) -> Optional[Path]:
    if scan.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(scan["阈值"], scan["净收益"], color=BLUE, linewidth=2, marker="o", markersize=5, markerfacecolor=SURFACE, markeredgewidth=1.5)
    bx, by = scan.attrs["best"], scan.attrs["best_net"]
    ax.plot([bx], [by], marker="o", markersize=9, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(
        T(f"最优 {bx:g}：{_money(by)}", f"best {bx:g}: {_money(by)}"),
        (bx, by),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        color=INK,
    )
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    ax.set_xlabel(T(f"阈值（{scan.attrs['field']} {scan.attrs['op']} 阈值 即命中）", f"threshold ({scan.attrs['field']} {scan.attrs['op']} t)"))
    ax.set_title(T("净收益随阈值变化", "Net benefit by threshold"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v)))
    ax.margins(y=0.15)
    _style(ax)
    return _save(fig, path)


def waterfall(row: pd.Series, title: str, path: Path) -> Path:
    av, op, oc, net = float(row["避免损失"]), float(row["机会成本"]), float(row["运营成本"]), float(row["净收益"])
    names = [T("避免损失", "Avoided loss"), T("机会成本", "Opportunity cost"), T("运营成本", "Operating cost"), T("净收益", "Net benefit")]
    bottoms = [0, av - op, av - op - oc, min(0, net)]
    heights = [av, op, oc, abs(net)]
    colors = [BLUE, ORANGE, ORANGE, INK2]
    texts = [f"+{_money(av)}", f"−{_money(op)}", f"−{_money(oc)}", _money(net)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(names, heights, bottom=bottoms, color=colors, width=0.5)
    levels = [av, av - op, av - op - oc]
    for i, lv in enumerate(levels):
        ax.plot([i + 0.25, i + 0.75], [lv, lv], color=NEUTRAL, linewidth=1, linestyle="--")
    span = max(av, 1.0)
    for i, (b, h, t) in enumerate(zip(bottoms, heights, texts)):
        ax.text(i, b + h + span * 0.015, t, ha="center", va="bottom", color=INK)
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v)))
    ax.margins(y=0.12)
    _style(ax)
    return _save(fig, path)


def lorenz(disc: Dict[str, Any], path: Path) -> Optional[Path]:
    if not disc:
        return None
    pts = disc["points"]
    x = np.r_[0, pts["pop_share"].to_numpy()]
    y = np.r_[0, pts["pos_share"].to_numpy()]
    br = disc["base_rate"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.fill_between(x, y, x, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(x, y, color=BLUE, linewidth=2, label=T(f"评分 {disc['field']}", f"score {disc['field']}"))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1.2, linestyle="--", label=T("随机排序", "random"))
    ax.plot([0, br, 1], [0, 1, 1], color=NEUTRAL, linewidth=1, linestyle=":", label=T("完美排序", "perfect"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(T("累计样本占比（风险从高到低）", "cumulative share of population (riskiest first)"))
    ax.set_ylabel(T("累计正样本占比", "cumulative share of positives"))
    ax.set_title(T(f"洛伦兹曲线　KS {disc['ks']:.3f}　AUC {disc['auc']:.3f}", f"Lorenz curve  KS {disc['ks']:.3f}  AUC {disc['auc']:.3f}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "both")
    return _save(fig, path)


def heatmap(matrix: pd.DataFrame, path: Path) -> Optional[Path]:
    if matrix.empty:
        return None
    rows = matrix.iloc[:, 0].tolist()
    cols = matrix.columns[1:].tolist()
    vals = matrix.iloc[:, 1:].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 2.6, 0.62 * len(rows) + 1.8))
    im = ax.imshow(vals, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    best = np.unravel_index(np.nanargmax(vals), vals.shape)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            dark = (vals[i, j] - lo) / ((hi - lo) or 1) > 0.5
            star = " ★" if (i, j) == best else ""
            ax.text(j, i, _money(vals[i, j]) + star, ha="center", va="center", color="#ffffff" if dark else INK, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(T("决策矩阵：两个阈值任一命中时的净收益（★ 最高）", "Decision matrix: net benefit when either condition hits (★ best)"))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=plt.FuncFormatter(lambda v, _: _money(v)))
    return _save(fig, path)


def render_all(res: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    """把结果字典画成图，返回 {名称: 路径}。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: Dict[str, Path] = {}

    def add(name: str, p: Optional[Path]) -> None:
        if p is not None:
            made[name] = p

    add("单规则提升度", lift_bars(res["single_rules"], out_dir / "lift_by_rule.png"))
    add("边际贡献", marginal_bars(res["marginal"], out_dir / "marginal.png"))
    st = res["strategies"]
    if len(st) >= 2:
        row = st.iloc[1]
        add("净收益瀑布图", waterfall(row, T(f"「{row['策略']}」净收益拆解", "Net benefit breakdown"), out_dir / "waterfall.png"))
    elif len(st) == 1:
        row = st.iloc[0]
        add("净收益瀑布图", waterfall(row, T(f"「{row['策略']}」净收益拆解", "Net benefit breakdown"), out_dir / "waterfall.png"))
    add("阈值扫描净收益曲线", net_curve(res["scan"], out_dir / "net_curve.png"))
    add("洛伦兹曲线", lorenz(res["discrimination"], out_dir / "lorenz.png"))
    add("决策矩阵热力图", heatmap(res["matrix"], out_dir / "heatmap.png"))
    return made
