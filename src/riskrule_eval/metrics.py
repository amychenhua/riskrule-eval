"""基础指标：区分度（KS、AUC、洛伦兹曲线）和稳定性（PSI）。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PSI_EPS = 1e-4


def auc(risk: np.ndarray, y: np.ndarray) -> float:
    """AUC，risk 越大越危险。用秩和计算，并列取平均秩。"""
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(risk).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def lorenz_points(risk: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """按风险从高到低排序后的累计占比，同分的样本合并成一个点。"""
    d = pd.DataFrame({"risk": np.asarray(risk, dtype=float), "y": np.asarray(y).astype(int)}).dropna()
    g = d.groupby("risk")["y"].agg(["size", "sum"]).sort_index(ascending=False)
    n, pos = g["size"].sum(), g["sum"].sum()
    neg = n - pos
    cum_n, cum_p = g["size"].cumsum(), g["sum"].cumsum()
    out = pd.DataFrame(
        {
            "pop_share": cum_n / n,
            "pos_share": cum_p / pos if pos else np.nan,
            "neg_share": (cum_n - cum_p) / neg if neg else np.nan,
        }
    ).reset_index()
    return out


def ks(risk: np.ndarray, y: np.ndarray) -> float:
    pts = lorenz_points(risk, y)
    if pts.empty:
        return float("nan")
    return float((pts["pos_share"] - pts["neg_share"]).abs().max())


def lorenz_table(risk: np.ndarray, y: np.ndarray, steps: int = 20) -> pd.DataFrame:
    """每隔 1/steps 人口占比读一个点：取累计人口占比首次达到该分位的位置。"""
    pts = lorenz_points(risk, y)
    rows = []
    for i in range(1, steps + 1):
        target = i / steps
        hit = pts[pts["pop_share"] >= target - 1e-12]
        r = hit.iloc[0] if len(hit) else pts.iloc[-1]
        rows.append(
            {
                "目标人口占比": target,
                "实际累计人口占比": float(r["pop_share"]),
                "累计正样本占比": float(r["pos_share"]),
                "该点提升度": float(r["pos_share"] / r["pop_share"]) if r["pop_share"] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def quantile_edges(values: Sequence[float], bins: int = 5) -> List[float]:
    """用基准样本的分位数生成分箱边界，首尾放开到无穷。"""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    qs = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    return [-np.inf, *qs.tolist(), np.inf]


def psi(base: Sequence[float], cur: Sequence[float], edges: Sequence[float]) -> Tuple[float, pd.DataFrame]:
    """群体稳定性指数。分箱为左开右闭；占比为 0 的箱用一个很小的数代替，避免取对数出错。"""
    b = pd.to_numeric(pd.Series(base), errors="coerce").dropna()
    c = pd.to_numeric(pd.Series(cur), errors="coerce").dropna()
    edges = list(edges)
    bc = pd.cut(b, edges, include_lowest=True).value_counts(sort=False)
    cc = pd.cut(c, edges, include_lowest=True).value_counts(sort=False)
    pb = (bc / max(len(b), 1)).to_numpy()
    pc = (cc / max(len(c), 1)).to_numpy()
    pb_, pc_ = np.where(pb == 0, PSI_EPS, pb), np.where(pc == 0, PSI_EPS, pc)
    contrib = (pc_ - pb_) * np.log(pc_ / pb_)
    table = pd.DataFrame(
        {
            "分箱": [str(i) for i in bc.index],
            "基准样本数": bc.to_numpy(),
            "基准占比": pb,
            "监控样本数": cc.to_numpy(),
            "监控占比": pc,
            "分箱贡献": contrib,
        }
    )
    return float(contrib.sum()), table


def psi_level(v: Optional[float]) -> str:
    if v is None or np.isnan(v):
        return "—"
    if v < 0.1:
        return "稳定"
    if v < 0.25:
        return "有迁移，加强观察"
    return "显著变化，需重评阈值"
