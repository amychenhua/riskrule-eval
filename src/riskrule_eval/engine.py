"""评估引擎：六个场景共用同一套计算。

核心口径（每个场景的取值模型把自己的字段换算成这三项）::

    净收益 = 避免损失 − 机会成本 − 运营成本
    避免损失 = Σ 命中且为正样本的 avoided_loss
    机会成本 = Σ 命中且为负样本的 opportunity_cost
    运营成本 = Σ 命中样本的 unit_cost

“正样本”指场景标签为 1 的记录（可疑交易、欺诈、坏账、违约）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics
from .expr import ExprError, evaluate_mask, evaluate_number
from .schema import Scenario, expand_values

_OPS = {
    "<": lambda s, t: s < t,
    "<=": lambda s, t: s <= t,
    ">": lambda s, t: s > t,
    ">=": lambda s, t: s >= t,
}


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


class Evaluator:
    """对一份校验过的数据评估一套策略。

    Parameters
    ----------
    scenario : Scenario
    df : DataFrame
        经过 :func:`riskrule_eval.validate.validate` 转换类型后的数据。
    strategy : dict, optional
        策略定义（rules / strategies / scan / matrix / monitor），缺省用场景自带的示例策略。
    params : dict, optional
        覆盖取值模型里的参数，例如 ``{"unit_cost": 20}``。
    """

    def __init__(
        self,
        scenario: Scenario,
        df: pd.DataFrame,
        strategy: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sc = scenario
        self.df = df.reset_index(drop=True)
        self.strategy = strategy if strategy is not None else scenario.default_strategy()
        self.params = {**scenario.default_params(), **(self.strategy.get("params") or {}), **(params or {})}
        self.warnings: List[str] = []

        self.y = pd.to_numeric(self.df[scenario.label_field], errors="coerce").fillna(0).astype(int).to_numpy()
        self.n = len(self.df)
        self.n_pos = int(self.y.sum())
        self.n_neg = self.n - self.n_pos
        self.base_rate = _div(self.n_pos, self.n)

        vm = scenario.value_model
        self.avoid = evaluate_number(vm["avoided_loss"], self.df, self.params)
        self.opp = evaluate_number(vm["opportunity_cost"], self.df, self.params)
        self.unit = evaluate_number(vm["unit_cost"], self.df, self.params)

        self.rules: Dict[str, Dict[str, Any]] = {}
        self.masks: Dict[str, np.ndarray] = {}
        for r in self.strategy.get("rules") or []:
            rid = str(r["id"])
            try:
                self.masks[rid] = evaluate_mask(r["expr"], self.df, self.params)
                self.rules[rid] = r
            except ExprError as e:
                self.warnings.append(f"规则 {rid} 已跳过：{e}")

        self.strategies: Dict[str, Dict[str, Any]] = {}
        for key, s in (self.strategy.get("strategies") or {}).items():
            ids = [str(i) for i in s.get("rules", [])]
            missing = [i for i in ids if i not in self.masks]
            if missing:
                self.warnings.append(f"策略 {key} 引用的规则 {', '.join(missing)} 不可用，已从该策略中去掉")
            self.strategies[key] = {"name": s.get("name", key), "rules": [i for i in ids if i in self.masks]}

    # ------------------------------------------------------------------ 基础
    def stats(self, mask: np.ndarray, loss_mult: float = 1.0, profit_mult: float = 1.0, cost_mult: float = 1.0) -> Dict[str, float]:
        """一组命中样本的全部指标。三个乘数用于敏感性分析。"""
        mask = np.asarray(mask, dtype=bool)
        pos = mask & (self.y == 1)
        neg = mask & (self.y == 0)
        n, tp, fp = int(mask.sum()), int(pos.sum()), int(neg.sum())
        avoided = float(self.avoid[pos].sum()) * loss_mult
        opp = float(self.opp[neg].sum()) * profit_mult
        op_cost = float(self.unit[mask].sum()) * cost_mult
        precision = _div(tp, n)
        return {
            "n": n,
            "tp": tp,
            "fp": fp,
            "fn": self.n_pos - tp,
            "tn": self.n_neg - fp,
            "hit_rate": _div(n, self.n),
            "recall": _div(tp, self.n_pos),
            "precision": precision,
            "fpr": _div(fp, self.n_neg),
            "lift": _div(precision, self.base_rate),
            "avoided": avoided,
            "opp": opp,
            "op_cost": op_cost,
            "net": avoided - opp - op_cost,
        }

    def strategy_mask(self, key: str, exclude: Sequence[str] = ()) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        for rid in self.strategies[key]["rules"]:
            if rid not in exclude:
                m |= self.masks[rid]
        return m

    def _pair(self) -> Optional[tuple]:
        """对比用的两套策略：按策略文件里的顺序取前两个，第一个是基准。"""
        keys = list(self.strategies)
        return (keys[0], keys[1]) if len(keys) >= 2 else None

    @staticmethod
    def _row(st: Dict[str, float]) -> Dict[str, float]:
        return {
            "命中量": st["n"],
            "命中率": st["hit_rate"],
            "命中正样本(TP)": st["tp"],
            "命中负样本(FP)": st["fp"],
            "召回率": st["recall"],
            "精确率": st["precision"],
            "误伤率": st["fpr"],
            "提升度": st["lift"],
            "避免损失": st["avoided"],
            "机会成本": st["opp"],
            "运营成本": st["op_cost"],
            "净收益": st["net"],
        }

    # ------------------------------------------------------------------ 概览
    def overview(self) -> pd.DataFrame:
        t = self.df[self.sc.time_field]
        rows = [
            ("场景", f"{self.sc.name}（{self.sc.key}）"),
            ("数据粒度", self.sc.unit),
            ("规则命中后的动作", self.sc.action),
            ("正样本定义", self.sc.label_positive),
            ("标签成熟所需时间", self.sc.label_window),
            ("样本量", self.n),
            ("正样本数", self.n_pos),
            ("正样本率", f"{self.base_rate:.2%}"),
            ("数据时间范围", f"{t.min():%Y-%m-%d} 至 {t.max():%Y-%m-%d}" if t.notna().any() else "—"),
            ("可用规则数", len(self.rules)),
        ]
        for k, v in self.params.items():
            rows.append((f"参数：{self.sc.param_labels().get(k, k)}", v))
        return pd.DataFrame(rows, columns=["项目", "内容"])

    # ------------------------------------------------------------------ 单规则
    def single_rules(self) -> pd.DataFrame:
        rows = []
        for rid, r in self.rules.items():
            rows.append({"规则ID": rid, "规则名称": r.get("name", rid), "表达式": r["expr"], **self._row(self.stats(self.masks[rid]))})
        return pd.DataFrame(rows)

    def overlap(self) -> pd.DataFrame:
        """行规则命中的样本里，有多大比例同时被列规则命中。"""
        ids = list(self.rules)
        data = {}
        for b in ids:
            data[b] = [_div(int((self.masks[a] & self.masks[b]).sum()), int(self.masks[a].sum())) for a in ids]
        out = pd.DataFrame(data, index=ids)
        out.insert(0, "规则ID", ids)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ 策略对比
    def strategies_table(self) -> pd.DataFrame:
        rows = []
        for key, s in self.strategies.items():
            st = self.stats(self.strategy_mask(key))
            rows.append({"策略": s["name"], "包含规则": ", ".join(s["rules"]), **self._row(st)})
        out = pd.DataFrame(rows)
        pair = self._pair()
        if pair and len(out) >= 2:
            a, b = out.iloc[0], out.iloc[1]
            diff = {"策略": f"差异（{b['策略']} − {a['策略']}）", "包含规则": ""}
            for c in out.columns[2:]:
                diff[c] = b[c] - a[c]
            out = pd.concat([out, pd.DataFrame([diff])], ignore_index=True)
        return out

    # ------------------------------------------------------------------ 边际贡献
    def marginal(self, key: Optional[str] = None) -> pd.DataFrame:
        """去一法：把策略里的规则逐条拿掉，看净收益少了多少。"""
        if key is None:
            pair = self._pair()
            key = pair[1] if pair else next(iter(self.strategies), None)
        if key is None:
            return pd.DataFrame()
        ids = self.strategies[key]["rules"]
        full = self.stats(self.strategy_mask(key))
        rows = []
        for rid in ids:
            rest_mask = self.strategy_mask(key, exclude=[rid])
            rest = self.stats(rest_mask)
            only = self.masks[rid] & ~rest_mask
            rows.append(
                {
                    "规则ID": rid,
                    "规则名称": self.rules[rid].get("name", rid),
                    "命中量": int(self.masks[rid].sum()),
                    "独占命中量": int(only.sum()),
                    "独占正样本": int((only & (self.y == 1)).sum()),
                    "剔除后净收益": rest["net"],
                    "边际贡献": full["net"] - rest["net"],
                }
            )
        out = pd.DataFrame(rows)
        out.attrs["strategy"] = self.strategies[key]["name"]
        out.attrs["full_net"] = full["net"]
        return out

    # ------------------------------------------------------------------ Swap
    def swap(self) -> pd.DataFrame:
        pair = self._pair()
        if not pair:
            return pd.DataFrame()
        a, b = self.strategy_mask(pair[0]), self.strategy_mask(pair[1])
        na, nb = self.strategies[pair[0]]["name"], self.strategies[pair[1]]["name"]
        groups = [
            (f"Swap-in：{na}放行 → {nb}命中", ~a & b, 1),
            (f"Swap-out：{na}命中 → {nb}放行", a & ~b, -1),
            ("两套策略都命中", a & b, 0),
            ("两套策略都放行", ~a & ~b, 0),
        ]
        rows = []
        for name, m, sign in groups:
            st = self.stats(m)
            rows.append(
                {
                    "客群": name,
                    "数量": st["n"],
                    "正样本数": st["tp"],
                    "正样本率": st["precision"],
                    "相对总体的倍数": st["lift"],
                    "净收益影响": sign * st["net"] if sign else 0.0,
                }
            )
        out = pd.DataFrame(rows)
        total = out["净收益影响"].sum()
        check = self.stats(b)["net"] - self.stats(a)["net"]
        out.attrs["total"] = float(total)
        out.attrs["check_diff"] = float(total - check)
        return out

    # ------------------------------------------------------------------ 阈值扫描
    def _cond(self, spec: Dict[str, Any], value: float) -> np.ndarray:
        s = pd.to_numeric(self.df[spec["field"]], errors="coerce")
        return _OPS[spec["op"]](s, value).fillna(False).to_numpy(dtype=bool)

    def _spec_ok(self, spec: Optional[Dict[str, Any]], what: str) -> bool:
        if not spec:
            return False
        if spec["field"] not in self.df.columns:
            self.warnings.append(f"{what}已跳过：数据里没有字段 {spec['field']}")
            return False
        if spec.get("op") not in _OPS:
            self.warnings.append(f"{what}已跳过：op 只能是 {', '.join(_OPS)}")
            return False
        return True

    def scan(self) -> pd.DataFrame:
        """滑动一个连续字段的阈值，逐点计算全部指标。"""
        spec = self.strategy.get("scan")
        if not self._spec_ok(spec, "阈值扫描"):
            return pd.DataFrame()
        rows = []
        for t in expand_values(spec["values"]):
            rows.append({"阈值": t, **self._row(self.stats(self._cond(spec, t)))})
        out = pd.DataFrame(rows)
        best = out["净收益"].idxmax()
        out["是否最优"] = ["★" if i == best else "" for i in out.index]
        out.attrs.update(field=spec["field"], op=spec["op"], best=float(out.loc[best, "阈值"]), best_net=float(out.loc[best, "净收益"]))
        return out

    def matrix(self) -> pd.DataFrame:
        """两个阈值组合（任一命中即处置）下的净收益矩阵。"""
        spec = self.strategy.get("matrix")
        if not spec or not self._spec_ok(spec.get("x"), "决策矩阵") or not self._spec_ok(spec.get("y"), "决策矩阵"):
            return pd.DataFrame()
        x, y = spec["x"], spec["y"]
        xv, yv = expand_values(x["values"]), expand_values(y["values"])
        data = {}
        for a in xv:
            cx = self._cond(x, a)
            data[f"{x['field']} {x['op']} {a:g}"] = [self.stats(cx | self._cond(y, b))["net"] for b in yv]
        out = pd.DataFrame(data)
        out.insert(0, "行阈值 ＼ 列阈值", [f"{y['field']} {y['op']} {b:g}" for b in yv])
        return out

    # ------------------------------------------------------------------ 敏感性
    def sensitivity(self, mults: Sequence[float] = (0.7, 1.0, 1.3)) -> Dict[str, pd.DataFrame]:
        """损失与机会成本各乘一个系数，看两套策略的净收益差是否变号；再看运营成本放大的影响。"""
        pair = self._pair()
        if not pair:
            return {}
        a, b = self.strategy_mask(pair[0]), self.strategy_mask(pair[1])
        grid = {}
        for lm in mults:
            grid[f"避免损失 ×{lm:g}"] = [
                self.stats(b, lm, pm)["net"] - self.stats(a, lm, pm)["net"] for pm in mults
            ]
        g = pd.DataFrame(grid)
        g.insert(0, "机会成本 ＼ 避免损失", [f"机会成本 ×{pm:g}" for pm in mults])
        rows = []
        for cm in (0, 0.5, 1, 2, 5):
            sa, sb = self.stats(a, cost_mult=cm), self.stats(b, cost_mult=cm)
            rows.append(
                {
                    "运营成本系数": cm,
                    f"{self.strategies[pair[0]]['name']}净收益": sa["net"],
                    f"{self.strategies[pair[1]]['name']}净收益": sb["net"],
                    "净收益差异": sb["net"] - sa["net"],
                }
            )
        return {"grid": g, "cost": pd.DataFrame(rows)}

    # ------------------------------------------------------------------ 区分度
    def discrimination(self) -> Dict[str, Any]:
        """扫描字段（通常是评分）的 KS、AUC 和洛伦兹曲线。"""
        spec = self.strategy.get("scan")
        if not self._spec_ok(spec, "区分度分析"):
            return {}
        s = pd.to_numeric(self.df[spec["field"]], errors="coerce")
        ok = s.notna().to_numpy()
        risk = (-s if spec["op"] in ("<", "<=") else s).to_numpy()[ok]
        y = self.y[ok]
        return {
            "field": spec["field"],
            "direction": "越低风险越高" if spec["op"] in ("<", "<=") else "越高风险越高",
            "ks": metrics.ks(risk, y),
            "auc": metrics.auc(risk, y),
            "table": metrics.lorenz_table(risk, y),
            "points": metrics.lorenz_points(risk, y),
            "base_rate": _div(int(y.sum()), len(y)),
        }

    # ------------------------------------------------------------------ 监控
    def monitor(self) -> Dict[str, pd.DataFrame]:
        """按月的命中率趋势，以及监控字段相对首月的 PSI。"""
        t = self.df[self.sc.time_field]
        if t.isna().all():
            self.warnings.append("月度监控已跳过：时间字段为空")
            return {}
        month = t.dt.to_period("M").astype(str).to_numpy()
        months = sorted(m for m in pd.unique(month) if m != "NaT")
        strat_masks = {s["name"]: self.strategy_mask(k) for k, s in self.strategies.items()}
        rows = []
        for m in months:
            sel = month == m
            row = {"月份": m, "样本量": int(sel.sum()), "正样本数": int(self.y[sel].sum()), "正样本率": _div(int(self.y[sel].sum()), int(sel.sum()))}
            for name, sm in strat_masks.items():
                row[f"{name}命中率"] = _div(int((sm & sel).sum()), int(sel.sum()))
            rows.append(row)
        out: Dict[str, pd.DataFrame] = {"monthly": pd.DataFrame(rows)}

        spec = self.strategy.get("monitor") or {}
        fld = spec.get("psi_field")
        if fld and fld in self.df.columns and len(months) >= 2:
            v = pd.to_numeric(self.df[fld], errors="coerce")
            base = v[month == months[0]]
            edges = spec.get("bins") or metrics.quantile_edges(base, int(spec.get("n_bins", 5)))
            prow, last = [], None
            for m in months[1:]:
                val, table = metrics.psi(base, v[month == m], edges)
                prow.append({"监控月份": m, "基准月份": months[0], "监控字段": fld, "PSI": val, "判读": metrics.psi_level(val)})
                last = table
            out["psi"] = pd.DataFrame(prow)
            if last is not None:
                last.attrs["month"] = months[-1]
                out["psi_detail"] = last
        elif fld and fld not in self.df.columns:
            self.warnings.append(f"PSI 已跳过：数据里没有字段 {fld}")
        return out

    # ------------------------------------------------------------------ 汇总
    def run_all(self) -> Dict[str, Any]:
        """跑完全部分析，返回结果字典，供报告模块使用。"""
        res: Dict[str, Any] = {
            "overview": self.overview(),
            "single_rules": self.single_rules(),
            "overlap": self.overlap(),
            "strategies": self.strategies_table(),
            "marginal": self.marginal(),
            "swap": self.swap(),
            "scan": self.scan(),
            "matrix": self.matrix(),
            "sensitivity": self.sensitivity(),
            "discrimination": self.discrimination(),
            "monitor": self.monitor(),
        }
        res["warnings"] = list(dict.fromkeys(self.warnings))
        return res
