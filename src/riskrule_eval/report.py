"""评估报告：一个 Excel 工作簿、一组 PNG 图表、一份 Markdown 摘要。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .schema import Scenario
from .validate import MIN_POSITIVES, ValidationReport

FONT = "Arial"
_HEAD = PatternFill("solid", fgColor="1F4E78")
_HL = PatternFill("solid", fgColor="D9E1F2")
_THIN = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

_PCT = ("率", "占比")
_MONEY = ("损失", "成本", "净收益", "边际贡献")
_RATIO = ("提升度", "倍数")
_FOUR = ("PSI", "分箱贡献")


def _fmt_for(col: str) -> Optional[str]:
    if any(k in col for k in _FOUR):
        return "0.0000"
    if any(k in col for k in _RATIO):
        return "0.00"
    if any(k in col for k in _MONEY):
        return "#,##0;[Red]-#,##0"
    if any(k in col for k in _PCT):
        return "0.00%"
    return None


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


class _Sheet:
    """按块往下写：标题、表格、备注。"""

    def __init__(self, wb: Workbook, name: str) -> None:
        self.ws = wb.create_sheet(name)
        self.ws.sheet_view.showGridLines = False
        self.row = 1
        self.widths: Dict[int, float] = {}

    def title(self, text: str, size: int = 13) -> None:
        c = self.ws.cell(row=self.row, column=1, value=text)
        c.font = Font(name=FONT, bold=True, size=size, color="1F4E78")
        self.row += 1

    def note(self, text: str) -> None:
        c = self.ws.cell(row=self.row, column=1, value=text)
        c.font = Font(name=FONT, italic=True, size=9, color="808080")
        self.row += 1

    def gap(self, n: int = 1) -> None:
        self.row += n

    def table(self, df: pd.DataFrame, fmt_all: Optional[str] = None, highlight: Optional[Sequence[int]] = None) -> None:
        if df is None or df.empty:
            self.note("（无数据）")
            return
        ws = self.ws
        for j, col in enumerate(df.columns, 1):
            c = ws.cell(row=self.row, column=j, value=str(col))
            c.font = Font(name=FONT, bold=True, color="FFFFFF")
            c.fill = _HEAD
            c.border = _BORDER
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            self.widths[j] = max(self.widths.get(j, 0), min(40, len(str(col)) * 1.9 + 2))
        ws.row_dimensions[self.row].height = 30
        self.row += 1
        hl = set(highlight or [])
        for i, rec in enumerate(df.itertuples(index=False)):
            for j, (col, v) in enumerate(zip(df.columns, rec), 1):
                v = _clean(v)
                c = ws.cell(row=self.row, column=j, value=v)
                c.font = Font(name=FONT, bold=i in hl)
                c.border = _BORDER
                if i in hl:
                    c.fill = _HL
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    f = (fmt_all if j > 1 else None) or _fmt_for(str(col))
                    if f:
                        c.number_format = f
                    elif isinstance(v, float):
                        c.number_format = "#,##0.00"
                else:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
                    self.widths[j] = max(self.widths.get(j, 0), min(60, len(str(v or "")) * 1.6 + 2))
            self.row += 1
        self.row += 1

    def finish(self) -> None:
        for j, w in self.widths.items():
            self.ws.column_dimensions[get_column_letter(j)].width = max(11, w)


def findings(res: Dict[str, Any], scenario: Scenario, vrep: Optional[ValidationReport] = None) -> List[str]:
    """把结果翻译成几句话的结论。"""
    out: List[str] = []
    ov = dict(zip(res["overview"]["项目"], res["overview"]["内容"]))
    n_pos = int(ov["正样本数"])
    out.append(f"样本 {int(ov['样本量']):,} 条，正样本 {n_pos:,} 个，正样本率 {ov['正样本率']}。")
    if n_pos < MIN_POSITIVES:
        out.append(f"正样本少于 {MIN_POSITIVES} 个，下面的提升度和净收益可能随样本波动而翻转，只能作参考。")

    sr = res["single_rules"]
    if not sr.empty:
        top = sr.loc[sr["提升度"].fillna(-1).idxmax()]
        out.append(f"提升度最高的规则是 {top['规则ID']} {top['规则名称']}：命中 {int(top['命中量'])} 条，精确率 {top['精确率']:.1%}，提升度 {top['提升度']:.2f}。")
        weak = sr[sr["提升度"] < 1]
        if len(weak):
            out.append("提升度低于 1（命中群体的正样本率低于总体）的规则：" + "、".join(weak["规则ID"]) + "，需要检查阈值或逻辑。")
        lose = sr[sr["净收益"] < 0]
        if len(lose):
            out.append("单独看净收益为负的规则：" + "、".join(lose["规则ID"]) + "。")

    st = res["strategies"]
    if len(st) >= 3:
        a, b, d = st.iloc[0], st.iloc[1], st.iloc[2]
        word = "多" if d["净收益"] >= 0 else "少"
        out.append(
            f"{b['策略']}净收益 {b['净收益']:,.0f} 元，{a['策略']} {a['净收益']:,.0f} 元，{b['策略']}{word} {abs(d['净收益']):,.0f} 元；"
            f"召回率 {a['召回率']:.1%} → {b['召回率']:.1%}，精确率 {a['精确率']:.1%} → {b['精确率']:.1%}，命中量 {int(a['命中量'])} → {int(b['命中量'])}。"
        )

    mg = res["marginal"]
    if not mg.empty:
        top = mg.loc[mg["边际贡献"].idxmax()]
        out.append(f"在{mg.attrs.get('strategy', '')}里，边际贡献最大的是 {top['规则ID']} {top['规则名称']}（{top['边际贡献']:,.0f} 元）。")
        neg = mg[mg["边际贡献"] < 0]
        if len(neg):
            out.append("边际贡献为负的规则：" + "、".join(f"{r['规则ID']}（{r['边际贡献']:,.0f}）" for _, r in neg.iterrows()) + "，剔除后整体净收益反而更高，应收紧、合并或淘汰。")

    sw = res["swap"]
    if not sw.empty:
        si, so = sw.iloc[0], sw.iloc[1]
        if si["数量"] and so["数量"]:
            out.append(f"Swap 分析：新增命中的 {int(si['数量'])} 条正样本率 {si['正样本率']:.1%}，新增放行的 {int(so['数量'])} 条正样本率 {so['正样本率']:.1%}（总体 {ov['正样本率']}）。")

    sc = res["scan"]
    if not sc.empty:
        b = sc[sc["是否最优"] == "★"].iloc[0]
        edge = b["阈值"] in (sc["阈值"].iloc[0], sc["阈值"].iloc[-1])
        out.append(
            f"阈值扫描：{sc.attrs['field']} {sc.attrs['op']} {b['阈值']:g} 时净收益最高，为 {b['净收益']:,.0f} 元（命中 {int(b['命中量'])} 条，召回率 {b['召回率']:.1%}）。"
            + ("最优点落在扫描范围的边界上，建议扩大扫描范围再看。" if edge else "")
        )

    dc = res["discrimination"]
    if dc:
        out.append(f"{dc['field']} 的 KS 为 {dc['ks']:.3f}，AUC 为 {dc['auc']:.3f}。")

    se = res["sensitivity"]
    if se:
        vals = se["grid"].iloc[:, 1:].to_numpy(dtype=float)
        if (vals > 0).all() or (vals < 0).all():
            out.append("敏感性分析：避免损失与机会成本各浮动 30%，两套策略的优劣关系不变。")
        else:
            out.append("敏感性分析：避免损失与机会成本浮动 30% 时，两套策略的优劣关系会翻转，结论依赖参数假设，需要先把参数估准。")

    mo = res["monitor"]
    if mo and "psi" in mo and len(mo["psi"]):
        last = mo["psi"].iloc[-1]
        out.append(f"稳定性：{last['监控字段']} 在 {last['监控月份']} 相对 {last['基准月份']} 的 PSI 为 {last['PSI']:.4f}（{last['判读']}）。")

    for w in res.get("warnings", []):
        out.append("提示：" + w)
    return out


def write_excel(
    res: Dict[str, Any],
    scenario: Scenario,
    path: Path,
    vrep: Optional[ValidationReport] = None,
    charts: Optional[Dict[str, Path]] = None,
) -> Path:
    wb = Workbook()
    wb.remove(wb.active)

    s = _Sheet(wb, "结论摘要")
    s.title(f"{scenario.name}｜规则策略收益与风险评估", 15)
    s.note("净收益 = 避免损失 − 机会成本 − 运营成本。金额取决于「概览与参数」里的参数取值，请先核对。")
    s.gap()
    for i, line in enumerate(findings(res, scenario, vrep), 1):
        c = s.ws.cell(row=s.row, column=1, value=f"{i}. {line}")
        c.font = Font(name=FONT)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        s.ws.merge_cells(start_row=s.row, start_column=1, end_row=s.row, end_column=8)
        s.ws.row_dimensions[s.row].height = 18 * max(1, int(len(line) / 62) + 1)
        s.row += 1
    s.ws.column_dimensions["A"].width = 22
    for col in "BCDEFGH":
        s.ws.column_dimensions[col].width = 16

    s = _Sheet(wb, "概览与参数")
    s.title("数据概览与参数假设")
    s.table(res["overview"])
    vm = scenario.value_model
    s.title("取值模型")
    s.table(
        pd.DataFrame(
            [
                ("避免损失（命中且为正样本）", vm["avoided_loss"]),
                ("机会成本（命中且为负样本）", vm["opportunity_cost"]),
                ("运营成本（每条命中）", vm["unit_cost"]),
            ],
            columns=["项目", "计算式"],
        )
    )
    if vrep is not None:
        s.title("数据校验")
        s.note(vrep.summary())
        s.table(vrep.to_frame())
    s.finish()

    s = _Sheet(wb, "单规则评估")
    s.title("单规则评估")
    s.note("提升度 = 精确率 ÷ 总体正样本率。净收益是该规则单独使用时的结果。")
    s.table(res["single_rules"])
    s.title("规则重叠：行规则命中的样本中，同时被列规则命中的比例")
    s.table(res["overlap"], fmt_all="0.0%")
    s.finish()

    s = _Sheet(wb, "策略对比")
    s.title("策略对比")
    st = res["strategies"]
    s.table(st, highlight=[len(st) - 1] if len(st) >= 3 else None)
    sw = res["swap"]
    if not sw.empty:
        s.title("Swap 分析：策略切换后的客群迁移")
        s.table(sw)
        s.note(f"净收益影响合计 {sw.attrs['total']:,.0f}，与上表净收益差异的差额 {sw.attrs['check_diff']:,.2f}（应为 0）。")
    s.finish()

    s = _Sheet(wb, "边际贡献")
    mg = res["marginal"]
    s.title(f"边际贡献（去一法）：{mg.attrs.get('strategy', '')}")
    if not mg.empty:
        s.note(f"全量净收益 {mg.attrs['full_net']:,.0f}。边际贡献 = 全量净收益 − 剔除该规则后的净收益；独占命中量 = 只有这条规则命中的样本数。")
    s.table(mg)
    s.finish()

    s = _Sheet(wb, "阈值扫描")
    sc = res["scan"]
    if not sc.empty:
        s.title(f"阈值扫描：{sc.attrs['field']} {sc.attrs['op']} 阈值 即命中")
        best = list(sc.index[sc["是否最优"] == "★"])
        s.table(sc, highlight=best)
    else:
        s.title("阈值扫描")
        s.table(sc)
    mx = res["matrix"]
    if not mx.empty:
        s.title("决策矩阵：行条件或列条件任一命中时的净收益")
        s.table(mx, fmt_all="#,##0;[Red]-#,##0")
    s.finish()

    se = res["sensitivity"]
    if se:
        s = _Sheet(wb, "敏感性分析")
        s.title("两套策略净收益之差（后者 − 前者），避免损失与机会成本各乘一个系数")
        s.note("九个格子正负号一致，说明结论不依赖这两项估计的精度。")
        s.table(se["grid"], fmt_all="#,##0;[Red]-#,##0")
        s.title("运营成本放大或缩小时的净收益")
        s.table(se["cost"])
        s.finish()

    dc = res["discrimination"]
    if dc:
        s = _Sheet(wb, "评分区分度")
        s.title(f"{dc['field']}（{dc['direction']}）")
        s.table(pd.DataFrame([("KS", round(dc["ks"], 4)), ("AUC", round(dc["auc"], 4))], columns=["指标", "数值"]))
        s.title("洛伦兹曲线取点（按风险从高到低排序）")
        s.table(dc["table"])
        s.finish()

    mo = res["monitor"]
    if mo:
        s = _Sheet(wb, "月度监控")
        s.title("月度趋势")
        s.table(mo["monthly"])
        if "psi" in mo:
            s.title("PSI：各月相对首月的分布稳定性")
            s.note("小于 0.1 稳定；0.1 到 0.25 有迁移，加强观察；大于 0.25 显著变化，需重评阈值。")
            s.table(mo["psi"])
            if "psi_detail" in mo:
                s.title(f"最近一个月（{mo['psi_detail'].attrs.get('month', '')}）的分箱明细")
                s.table(mo["psi_detail"])
        s.finish()

    if charts:
        ws = wb.create_sheet("图表")
        ws.sheet_view.showGridLines = False
        row = 1
        for name, p in charts.items():
            ws.cell(row=row, column=1, value=name).font = Font(name=FONT, bold=True, size=12, color="1F4E78")
            img = XLImage(str(p))
            scale = min(1.0, 760 / img.width)
            img.width, img.height = int(img.width * scale), int(img.height * scale)
            ws.add_image(img, f"A{row + 1}")
            row += int(img.height / 20) + 4

    wb.active = 0
    wb.save(path)
    return path


def write_markdown(res: Dict[str, Any], scenario: Scenario, path: Path, charts: Optional[Dict[str, Path]] = None) -> Path:
    lines = [f"# {scenario.name}：规则策略评估摘要", "", "净收益 = 避免损失 − 机会成本 − 运营成本", "", "## 结论", ""]
    lines += [f"{i}. {t}" for i, t in enumerate(findings(res, scenario), 1)]

    def md(df: pd.DataFrame, cols: List[str], fmts: Dict[str, str]) -> List[str]:
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                    cells.append(format(v, fmts.get(c, ",.0f")))
                else:
                    cells.append("" if pd.isna(v) else str(v))
            out.append("| " + " | ".join(cells) + " |")
        return out

    f = {"命中率": ".2%", "召回率": ".1%", "精确率": ".1%", "提升度": ".2f"}
    if not res["single_rules"].empty:
        lines += ["", "## 单规则", ""] + md(res["single_rules"], ["规则ID", "规则名称", "命中量", "召回率", "精确率", "提升度", "净收益"], f)
    if not res["strategies"].empty:
        lines += ["", "## 策略对比", ""] + md(res["strategies"], ["策略", "命中量", "召回率", "精确率", "避免损失", "机会成本", "运营成本", "净收益"], f)
    if charts:
        lines += ["", "## 图表", ""]
        for name, p in charts.items():
            lines += [f"### {name}", "", f"![{name}](charts/{p.name})", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_report(
    res: Dict[str, Any],
    scenario: Scenario,
    out_dir: str,
    vrep: Optional[ValidationReport] = None,
    with_charts: bool = True,
) -> Dict[str, Path]:
    """写出完整报告，返回各文件路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    charts: Dict[str, Path] = {}
    if with_charts:
        from .charts import render_all

        charts = render_all(res, out / "charts")
    xlsx = write_excel(res, scenario, out / f"{scenario.key}_评估报告.xlsx", vrep, charts)
    md = write_markdown(res, scenario, out / "summary.md", charts)
    return {"xlsx": xlsx, "summary": md, **{f"chart:{k}": v for k, v in charts.items()}}
