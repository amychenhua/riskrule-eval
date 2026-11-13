"""数据上传模板生成：每个场景一份 Excel 模板、一份 CSV 表头和一份字段说明。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .dataio import DATA_SHEET
from .schema import Scenario
from .synth import generate

FONT = "Arial"
_REQ = PatternFill("solid", fgColor="1F4E78")
_OPT = PatternFill("solid", fgColor="D9E1F2")
_LAB = PatternFill("solid", fgColor="C55A11")
_HEAD = PatternFill("solid", fgColor="1F4E78")
_THIN = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_WRAP = Alignment(wrap_text=True, vertical="top")

ROLE_CN = {
    "id": "标识",
    "time": "时间",
    "label": "标签",
    "amount": "金额",
    "feature": "规则字段",
    "score": "评分",
    "value": "收益成本",
}


def _range_text(f) -> str:
    if f.dtype == "flag":
        return "0 或 1"
    if f.dtype == "enum":
        return " / ".join(f.choices)
    parts = []
    if f.min is not None:
        parts.append(f"≥ {f.min:g}")
    if f.max is not None:
        parts.append(f"≤ {f.max:g}")
    return "，".join(parts)


def field_table(scenario: Scenario) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(scenario.fields, 1):
        rows.append(
            {
                "序号": i,
                "字段名": f.name,
                "中文名": f.label,
                "类型": f.dtype_cn,
                "是否必填": "必填" if f.required else "选填",
                "用途": ROLE_CN[f.role],
                "取值范围": _range_text(f),
                "说明": f.desc,
                "示例": f.example,
            }
        )
    return pd.DataFrame(rows)


def _write_table(ws, df: pd.DataFrame, start_row: int = 1, widths: List[int] | None = None) -> None:
    for j, c in enumerate(df.columns, 1):
        cell = ws.cell(row=start_row, column=j, value=c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = _HEAD
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _BORDER
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, pd.Timestamp):
                v = v.strftime("%Y-%m-%d %H:%M:%S") if (v.hour or v.minute or v.second) else v.strftime("%Y-%m-%d")
            elif hasattr(v, "item"):
                v = v.item()
            cell = ws.cell(row=i, column=j, value=v)
            cell.font = Font(name=FONT)
            cell.border = _BORDER
            cell.alignment = _WRAP
    if widths:
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w


def build_workbook(scenario: Scenario, header: str = "name", sample_rows: int = 5) -> Workbook:
    """生成某个场景的上传模板工作簿。

    header="name" 时表头用字段名，header="cn" 时用中文名；上传时两种都能识别。
    """
    if header not in ("name", "cn"):
        raise ValueError("header 只能是 name 或 cn")
    wb = Workbook()

    # ---------- 填写须知 ----------
    ws = wb.active
    ws.title = "填写须知"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 96
    ws["B2"] = f"{scenario.name}｜数据上传模板"
    ws["B2"].font = Font(name=FONT, bold=True, size=16, color="1F4E78")
    ws["B3"] = f"场景代码：{scenario.key}　　由 riskrule-eval 生成"
    ws["B3"].font = Font(name=FONT, italic=True, color="808080")

    r = 5
    info = [
        ("数据粒度", scenario.unit),
        ("规则命中后的动作", scenario.action),
        ("标签字段", f"{scenario.get_field(scenario.label_field).label}（{scenario.label_field}）"),
        ("正样本(=1)的定义", scenario.label_positive),
        ("标签成熟所需时间", scenario.label_window),
        ("收益成本口径", scenario.value_model.get("description", "")),
    ]
    for k, v in info:
        ws.cell(row=r, column=2, value=k).font = Font(name=FONT, bold=True)
        c = ws.cell(row=r, column=3, value=v)
        c.font = Font(name=FONT)
        c.alignment = _WRAP
        r += 1

    r += 1
    ws.cell(row=r, column=2, value="怎么填").font = Font(name=FONT, bold=True, size=12, color="1F4E78")
    r += 1
    steps = [
        f"1. 在「{DATA_SHEET}」工作表从第 2 行开始填写，一行一条记录，不要改动第 1 行表头，不要留空行。",
        "2. 表头颜色：深蓝＝必填，浅蓝＝选填，橙色＝标签字段（必填）。把鼠标停在表头上可以看到字段说明。",
        "3. 「示例数据」工作表给出 5 行模拟数据作为格式参考，上传时不会被读取。「字段说明」列出每个字段的类型和取值范围。",
        "4. 0/1 字段和枚举字段带下拉校验；日期写成 2026-01-31，日期时间写成 2026-01-31 14:30:00。",
        "5. 选填字段没有就留空或整列删除，引用到它的规则会被自动跳过并在报告里提示。可以追加模板之外的列，在自定义规则里按列名引用。",
        "6. 也可以直接上传 CSV（UTF-8 或 GBK 编码），表头用字段名或中文名都可以。",
        f"7. 填好后先校验再评估：riskrule validate -s {scenario.key} -d 你的文件.xlsx",
    ]
    for s in steps:
        c = ws.cell(row=r, column=2, value=s)
        c.font = Font(name=FONT)
        c.alignment = _WRAP
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=3)
        ws.row_dimensions[r].height = 32
        r += 1

    if scenario.notes:
        r += 1
        ws.cell(row=r, column=2, value="本场景要特别注意").font = Font(name=FONT, bold=True, size=12, color="1F4E78")
        r += 1
        for n in scenario.notes:
            c = ws.cell(row=r, column=2, value="· " + n)
            c.font = Font(name=FONT)
            c.alignment = _WRAP
            ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=3)
            ws.row_dimensions[r].height = 34
            r += 1

    r += 1
    ws.cell(row=r, column=2, value="收益成本参数（默认值，可在策略文件或命令行里覆盖）").font = Font(name=FONT, bold=True, size=12, color="1F4E78")
    r += 1
    vm = scenario.value_model
    for k, v in (vm.get("params") or {}).items():
        ws.cell(row=r, column=2, value=k).font = Font(name=FONT)
        ws.cell(row=r, column=3, value=f"{v.get('label', k)}：{v['value']}").font = Font(name=FONT)
        r += 1
    for k, lab in (("avoided_loss", "避免损失（命中且为正样本）"), ("opportunity_cost", "机会成本（命中且为负样本）"), ("unit_cost", "运营成本（每条命中）")):
        ws.cell(row=r, column=2, value=lab).font = Font(name=FONT)
        ws.cell(row=r, column=3, value=f"= {vm[k]}").font = Font(name="Consolas")
        r += 1

    # ---------- 数据 ----------
    wd = wb.create_sheet(DATA_SHEET)
    wd.freeze_panes = "A2"
    wd.row_dimensions[1].height = 32
    for j, f in enumerate(scenario.fields, 1):
        col = get_column_letter(j)
        cell = wd.cell(row=1, column=j, value=f.name if header == "name" else f.label)
        is_label = f.name == scenario.label_field
        cell.fill = _LAB if is_label else (_REQ if f.required else _OPT)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF" if (f.required or is_label) else "1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _BORDER
        other = f.label if header == "name" else f.name
        rng = _range_text(f)
        note = f"{other}\n{'必填' if f.required else '选填'}｜{f.dtype_cn}" + (f"｜{rng}" if rng else "") + f"\n{f.desc}\n示例：{f.example}"
        cm = Comment(note, "riskrule-eval")
        cm.width, cm.height = 280, 110
        cell.comment = cm
        wd.column_dimensions[col].width = max(14, min(30, len(str(cell.value)) * (2.1 if header == "cn" else 1.25) + 3))
        if f.dtype in ("flag", "enum"):
            choices = ["0", "1"] if f.dtype == "flag" else list(f.choices)
            dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True)
            dv.error = f"{f.label} 只能填：{' / '.join(choices)}"
            dv.errorTitle = "取值不合法"
            wd.add_data_validation(dv)
            dv.add(f"{col}2:{col}1048576")
        if f.dtype in ("str", "enum", "date", "datetime"):
            for row in range(2, 202):
                wd.cell(row=row, column=j).number_format = "@"

    # ---------- 示例数据 ----------
    we = wb.create_sheet("示例数据")
    sample = generate(scenario, n=400, seed=7)
    pos = sample[sample[scenario.label_field] == 1].head(max(1, sample_rows // 3))
    neg = sample[sample[scenario.label_field] == 0].head(sample_rows - len(pos))
    ex = pd.concat([neg, pos]).head(sample_rows)
    if header == "cn":
        ex = ex.rename(columns={f.name: f.label for f in scenario.fields})
    _write_table(we, ex, widths=[24] * len(ex.columns))
    we.freeze_panes = "A2"
    note_row = len(ex) + 3
    we.cell(row=note_row, column=1, value="以上为随机生成的模拟数据，仅作格式参考。").font = Font(name=FONT, italic=True, color="808080")

    # ---------- 字段说明 ----------
    wf = wb.create_sheet("字段说明")
    _write_table(wf, field_table(scenario), widths=[6, 28, 30, 10, 10, 12, 30, 62, 24])
    wf.freeze_panes = "A2"

    wb.active = 0
    return wb


def write_templates(scenario: Scenario, out_dir: str, header: str = "name") -> Dict[str, Path]:
    """写出一个场景的 xlsx 模板和 csv 表头，返回文件路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario.order:02d}_{scenario.key}_{scenario.name}_上传模板"
    xlsx = out / f"{stem}.xlsx"
    build_workbook(scenario, header=header).save(xlsx)
    csv = out / f"{stem}.csv"
    cols = [f.name if header == "name" else f.label for f in scenario.fields]
    csv.write_text(",".join(cols) + "\n", encoding="utf-8-sig")
    return {"xlsx": xlsx, "csv": csv}


def data_dictionary_md(scenarios: List[Scenario]) -> str:
    """全部场景的字段说明，Markdown 格式。"""
    lines = ["# 数据上传模板字段说明", "", "本文件由 `riskrule template --all` 自动生成，请勿手工修改。", ""]
    for sc in scenarios:
        lines += [
            f"## {sc.order}. {sc.name}（`{sc.key}`）",
            "",
            f"- 数据粒度：{sc.unit}",
            f"- 规则命中后的动作：{sc.action}",
            f"- 标签：`{sc.label_field}` = 1 表示{sc.label_positive}",
            f"- 标签成熟所需时间：{sc.label_window}",
            f"- 收益成本口径：{sc.value_model.get('description', '')}",
            "",
            "| 序号 | 字段名 | 中文名 | 类型 | 必填 | 用途 | 取值范围 | 说明 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in field_table(sc).iterrows():
            lines.append(f"| {r['序号']} | `{r['字段名']}` | {r['中文名']} | {r['类型']} | {r['是否必填']} | {r['用途']} | {r['取值范围']} | {r['说明']} |")
        vm = sc.value_model
        lines += ["", "取值模型：", "", "```", f"避免损失 = {vm['avoided_loss']}", f"机会成本 = {vm['opportunity_cost']}", f"运营成本 = {vm['unit_cost']}", "```", "", "默认参数：", ""]
        for k, v in (vm.get("params") or {}).items():
            lines.append(f"- `{k}` = {v['value']}（{v.get('label', k)}）")
        if sc.notes:
            lines += ["", "注意事项：", ""] + [f"- {n}" for n in sc.notes]
        lines.append("")
    return "\n".join(lines)
