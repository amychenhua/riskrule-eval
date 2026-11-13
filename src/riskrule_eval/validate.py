"""上传数据校验：按场景的字段定义检查并转换类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from .schema import Scenario

MIN_POSITIVES = 100


@dataclass
class Issue:
    level: str  # error / warn / info
    field: str
    message: str
    count: int = 0

    def as_row(self) -> dict:
        cn = {"error": "错误", "warn": "警告", "info": "提示"}[self.level]
        return {"级别": cn, "字段": self.field, "问题": self.message, "涉及行数": self.count}


@dataclass
class ValidationReport:
    issues: List[Issue] = field(default_factory=list)
    n_rows: int = 0
    n_positive: int = 0

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "error"]

    def add(self, level: str, fld: str, message: str, count: int = 0) -> None:
        self.issues.append(Issue(level, fld, message, int(count)))

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame([{"级别": "通过", "字段": "", "问题": "未发现问题", "涉及行数": 0}])
        order = {"error": 0, "warn": 1, "info": 2}
        return pd.DataFrame([i.as_row() for i in sorted(self.issues, key=lambda i: order[i.level])])

    def summary(self) -> str:
        e = sum(i.level == "error" for i in self.issues)
        w = sum(i.level == "warn" for i in self.issues)
        return f"{self.n_rows} 行，正样本 {self.n_positive} 个；错误 {e} 项，警告 {w} 项"


def validate(df: pd.DataFrame, scenario: Scenario) -> Tuple[pd.DataFrame, ValidationReport]:
    """校验并返回类型转换后的数据。有 error 级问题时不应继续评估。"""
    rep = ValidationReport(n_rows=len(df))
    out = df.copy()

    if len(out) == 0:
        rep.add("error", "", "数据为空")
        return out, rep

    for name in scenario.required_names:
        if name not in out.columns:
            f = scenario.get_field(name)
            rep.add("error", name, f"缺少必填字段「{f.label}」({name})")
    extra = [c for c in out.columns if c not in scenario.field_names]
    if extra:
        rep.add("info", "", f"有 {len(extra)} 个模板之外的列，会原样保留，可在自定义规则里引用：{', '.join(extra[:8])}" + ("…" if len(extra) > 8 else ""))

    for f in scenario.fields:
        if f.name not in out.columns:
            if not f.required:
                rep.add("info", f.name, f"选填字段「{f.label}」未提供，引用它的规则会报错")
            continue
        col = out[f.name]
        blank = col.isna() | (col.astype(str).str.strip() == "")
        n_blank = int(blank.sum())
        if n_blank and f.required:
            rep.add("error", f.name, f"必填字段「{f.label}」有空值", n_blank)
        elif n_blank:
            rep.add("info", f.name, f"选填字段「{f.label}」有空值，规则比较时按未命中处理", n_blank)

        if f.dtype in ("int", "float", "flag"):
            num = pd.to_numeric(col.where(~blank), errors="coerce")
            bad = int((num.isna() & ~blank).sum())
            if bad:
                rep.add("error", f.name, f"「{f.label}」应为数字，有无法解析的值", bad)
            if f.dtype == "flag":
                wrong = int((~num.isin([0, 1]) & num.notna()).sum())
                if wrong:
                    rep.add("error", f.name, f"「{f.label}」只能填 0 或 1", wrong)
            if f.dtype == "int":
                frac = int(((num % 1) != 0).sum())
                if frac:
                    rep.add("warn", f.name, f"「{f.label}」应为整数，存在小数", frac)
            if f.min is not None:
                lo = int((num < f.min).sum())
                if lo:
                    rep.add("error", f.name, f"「{f.label}」小于下限 {f.min:g}", lo)
            if f.max is not None:
                hi = int((num > f.max).sum())
                if hi:
                    rep.add("error", f.name, f"「{f.label}」大于上限 {f.max:g}", hi)
            out[f.name] = num
        elif f.dtype in ("date", "datetime"):
            dt = pd.to_datetime(col.where(~blank), errors="coerce")
            bad = int((dt.isna() & ~blank).sum())
            if bad:
                rep.add("error", f.name, f"「{f.label}」有无法解析的日期，建议格式 2026-01-31 或 2026-01-31 14:30:00", bad)
            out[f.name] = dt
        elif f.dtype == "enum":
            s = col.where(~blank).astype("string").str.strip()
            wrong = int((~s.isin(f.choices) & s.notna()).sum())
            if wrong:
                rep.add("error", f.name, f"「{f.label}」取值不在允许范围内：{', '.join(f.choices)}", wrong)
            out[f.name] = s.astype(object).where(s.notna(), None)
        else:
            s = col.where(~blank).astype("string").str.strip()
            out[f.name] = s.astype(object).where(s.notna(), None)

    if scenario.id_field in out.columns:
        dup = int(out[scenario.id_field].duplicated(keep=False).sum())
        if dup:
            rep.add("warn", scenario.id_field, "主键重复，同一条记录会被重复计入", dup)

    lab = scenario.label_field
    if lab in out.columns:
        y = pd.to_numeric(out[lab], errors="coerce")
        pos = int((y == 1).sum())
        rep.n_positive = pos
        if pos == 0 or pos == int(y.notna().sum()):
            rep.add("error", lab, "标签只有一个类别，无法评估")
        elif pos < MIN_POSITIVES:
            rep.add("warn", lab, f"正样本只有 {pos} 个，少于建议的 {MIN_POSITIVES} 个，提升度和净收益的结论可能随样本波动而翻转")

    return out, rep
