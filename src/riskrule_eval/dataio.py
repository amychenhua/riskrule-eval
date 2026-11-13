"""读取上传数据：支持 CSV 和 Excel，表头用字段名或中文名都可以。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import Scenario

DATA_SHEET = "数据"


def read_data(path: str, scenario: Scenario) -> pd.DataFrame:
    """读入数据并把中文表头统一成字段名。类型转换留给校验步骤。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        xl = pd.ExcelFile(p)
        sheet = DATA_SHEET if DATA_SHEET in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet, dtype=object)
    elif suffix in (".csv", ".txt"):
        df = None
        for enc in ("utf-8-sig", "gb18030"):
            try:
                df = pd.read_csv(p, dtype=object, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"无法识别 {path} 的编码，请另存为 UTF-8 或 GBK")
    else:
        raise ValueError(f"不支持的文件类型 {suffix}，请使用 .csv 或 .xlsx")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=scenario.label_to_name())
    df = df.dropna(how="all").reset_index(drop=True)
    return df
