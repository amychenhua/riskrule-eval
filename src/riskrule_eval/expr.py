"""规则表达式的安全求值器。

规则和取值模型都写成字符串表达式，例如 ``"query_cnt_3m >= 5 and dti > 0.6"``。
这里不使用 ``eval`` 或 ``DataFrame.eval``，而是把表达式解析成语法树后只放行白名单里的
节点类型，因此策略文件里写不进任意代码。

支持：
- 比较：``== != < <= > >=``，``in`` / ``not in``（右侧为常量列表）
- 逻辑：``and`` ``or`` ``not``（也接受 ``& | ~``）
- 算术：``+ - * /``
- 函数：``abs`` ``maximum`` ``minimum`` ``isnull`` ``notnull`` ``log``
- 名称：先按数据列名解析，再按参数名解析
"""

from __future__ import annotations

import ast
from typing import Any, Mapping, Set

import numpy as np
import pandas as pd


class ExprError(ValueError):
    """表达式不合法，或引用了不存在的字段。"""


_FUNCS = {
    "abs": np.abs,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "isnull": pd.isna,
    "notnull": pd.notna,
    "log": np.log,
}

_CMP = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}

_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def _parse(expr: str) -> ast.AST:
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError("表达式为空")
    try:
        return ast.parse(expr.strip(), mode="eval").body
    except SyntaxError as e:
        raise ExprError(f"表达式语法错误：{expr!r}（{e.msg}）") from None


def referenced_names(expr: str) -> Set[str]:
    """表达式里引用到的名称（字段名或参数名），不含函数名。"""
    tree = _parse(expr)
    funcs = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - funcs


def _as_bool(v: Any) -> Any:
    if isinstance(v, pd.Series):
        return v.fillna(False).astype(bool)
    if isinstance(v, np.ndarray):
        return v.astype(bool)
    return bool(v)


def _ev(node: ast.AST, df: pd.DataFrame, params: Mapping[str, Any], src: str) -> Any:
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise ExprError(f"不支持的常量：{node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in df.columns:
            return df[node.id]
        if node.id in params:
            return params[node.id]
        raise ExprError(f"表达式 {src!r} 引用了不存在的字段或参数：{node.id}")
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [_ev(e, df, params, src) for e in node.elts]
        if any(isinstance(v, (pd.Series, np.ndarray)) for v in vals):
            raise ExprError("列表里只能写常量")
        return vals
    if isinstance(node, ast.BoolOp):
        vals = [_as_bool(_ev(v, df, params, src)) for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = (out & v) if isinstance(node.op, ast.And) else (out | v)
        return out
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, df, params, src)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            b = _as_bool(v)
            return ~b if isinstance(b, (pd.Series, np.ndarray)) else (not b)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ExprError("不支持的一元运算")
    if isinstance(node, ast.BinOp):
        a = _ev(node.left, df, params, src)
        b = _ev(node.right, df, params, src)
        if isinstance(node.op, ast.BitAnd):
            return _as_bool(a) & _as_bool(b)
        if isinstance(node.op, ast.BitOr):
            return _as_bool(a) | _as_bool(b)
        fn = _BIN.get(type(node.op))
        if fn is None:
            raise ExprError(f"不支持的运算符：{type(node.op).__name__}")
        return fn(a, b)
    if isinstance(node, ast.Compare):
        left = _ev(node.left, df, params, src)
        out = None
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, df, params, src)
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(right, list):
                    raise ExprError("in 的右侧必须是常量列表，例如 channel in ['app', 'h5']")
                if isinstance(left, pd.Series):
                    res = left.isin(right)
                else:
                    res = left in right
                if isinstance(op, ast.NotIn):
                    res = ~res if isinstance(res, pd.Series) else (not res)
            else:
                fn = _CMP.get(type(op))
                if fn is None:
                    raise ExprError(f"不支持的比较：{type(op).__name__}")
                res = fn(left, right)
            out = res if out is None else (_as_bool(out) & _as_bool(res))
            left = right
        return out
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            name = getattr(node.func, "id", "?")
            raise ExprError(f"不支持的函数：{name}（可用：{', '.join(sorted(_FUNCS))}）")
        if node.keywords:
            raise ExprError("函数不支持关键字参数")
        args = [_ev(a, df, params, src) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise ExprError(f"表达式 {src!r} 含有不支持的语法：{type(node).__name__}")


def evaluate(expr: str, df: pd.DataFrame, params: Mapping[str, Any] | None = None) -> Any:
    """对数据求值，返回 Series 或标量。"""
    return _ev(_parse(expr), df, params or {}, expr)


def evaluate_mask(expr: str, df: pd.DataFrame, params: Mapping[str, Any] | None = None) -> np.ndarray:
    """求值并转成布尔数组（缺失值按未命中处理）。"""
    v = evaluate(expr, df, params)
    if isinstance(v, pd.Series):
        return v.fillna(False).astype(bool).to_numpy()
    if isinstance(v, np.ndarray):
        return v.astype(bool)
    return np.full(len(df), bool(v))


def evaluate_number(expr: str, df: pd.DataFrame, params: Mapping[str, Any] | None = None) -> np.ndarray:
    """求值并转成浮点数组（缺失值按 0 处理，标量自动扩展到每一行）。"""
    v = evaluate(expr, df, params)
    if isinstance(v, pd.Series):
        return pd.to_numeric(v, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if isinstance(v, np.ndarray):
        return np.nan_to_num(v.astype(float))
    return np.full(len(df), float(v))
