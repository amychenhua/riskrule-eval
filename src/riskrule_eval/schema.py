"""场景定义：字段、标签、取值模型、默认策略。

每个场景是 ``scenarios/`` 目录下的一个 YAML 文件。新增场景只需要加一个 YAML，
不用改代码。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DTYPES = ("str", "int", "float", "flag", "enum", "date", "datetime")
ROLES = ("id", "time", "label", "amount", "feature", "score", "value")


@dataclass
class Field:
    name: str
    label: str
    dtype: str
    required: bool = False
    role: str = "feature"
    desc: str = ""
    example: Any = None
    choices: Optional[List[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"字段 {self.name} 的类型 {self.dtype!r} 不合法，可选：{DTYPES}")
        if self.role not in ROLES:
            raise ValueError(f"字段 {self.name} 的角色 {self.role!r} 不合法，可选：{ROLES}")
        if self.dtype == "enum" and not self.choices:
            raise ValueError(f"枚举字段 {self.name} 必须给出 choices")

    @property
    def dtype_cn(self) -> str:
        return {
            "str": "文本",
            "int": "整数",
            "float": "数值",
            "flag": "0/1",
            "enum": "枚举",
            "date": "日期",
            "datetime": "日期时间",
        }[self.dtype]


@dataclass
class Scenario:
    key: str
    name: str
    name_en: str
    unit: str
    action: str
    id_field: str
    time_field: str
    label_field: str
    label_positive: str
    label_window: str
    fields: List[Field]
    value_model: Dict[str, Any]
    strategy: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    order: int = 99

    # ---- 字段访问 ----
    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    @property
    def required_names(self) -> List[str]:
        return [f.name for f in self.fields if f.required]

    def get_field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def label_to_name(self) -> Dict[str, str]:
        """中文表头到字段名的映射，上传时两种表头都接受。"""
        return {f.label: f.name for f in self.fields}

    # ---- 参数 ----
    def default_params(self) -> Dict[str, float]:
        return {k: v["value"] for k, v in (self.value_model.get("params") or {}).items()}

    def param_labels(self) -> Dict[str, str]:
        return {k: v.get("label", k) for k, v in (self.value_model.get("params") or {}).items()}

    def default_strategy(self) -> Dict[str, Any]:
        return copy.deepcopy(self.strategy)


def _from_dict(d: Dict[str, Any]) -> Scenario:
    fields = [Field(**f) for f in d["fields"]]
    names = [f.name for f in fields]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"场景 {d['key']} 的字段名重复：{sorted(dup)}")
    lab = d["label"]
    sc = Scenario(
        key=d["key"],
        name=d["name"],
        name_en=d.get("name_en", d["key"]),
        unit=d.get("unit", ""),
        action=d.get("action", ""),
        id_field=d["id_field"],
        time_field=d["time_field"],
        label_field=lab["field"],
        label_positive=lab.get("positive", ""),
        label_window=lab.get("window", ""),
        fields=fields,
        value_model=d["value_model"],
        strategy=d.get("strategy") or {},
        notes=d.get("notes") or [],
        order=d.get("order", 99),
    )
    for must in (sc.id_field, sc.time_field, sc.label_field):
        if must not in names:
            raise ValueError(f"场景 {sc.key} 的关键字段 {must} 未在 fields 中定义")
    return sc


def _builtin_files() -> Dict[str, Any]:
    root = resources.files(__package__) / "scenarios"
    return {p.name[:-5]: p for p in root.iterdir() if p.name.endswith(".yaml")}


def list_scenarios() -> List[Scenario]:
    """内置的全部场景，按 order 排序。"""
    out = [load_scenario(k) for k in _builtin_files()]
    return sorted(out, key=lambda s: s.order)


def load_scenario(key_or_path: str) -> Scenario:
    """按场景代码加载内置场景，或按路径加载自定义场景 YAML。"""
    files = _builtin_files()
    if key_or_path in files:
        text = files[key_or_path].read_text(encoding="utf-8")
    else:
        p = Path(key_or_path)
        if not p.exists():
            raise KeyError(f"未知场景 {key_or_path!r}。内置场景：{', '.join(sorted(files))}；也可以传入自定义场景 YAML 的路径。")
        text = p.read_text(encoding="utf-8")
    return _from_dict(yaml.safe_load(text))


def load_strategy(path: str) -> Dict[str, Any]:
    """读取策略文件。文件可以是完整的 ``strategy`` 节，也可以外面再包一层 ``strategy:``。"""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"策略文件 {path} 内容不是字典")
    return d.get("strategy", d)


def expand_values(spec: Any) -> List[float]:
    """把阈值配置展开成列表：既可以是列表，也可以是 {start, stop, step}（含 stop）。"""
    if isinstance(spec, dict):
        start, stop, step = spec["start"], spec["stop"], spec["step"]
        if step <= 0:
            raise ValueError("step 必须大于 0")
        out, v = [], start
        while v <= stop + 1e-9:
            out.append(round(v, 10))
            v += step
        return out
    return list(spec)
