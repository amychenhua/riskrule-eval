"""riskrule-eval：银行风控规则策略的收益与风险量化评估工具。

覆盖六个风险管理场景：反洗钱、反电诈、贷前反欺诈、贷前审批、贷中调额、贷后预警。
"""

from .schema import Scenario, Field, load_scenario, list_scenarios
from .engine import Evaluator
from .dataio import read_data
from .validate import validate

__version__ = "0.1.0"

__all__ = [
    "Scenario",
    "Field",
    "load_scenario",
    "list_scenarios",
    "Evaluator",
    "read_data",
    "validate",
    "__version__",
]
