import numpy as np
import pandas as pd
import pytest

from riskrule_eval import Evaluator, metrics
from riskrule_eval.expr import ExprError, evaluate, evaluate_mask, referenced_names
from riskrule_eval.schema import _from_dict, expand_values


# ---------------------------------------------------------------- 表达式
@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1, 5, 10, None],
            "b": [0, 1, 1, 0],
            "c": ["x", "y", "z", "x"],
        }
    )


def test_expr_compare_and_logic(df):
    assert evaluate_mask("a >= 5 and b == 1", df).tolist() == [False, True, True, False]
    assert evaluate_mask("a < 5 or c == 'z'", df).tolist() == [True, False, True, False]
    assert evaluate_mask("not (b == 1)", df).tolist() == [True, False, False, True]
    assert evaluate_mask("(a >= 5) & (b == 1)", df).tolist() == [False, True, True, False]


def test_expr_in_and_chain(df):
    assert evaluate_mask("c in ['x', 'z']", df).tolist() == [True, False, True, True]
    assert evaluate_mask("c not in ['x']", df).tolist() == [False, True, True, False]
    assert evaluate_mask("1 < a <= 10", df).tolist() == [False, True, True, False]


def test_expr_missing_value_is_not_hit(df):
    assert evaluate_mask("a > 0", df).tolist() == [True, True, True, False]


def test_expr_arithmetic_params_and_functions(df):
    out = evaluate("maximum(a * rate - 2, 0)", df, {"rate": 0.5})
    assert out.tolist()[:3] == [0.0, 0.5, 3.0]
    assert referenced_names("maximum(a * rate - 2, 0)") == {"a", "rate"}


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os').system('echo hi')",
        "a.__class__",
        "open('x')",
        "[i for i in a]",
        "lambda: 1",
        "a ** 2",
        "unknown_col > 1",
        "a >",
        "",
    ],
)
def test_expr_rejects_unsafe_or_invalid(df, bad):
    with pytest.raises(ExprError):
        evaluate(bad, df)


# ---------------------------------------------------------------- 指标
def test_auc_and_ks_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1])
    risk = np.array([1, 2, 3, 4, 5])
    assert metrics.auc(risk, y) == pytest.approx(1.0)
    assert metrics.ks(risk, y) == pytest.approx(1.0)
    assert metrics.auc(-risk, y) == pytest.approx(0.0)
    assert metrics.auc(np.ones(5), y) == pytest.approx(0.5)


def test_psi_zero_for_same_distribution_and_positive_for_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    edges = metrics.quantile_edges(base, 5)
    same, _ = metrics.psi(base, base, edges)
    shifted, table = metrics.psi(base, base + 1.0, edges)
    assert same == pytest.approx(0.0, abs=1e-12)
    assert shifted > 0.25
    assert table["基准样本数"].sum() == 5000


def test_expand_values():
    assert expand_values({"start": 1, "stop": 2, "step": 0.5}) == [1, 1.5, 2]
    assert expand_values([3, 1]) == [3, 1]


# ---------------------------------------------------------------- 引擎（手算核对）
def _toy_scenario():
    return _from_dict(
        {
            "key": "toy",
            "name": "玩具场景",
            "id_field": "id",
            "time_field": "t",
            "label": {"field": "y"},
            "fields": [
                {"name": "id", "label": "编号", "dtype": "str", "required": True, "role": "id"},
                {"name": "t", "label": "时间", "dtype": "date", "required": True, "role": "time"},
                {"name": "x", "label": "特征", "dtype": "int", "required": True},
                {"name": "s", "label": "评分", "dtype": "float", "required": True, "role": "score"},
                {"name": "loss", "label": "损失", "dtype": "float", "required": True, "role": "value"},
                {"name": "profit", "label": "利润", "dtype": "float", "required": True, "role": "value"},
                {"name": "y", "label": "标签", "dtype": "flag", "required": True, "role": "label"},
            ],
            "value_model": {
                "params": {"c": {"value": 10, "label": "单位成本"}},
                "avoided_loss": "loss",
                "opportunity_cost": "profit",
                "unit_cost": "c",
            },
            "strategy": {
                "rules": [
                    {"id": "A", "name": "x大于等于2", "expr": "x >= 2"},
                    {"id": "B", "name": "x大于等于3", "expr": "x >= 3"},
                    {"id": "C", "name": "评分低", "expr": "s < 50"},
                ],
                "strategies": {"old": {"name": "旧", "rules": ["A"]}, "new": {"name": "新", "rules": ["B", "C"]}},
                "scan": {"field": "s", "op": "<", "values": [30, 50, 70, 90]},
                "matrix": {"x": {"field": "x", "op": ">=", "values": [2, 3]}, "y": {"field": "s", "op": "<", "values": [50, 70]}},
                "monitor": {"psi_field": "s", "bins": [-1, 50, 100]},
            },
        }
    )


def _toy_data():
    #      id  x   s   y  loss profit
    rows = [
        ("1", 0, 90, 0, 0, 100),
        ("2", 1, 80, 0, 0, 100),
        ("3", 2, 60, 0, 0, 100),  # 只被 A 命中的好客户
        ("4", 2, 40, 1, 1000, 0),  # A、C 命中的坏客户
        ("5", 3, 20, 1, 2000, 0),  # A、B、C 命中的坏客户
        ("6", 3, 85, 0, 0, 300),  # A、B 命中的好客户
        ("7", 0, 45, 0, 0, 200),  # 只被 C 命中的好客户
        ("8", 1, 95, 1, 500, 0),  # 漏掉的坏客户
    ]
    d = pd.DataFrame(rows, columns=["id", "x", "s", "y", "loss", "profit"])
    d["t"] = pd.to_datetime(["2026-01-05"] * 4 + ["2026-02-05"] * 4)
    return d


@pytest.fixture
def toy():
    return Evaluator(_toy_scenario(), _toy_data())


def test_single_rule_numbers(toy):
    r = toy.single_rules().set_index("规则ID")
    a = r.loc["A"]
    assert (a["命中量"], a["命中正样本(TP)"], a["命中负样本(FP)"]) == (4, 2, 2)
    assert a["召回率"] == pytest.approx(2 / 3)
    assert a["精确率"] == pytest.approx(0.5)
    assert a["提升度"] == pytest.approx(0.5 / (3 / 8))
    assert a["避免损失"] == 3000 and a["机会成本"] == 400 and a["运营成本"] == 40
    assert a["净收益"] == 3000 - 400 - 40


def test_strategy_compare_and_swap_reconcile(toy):
    st = toy.strategies_table()
    old, new, diff = st.iloc[0], st.iloc[1], st.iloc[2]
    assert old["净收益"] == 2560
    # 新策略命中 4、5、6、7：避免 3000，机会成本 500，运营成本 40
    assert new["净收益"] == 3000 - 500 - 40
    assert diff["净收益"] == new["净收益"] - old["净收益"]
    sw = toy.swap()
    assert sw.iloc[0]["数量"] == 1 and sw.iloc[1]["数量"] == 1
    assert sw.attrs["total"] == pytest.approx(diff["净收益"])
    assert sw.attrs["check_diff"] == pytest.approx(0.0)


def test_marginal_leave_one_out(toy):
    m = toy.marginal("new").set_index("规则ID")
    # 去掉 C 后只剩 B（命中 5、6）：2000 − 300 − 20 = 1680；全量 2460
    assert m.loc["C", "剔除后净收益"] == 1680
    assert m.loc["C", "边际贡献"] == 2460 - 1680
    # 去掉 B 后只剩 C（命中 4、5、7）：3000 − 200 − 30 = 2770，B 的边际贡献为负
    assert m.loc["B", "边际贡献"] == 2460 - 2770
    assert m.loc["B", "独占命中量"] == 1 and m.loc["B", "独占正样本"] == 0


def test_scan_matrix_and_sensitivity(toy):
    sc = toy.scan()
    assert sc.attrs["best"] == 50  # s<50 命中 4、5、7
    assert sc.attrs["best_net"] == 2770
    assert sc.loc[sc["阈值"] == 30, "净收益"].iloc[0] == 2000 - 10
    mx = toy.matrix()
    assert mx.shape == (2, 3)
    assert mx.iloc[0, 2] == 2460  # x>=3 或 s<50，与新策略相同
    se = toy.sensitivity()
    assert se["grid"].iloc[1, 2] == pytest.approx(2460 - 2560)
    zero_cost = se["cost"].iloc[0]
    assert zero_cost["净收益差异"] == pytest.approx((3000 - 500) - (3000 - 400))


def test_discrimination_and_monitor(toy):
    d = toy.discrimination()
    assert 0.5 < d["auc"] <= 1 and 0 < d["ks"] <= 1
    mo = toy.monitor()
    assert mo["monthly"]["样本量"].tolist() == [4, 4]
    assert len(mo["psi"]) == 1


def test_params_override_and_bad_rule_is_skipped():
    sc = _toy_scenario()
    strat = sc.default_strategy()
    strat["rules"].append({"id": "Z", "name": "引用不存在的列", "expr": "nope > 1"})
    strat["strategies"]["new"]["rules"].append("Z")
    ev = Evaluator(sc, _toy_data(), strategy=strat, params={"c": 100})
    assert "Z" not in ev.rules
    assert any("Z" in w for w in ev.warnings)
    assert ev.stats(ev.masks["A"])["op_cost"] == 400
