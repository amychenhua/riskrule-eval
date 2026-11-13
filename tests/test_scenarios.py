import pandas as pd
import pytest

from riskrule_eval import Evaluator, list_scenarios, load_scenario, read_data, validate
from riskrule_eval.cli import main
from riskrule_eval.expr import referenced_names
from riskrule_eval.report import findings, write_report
from riskrule_eval.synth import generate
from riskrule_eval.templates import build_workbook, data_dictionary_md, write_templates

KEYS = ["aml", "anti_telecom_fraud", "pre_loan_fraud", "pre_loan_approval", "credit_limit", "post_loan_warning"]


def test_six_builtin_scenarios_in_order():
    assert [s.key for s in list_scenarios()] == KEYS


@pytest.mark.parametrize("key", KEYS)
def test_scenario_definition_is_consistent(key):
    sc = load_scenario(key)
    names, params = set(sc.field_names), set(sc.default_params())
    assert sc.get_field(sc.label_field).dtype == "flag" and sc.get_field(sc.label_field).required
    assert sc.get_field(sc.time_field).dtype in ("date", "datetime")
    assert len(set(f.label for f in sc.fields)) == len(sc.fields), "中文名不能重复"
    vm = sc.value_model
    for k in ("avoided_loss", "opportunity_cost", "unit_cost"):
        assert referenced_names(vm[k]) <= names | params, f"取值模型 {k} 引用了未定义的名称"
    ids = [r["id"] for r in sc.strategy["rules"]]
    assert len(ids) == len(set(ids))
    for r in sc.strategy["rules"]:
        assert referenced_names(r["expr"]) <= names | params, f"规则 {r['id']} 引用了未定义的字段"
    for s in sc.strategy["strategies"].values():
        assert set(s["rules"]) <= set(ids)
    assert sc.strategy["scan"]["field"] in names
    assert sc.strategy["monitor"]["psi_field"] in names


@pytest.mark.parametrize("key", KEYS)
def test_template_has_scenario_specific_columns(key, tmp_path):
    sc = load_scenario(key)
    paths = write_templates(sc, tmp_path)
    wb = build_workbook(sc)
    assert wb.sheetnames == ["填写须知", "数据", "示例数据", "字段说明"]
    header = [c.value for c in wb["数据"][1]]
    assert header == sc.field_names
    assert paths["csv"].read_text(encoding="utf-8-sig").strip().split(",") == sc.field_names
    cn = [c.value for c in build_workbook(sc, header="cn")["数据"][1]]
    assert cn == [f.label for f in sc.fields]


def test_templates_differ_between_scenarios():
    cols = {k: tuple(load_scenario(k).field_names) for k in KEYS}
    assert len(set(cols.values())) == len(KEYS)
    assert "字段说明" in data_dictionary_md(list_scenarios())


@pytest.mark.parametrize("key", KEYS)
def test_end_to_end_with_chinese_header_upload(key, tmp_path):
    sc = load_scenario(key)
    df = generate(sc, n=3000, seed=1)
    path = tmp_path / "upload.csv"
    df.rename(columns={f.name: f.label for f in sc.fields}).to_csv(path, index=False, encoding="utf-8-sig")
    clean, rep = validate(read_data(str(path), sc), sc)
    assert rep.ok, rep.to_frame()
    ev = Evaluator(sc, clean)
    res = ev.run_all()
    assert not res["warnings"]
    assert len(res["single_rules"]) == len(sc.strategy["rules"])
    st = res["strategies"]
    assert st.iloc[2]["净收益"] == pytest.approx(st.iloc[1]["净收益"] - st.iloc[0]["净收益"])
    assert res["swap"].attrs["check_diff"] == pytest.approx(0, abs=1e-6)
    full = res["marginal"].attrs["full_net"]
    assert full == pytest.approx(st.iloc[1]["净收益"])
    assert not res["scan"].empty and not res["matrix"].empty
    assert 0.5 < res["discrimination"]["auc"] < 1
    out = write_report(res, sc, tmp_path / "rep", vrep=rep, with_charts=(key == "pre_loan_approval"))
    assert out["xlsx"].exists() and out["summary"].exists()
    assert findings(res, sc)


def test_xlsx_upload_reads_data_sheet(tmp_path):
    sc = load_scenario("pre_loan_approval")
    wb = build_workbook(sc)
    ws = wb["数据"]
    df = generate(sc, n=50, seed=3)
    for row in df.itertuples(index=False):
        ws.append([v.strftime("%Y-%m-%d %H:%M:%S") if isinstance(v, pd.Timestamp) else (v.item() if hasattr(v, "item") else v) for v in row])
    p = tmp_path / "filled.xlsx"
    wb.save(p)
    clean, rep = validate(read_data(str(p), sc), sc)
    assert len(clean) == 50
    assert not rep.errors


def test_validate_catches_problems():
    sc = load_scenario("pre_loan_approval")
    df = generate(sc, n=300, seed=2).astype(object)
    df.loc[0, "label_bad"] = 2
    df.loc[1, "dti"] = "abc"
    df.loc[2, "apply_time"] = "不是日期"
    df.loc[3, "query_cnt_3m"] = -1
    df.loc[4, "app_id"] = df.loc[5, "app_id"]
    df = df.drop(columns=["loss_amount"])
    _, rep = validate(df, sc)
    msgs = " ".join(i.message for i in rep.issues)
    assert not rep.ok
    for kw in ("缺少必填字段", "只能填 0 或 1", "应为数字", "无法解析的日期", "小于下限", "主键重复", "正样本只有"):
        assert kw in msgs, kw


def test_optional_field_missing_skips_rule_with_warning():
    sc = load_scenario("pre_loan_approval")
    df = generate(sc, n=2000, seed=4).drop(columns=["large_abnormal_txn", "credit_score"])
    clean, rep = validate(df, sc)
    assert rep.ok
    ev = Evaluator(sc, clean)
    res = ev.run_all()
    assert "PA01" not in ev.rules
    assert any("PA01" in w for w in res["warnings"])
    assert res["scan"].empty and any("credit_score" in w for w in res["warnings"])


def test_cli_roundtrip(tmp_path, capsys):
    assert main(["scenarios"]) == 0
    assert main(["template", "--all", "-o", str(tmp_path / "tpl")]) == 0
    assert len(list((tmp_path / "tpl").glob("*.xlsx"))) == 6
    data = tmp_path / "d.csv"
    assert main(["sample", "-s", "aml", "-n", "2000", "-o", str(data)]) == 0
    assert main(["validate", "-s", "aml", "-d", str(data)]) == 0
    strat = tmp_path / "s.yaml"
    assert main(["init-strategy", "-s", "aml", "-o", str(strat)]) == 0
    code = main(["evaluate", "-s", "aml", "-d", str(data), "--strategy", str(strat), "-p", "alert_review_cost=120", "-o", str(tmp_path / "r"), "--no-charts"])
    assert code == 0
    assert (tmp_path / "r" / "aml_评估报告.xlsx").exists()
    assert main(["evaluate", "-s", "nope", "-d", str(data)]) == 2
