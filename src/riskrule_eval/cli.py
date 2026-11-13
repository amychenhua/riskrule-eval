"""命令行入口：riskrule <子命令>。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import __version__
from .dataio import read_data
from .engine import Evaluator
from .expr import ExprError
from .schema import Scenario, list_scenarios, load_scenario, load_strategy
from .validate import validate


def _scenarios(args) -> List[Scenario]:
    if getattr(args, "all", False):
        return list_scenarios()
    if not args.scenario:
        raise SystemExit("请用 -s 指定场景，或加 --all 处理全部场景。可用场景见 riskrule scenarios")
    return [load_scenario(args.scenario)]


def _params(items: Optional[List[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"参数格式应为 名称=数值，收到：{it}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"参数 {k} 的值不是数字：{v}") from None
    return out


def cmd_scenarios(args) -> int:
    print(f"{'代码':<22}{'名称':<10}{'字段数':<8}数据粒度")
    for sc in list_scenarios():
        print(f"{sc.key:<22}{sc.name:<10}{len(sc.fields):<8}{sc.unit}")
    return 0


def cmd_template(args) -> int:
    from .templates import data_dictionary_md, write_templates

    scs = _scenarios(args)
    for sc in scs:
        paths = write_templates(sc, args.out, header=args.header)
        print(f"[{sc.name}] {paths['xlsx']}")
    if args.all:
        p = Path(args.out) / "字段说明.md"
        p.write_text(data_dictionary_md(scs), encoding="utf-8")
        print(f"[字段说明] {p}")
    return 0


def cmd_sample(args) -> int:
    from .synth import generate

    for sc in _scenarios(args):
        df = generate(sc, n=args.n, seed=args.seed)
        out = Path(args.out)
        if args.all or out.suffix == "":
            out.mkdir(parents=True, exist_ok=True)
            out = out / f"{sc.key}_sample.csv"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".xlsx":
            df.to_excel(out, index=False, sheet_name="数据")
        else:
            df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[{sc.name}] {len(df)} 行模拟数据 → {out}")
    return 0


def cmd_validate(args) -> int:
    sc = load_scenario(args.scenario)
    _, rep = validate(read_data(args.data, sc), sc)
    print(f"[{sc.name}] {rep.summary()}")
    for i in rep.issues:
        r = i.as_row()
        cnt = f"（{r['涉及行数']} 行）" if r["涉及行数"] else ""
        print(f"  {r['级别']}  {r['字段'] or '-'}  {r['问题']}{cnt}")
    print("校验通过，可以评估。" if rep.ok else "存在错误，请修正后再评估。")
    return 0 if rep.ok else 1


def cmd_init_strategy(args) -> int:
    sc = load_scenario(args.scenario)
    doc = {"strategy": {"params": sc.default_params(), **sc.default_strategy()}}
    head = (
        f"# {sc.name}（{sc.key}）策略文件\n"
        "# rules：规则清单，expr 里可以用数据的列名和下面 params 里的参数名\n"
        "# strategies：按顺序取前两套做对比，第一套是基准\n"
        "# scan / matrix / monitor：阈值扫描、决策矩阵、PSI 监控的配置\n"
    )
    Path(args.out).write_text(head + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    print(f"已写出 {args.out}，修改后用 riskrule evaluate --strategy {args.out} 评估")
    return 0


def run_evaluate(sc: Scenario, data: str, out: str, strategy: Optional[str], params: Dict[str, float], charts: bool = True) -> int:
    from .report import findings, write_report

    df, rep = validate(read_data(data, sc), sc)
    print(f"[{sc.name}] 校验：{rep.summary()}")
    if not rep.ok:
        for i in rep.errors:
            print(f"  错误  {i.field or '-'}  {i.message}" + (f"（{i.count} 行）" if i.count else ""))
        print("存在错误，未生成报告。")
        return 1
    strat = load_strategy(strategy) if strategy else None
    try:
        ev = Evaluator(sc, df, strategy=strat, params=params)
        res = ev.run_all()
    except ExprError as e:
        print(f"策略或取值模型有误：{e}")
        return 1
    paths = write_report(res, sc, out, vrep=rep, with_charts=charts)
    for line in findings(res, sc, rep):
        print("  · " + line)
    print(f"  报告：{paths['xlsx']}")
    return 0


def cmd_evaluate(args) -> int:
    sc = load_scenario(args.scenario)
    return run_evaluate(sc, args.data, args.out, args.strategy, _params(args.param), charts=not args.no_charts)


def cmd_demo(args) -> int:
    """六个场景各生成一份模拟数据并出报告，用来快速了解产出。"""
    from .synth import generate
    from .templates import data_dictionary_md, write_templates

    root = Path(args.out)
    scs = list_scenarios()
    for sc in scs:
        write_templates(sc, root / "templates")
        data = root / "data" / f"{sc.key}_sample.csv"
        data.parent.mkdir(parents=True, exist_ok=True)
        generate(sc, n=args.n, seed=args.seed).to_csv(data, index=False, encoding="utf-8-sig")
        code = run_evaluate(sc, str(data), str(root / "reports" / sc.key), None, {}, charts=not args.no_charts)
        if code:
            return code
    (root / "templates" / "字段说明.md").write_text(data_dictionary_md(scs), encoding="utf-8")
    print(f"全部完成，输出在 {root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="riskrule", description="银行风控规则策略的收益与风险量化评估")
    p.add_argument("--version", action="version", version=f"riskrule-eval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scenarios", help="列出内置场景").set_defaults(func=cmd_scenarios)

    t = sub.add_parser("template", help="生成数据上传模板")
    t.add_argument("-s", "--scenario", help="场景代码或自定义场景 YAML 路径")
    t.add_argument("--all", action="store_true", help="生成全部场景的模板和字段说明")
    t.add_argument("-o", "--out", default="templates", help="输出目录")
    t.add_argument("--header", choices=["name", "cn"], default="name", help="表头用字段名(name)还是中文名(cn)")
    t.set_defaults(func=cmd_template)

    s = sub.add_parser("sample", help="生成模拟数据（仅用于试跑）")
    s.add_argument("-s", "--scenario")
    s.add_argument("--all", action="store_true")
    s.add_argument("-n", type=int, default=10000, help="行数")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("-o", "--out", default="data", help="输出文件(.csv/.xlsx)或目录")
    s.set_defaults(func=cmd_sample)

    v = sub.add_parser("validate", help="校验上传数据")
    v.add_argument("-s", "--scenario", required=True)
    v.add_argument("-d", "--data", required=True)
    v.set_defaults(func=cmd_validate)

    i = sub.add_parser("init-strategy", help="导出场景的示例策略文件，供修改")
    i.add_argument("-s", "--scenario", required=True)
    i.add_argument("-o", "--out", default="strategy.yaml")
    i.set_defaults(func=cmd_init_strategy)

    e = sub.add_parser("evaluate", help="评估策略并生成报告")
    e.add_argument("-s", "--scenario", required=True)
    e.add_argument("-d", "--data", required=True)
    e.add_argument("--strategy", help="策略 YAML；缺省用场景自带的示例策略")
    e.add_argument("-p", "--param", action="append", help="覆盖取值模型参数，如 -p unit_cost=20，可重复")
    e.add_argument("-o", "--out", default="report", help="报告输出目录")
    e.add_argument("--no-charts", action="store_true", help="不生成图表")
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("demo", help="用模拟数据把六个场景全部跑一遍")
    d.add_argument("-o", "--out", default="demo_output")
    d.add_argument("-n", type=int, default=10000)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--no-charts", action="store_true")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"出错：{e.args[0] if e.args else e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
