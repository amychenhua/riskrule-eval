# 参与贡献

欢迎提交问题和合并请求。

## 开发环境

```bash
pip install -e ".[dev]"
pytest
```

## 约定

- 改动 `engine.py` 的计算口径时，先在 `tests/test_core.py` 的手算数据上补一个能说明问题的断言。
- 新增或修改场景 YAML 后跑 `pytest tests/test_scenarios.py`，它会检查规则和取值模型引用的字段是否都已定义。
- 改了场景字段后重新生成随仓库附带的模板和示例：

  ```bash
  riskrule demo -o /tmp/demo
  cp /tmp/demo/templates/* templates/
  cp /tmp/demo/templates/字段说明.md docs/data-templates.md
  rm -rf examples/reports && cp -r /tmp/demo/reports examples/reports
  ```

- 不要提交任何真实客户数据。示例数据一律用 `riskrule sample` 生成。
- 规则表达式求值器（`expr.py`）只放行白名单语法。要增加函数或运算符，请同时补上“拒绝不安全输入”的测试。

## 提交问题时请附上

- 场景代码和命令
- 校验输出（`riskrule validate`）
- 能复现问题的最小数据（脱敏或模拟）
