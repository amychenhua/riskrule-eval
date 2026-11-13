"""示例数据生成器：给每个场景造一份带有合理相关性的模拟数据。

数据完全是随机生成的，只用于演示流程和跑通测试，不代表任何真实客群，
用它得出的阈值和结论不能用于真实决策。
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from .schema import Scenario


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate(logit: np.ndarray, target: float) -> np.ndarray:
    """平移截距，使平均概率等于目标正样本率。"""
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(logit + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return _sigmoid(logit + (lo + hi) / 2)


def _times(rng: np.random.Generator, n: int, with_time: bool = True) -> pd.Series:
    start = pd.Timestamp("2026-01-01")
    secs = rng.integers(0, 181 * 86400, n)
    t = start + pd.to_timedelta(secs, unit="s")
    t = pd.Series(t)
    return t if with_time else t.dt.normalize()


def _month_end(t: pd.Series) -> pd.Series:
    return t.dt.normalize() + pd.offsets.MonthEnd(0)


def _ids(prefix: str, n: int, width: int = 8) -> list:
    return [f"{prefix}{i:0{width}d}" for i in range(1, n + 1)]


def _cust(rng: np.random.Generator, n: int, prefix: str) -> list:
    return [f"{prefix}{i:07d}" for i in rng.integers(1, max(n // 2, 2), n)]


def _score_high(rng, p, n, noise=9.0):
    """越高越危险的 0-100 评分。"""
    z = np.log(p / (1 - p))
    z = (z - z.mean()) / z.std()
    return np.clip(np.round(42 + 15 * z + rng.normal(0, noise, n), 1), 0, 100)


def _score_low(rng, p, n, noise=22.0):
    """越低越危险的 300-850 评分。"""
    z = np.log(p / (1 - p))
    z = (z - z.mean()) / z.std()
    return np.clip(np.round(660 - 45 * z + rng.normal(0, noise, n)), 300, 850)


# ---------------------------------------------------------------- 反洗钱
def _aml(rng: np.random.Generator, n: int) -> pd.DataFrame:
    amt = np.round(rng.lognormal(9.0, 1.5, n), 2)
    channel = rng.choice(["counter", "online", "mobile", "atm"], n, p=[0.12, 0.3, 0.5, 0.08])
    cross = rng.binomial(1, 0.05, n)
    cash = np.where(np.isin(channel, ["counter", "atm"]), rng.binomial(1, 0.45, n), 0)
    region = rng.choice(["low", "medium", "high"], n, p=[0.85, 0.12, 0.03])
    risk = rng.choice(["low", "medium", "high"], n, p=[0.8, 0.17, 0.03])
    age = rng.integers(1, 4000, n)
    burst = rng.random(n) < 0.03
    cnt24 = 1 + np.where(burst, rng.poisson(8.0, n), rng.poisson(1.5, n))
    sum24 = np.round(amt * (1 + rng.gamma(1.0, 0.8, n)), 2)
    ratio = np.clip(rng.normal(0.5, 0.25, n), 0, 1.5).round(3)
    newcp = rng.poisson(1.0, n)
    split = rng.random(n) < 0.025
    near = np.where(split, 1 + rng.poisson(2.5, n), rng.poisson(0.1, n))
    dormant = rng.binomial(1, 0.02, n)
    logit = (
        1.1 * (cash & (amt >= 50000))
        + 0.75 * np.clip(near, 0, 4)
        + 1.6 * ((ratio >= 0.95) & (sum24 >= 100000))
        + 2.4 * (cross & (region == "high"))
        + 0.25 * np.clip(cnt24 - 4, 0, 10)
        + 2.0 * (dormant & (amt >= 20000))
        + 1.5 * ((risk == "high") & (newcp >= 3))
        + 0.8 * (risk == "high")
        + 0.35 * np.log1p(amt / 50000)
    )
    p = _calibrate(logit, 0.02)
    return pd.DataFrame(
        {
            "txn_id": _ids("T", n, 10),
            "customer_id": _cust(rng, n, "C1"),
            "txn_time": _times(rng, n),
            "txn_amount": amt,
            "channel": channel,
            "is_cross_border": cross,
            "cash_flag": cash,
            "counterparty_region_risk": region,
            "customer_risk_level": risk,
            "account_age_days": age,
            "txn_cnt_24h": cnt24,
            "amt_sum_24h": sum24,
            "in_out_ratio_7d": ratio,
            "new_counterparty_cnt_7d": newcp,
            "near_threshold_cnt_7d": near,
            "dormant_reactivated": dormant,
            "aml_score": _score_high(rng, p, n),
            "label_suspicious": rng.binomial(1, p),
        }
    )


# ---------------------------------------------------------------- 反电诈
def _telecom(rng: np.random.Generator, n: int) -> pd.DataFrame:
    amt = np.round(rng.lognormal(8.0, 1.2, n), 2)
    channel = rng.choice(["mobile", "online", "counter", "atm"], n, p=[0.7, 0.2, 0.06, 0.04])
    age = rng.integers(18, 86, n)
    new = rng.binomial(1, 0.25, n)
    payee_age = np.where(new == 1, rng.integers(0, 400, n), rng.integers(30, 3000, n))
    tag = rng.binomial(1, 0.008, n)
    night = rng.binomial(1, 0.1, n)
    dev = rng.binomial(1, 0.05, n)
    remote = rng.binomial(1, 0.008, n)
    call = rng.binomial(1, 0.04, n)
    cnt1h = 1 + rng.poisson(0.3, n)
    ratio = np.round(rng.lognormal(0.0, 0.8, n), 2)
    first_large = rng.binomial(1, 0.06, n)
    logit = (
        4.0 * tag
        + 3.2 * remote
        + 2.6 * (call & new)
        + 2.2 * ((age >= 60) & new & first_large)
        + 1.8 * (dev & night & (amt >= 10000))
        + 0.9 * ((ratio >= 5) & new)
        + 0.5 * np.clip(cnt1h - 2, 0, 5)
        + 0.6 * new
        + 0.3 * np.log1p(ratio)
    )
    p = _calibrate(logit, 0.015)
    y = rng.binomial(1, p)
    rec = np.where(y == 1, np.round(amt * rng.uniform(0, 0.3, n), 2), 0.0)
    return pd.DataFrame(
        {
            "txn_id": _ids("P", n, 10),
            "customer_id": _cust(rng, n, "C2"),
            "txn_time": _times(rng, n),
            "txn_amount": amt,
            "channel": channel,
            "customer_age": age,
            "payee_is_new": new,
            "payee_account_age_days": payee_age,
            "payee_risk_tag": tag,
            "night_txn": night,
            "device_changed": dev,
            "remote_control_app_flag": remote,
            "call_in_progress": call,
            "txn_cnt_1h": cnt1h,
            "amt_ratio_to_avg_90d": ratio,
            "first_large_transfer": first_large,
            "fraud_score": _score_high(rng, p, n),
            "recovered_amount": rec,
            "label_fraud": y,
        }
    )


# ---------------------------------------------------------------- 贷前反欺诈
def _pre_fraud(rng: np.random.Generator, n: int) -> pd.DataFrame:
    t = _times(rng, n)
    w = np.array([1, 1, 1, 1, 1, 1, 2, 4, 6, 7, 7, 6, 5, 5, 6, 6, 6, 6, 6, 6, 6, 5, 3, 2], dtype=float)
    hour = rng.choice(np.arange(24), n, p=w / w.sum())
    t = t.dt.normalize() + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
    amt = np.round(rng.uniform(0.5, 20, n) * 10000, -2)
    channel = rng.choice(["app", "h5", "partner", "branch"], n, p=[0.45, 0.15, 0.35, 0.05])
    dev = 1 + rng.poisson(0.6, n)
    idp = 1 + rng.poisson(0.25, n)
    ip = 1 + rng.poisson(0.7, n)
    idbl = rng.binomial(1, 0.008, n)
    cbl = rng.binomial(1, 0.03, n)
    live = rng.poisson(0.15, n)
    incons = rng.binomial(1, 0.1, n)
    gps = rng.binomial(1, 0.08, n)
    logit = (
        4.0 * idbl
        + 2.2 * cbl
        + 0.9 * np.clip(dev - 2, 0, 6)
        + 1.2 * np.clip(idp - 1, 0, 4)
        + 1.0 * np.clip(live - 1, 0, 4)
        + 1.6 * ((hour <= 5) & (incons == 1))
        + 0.25 * np.clip(ip - 3, 0, 6)
        + 0.5 * incons
        + 0.5 * gps
        + 0.4 * (channel == "partner")
    )
    p = _calibrate(logit, 0.03)
    return pd.DataFrame(
        {
            "app_id": _ids("A", n, 10),
            "customer_id": _cust(rng, n, "C3"),
            "apply_time": t,
            "apply_amount": amt,
            "channel": channel,
            "apply_hour": hour,
            "device_app_cnt_7d": dev,
            "id_phone_cnt": idp,
            "ip_app_cnt_1d": ip,
            "id_blacklist_hit": idbl,
            "contact_blacklist_hit": cbl,
            "liveness_fail_cnt": live,
            "info_inconsistent": incons,
            "gps_city_mismatch": gps,
            "fraud_score": _score_high(rng, p, n),
            "expected_profit": np.round(amt * rng.uniform(0.03, 0.08, n)),
            "label_fraud": rng.binomial(1, p),
        }
    )


# ---------------------------------------------------------------- 贷前审批
def _pre_approval(rng: np.random.Generator, n: int) -> pd.DataFrame:
    amt = np.round(rng.uniform(1, 30, n) * 10000, -2)
    term = rng.choice([6, 12, 24, 36], n, p=[0.15, 0.4, 0.3, 0.15])
    income = np.round(rng.lognormal(9.2, 0.5, n), -2)
    dti = np.clip(rng.normal(0.35, 0.18, n), 0, 1.5).round(3)
    q = rng.choice(np.arange(9), n, p=[0.33, 0.24, 0.15, 0.10, 0.08, 0.05, 0.02, 0.02, 0.01])
    od = rng.poisson(0.3, n)
    bl = rng.binomial(1, 0.03, n)
    ab = rng.binomial(1, 0.06, n)
    inc = rng.binomial(1, 0.12, n)
    logit = (
        0.22 * np.clip(q - 2, 0, 6)
        + 3.0 * np.clip(dti - 0.5, 0, 1)
        + 0.8 * np.clip(od, 0, 4)
        + 3.0 * bl
        + 1.0 * ab
        + 0.4 * inc
    )
    p = _calibrate(logit, 0.08)
    return pd.DataFrame(
        {
            "app_id": _ids("A", n, 10),
            "customer_id": _cust(rng, n, "C4"),
            "apply_time": _times(rng, n),
            "apply_amount": amt,
            "term_months": term,
            "income_monthly": income,
            "dti": dti,
            "query_cnt_3m": q,
            "overdue_cnt_24m": od,
            "blacklist_hit": bl,
            "large_abnormal_txn": ab,
            "info_inconsistent": inc,
            "credit_score": _score_low(rng, p, n),
            "expected_profit": np.round(amt * rng.uniform(0.03, 0.09, n)),
            "loss_amount": np.round(amt * rng.uniform(0.4, 0.9, n)),
            "label_bad": rng.binomial(1, p),
        }
    )


# ---------------------------------------------------------------- 贷中调额
def _limit(rng: np.random.Generator, n: int) -> pd.DataFrame:
    limit = rng.choice([5000, 10000, 20000, 30000, 50000, 80000, 100000], n, p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.07, 0.03]).astype(float)
    maxed = rng.random(n) < 0.08
    util = np.clip(np.where(maxed, rng.beta(9, 1.0, n) * 1.04, rng.beta(2, 3, n)), 0, 1.05).round(3)
    bal = np.round(limit * util, 2)
    mob = rng.integers(6, 121, n)
    dpd = np.where(rng.random(n) < 0.9, 0, rng.choice([5, 15, 30, 45, 60], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]))
    over = rng.poisson(0.1, n) + (util > 1).astype(int)
    cash = rng.beta(1, 8, n).round(3)
    minpay = rng.binomial(6, np.where(maxed, 0.45, 0.12))
    q = rng.poisson(1.0, n)
    debt = rng.normal(0.05, 0.3, n).round(3)
    logit = (
        2.4 * (dpd >= 30)
        + 0.8 * (dpd > 0)
        + 1.8 * ((util >= 0.9) & (minpay >= 3))
        + 1.5 * (cash >= 0.5)
        + 1.3 * (debt >= 0.5)
        + 0.2 * np.clip(q - 2, 0, 6)
        + 1.0 * np.clip(over, 0, 3)
        + 1.2 * util
        + 0.25 * minpay
    )
    p = _calibrate(logit, 0.05)
    proposed = np.maximum(np.ceil(bal / 1000) * 1000, limit * 0.5)
    reduction = np.clip(limit - proposed, 0, None)
    return pd.DataFrame(
        {
            "account_id": _ids("AC", n, 10),
            "customer_id": _cust(rng, n, "C5"),
            "review_date": _month_end(_times(rng, n, with_time=False)),
            "current_limit": limit,
            "balance": bal,
            "utilization": util,
            "months_on_book": mob,
            "dpd_max_6m": dpd,
            "overlimit_cnt_6m": over,
            "cash_advance_ratio_3m": cash,
            "min_pay_cnt_6m": minpay,
            "external_query_cnt_3m": q,
            "external_debt_growth_6m": debt,
            "behavior_score": _score_low(rng, p, n),
            "limit_reduction": reduction,
            "annual_revenue": np.round(bal * 0.12 + limit * 0.005, 2),
            "label_bad_12m": rng.binomial(1, p),
        }
    )


# ---------------------------------------------------------------- 贷后预警
def _post_loan(rng: np.random.Generator, n: int) -> pd.DataFrame:
    bal = np.round(rng.lognormal(10.5, 0.8, n), 2)
    mob = rng.integers(1, 61, n)
    dpd = np.where(rng.random(n) < 0.88, 0, rng.integers(1, 30, n))
    delay = np.round(rng.gamma(0.8, 1.5, n) + 0.15 * dpd, 2)
    ratio = np.clip(rng.normal(1.0, 0.15, n), 0, 1.5).round(3)
    ext_od = rng.binomial(1, 0.05, n)
    q1 = rng.poisson(0.5, n)
    multi = rng.poisson(0.3, n)
    salary = rng.binomial(1, 0.06, n)
    lost = rng.binomial(1, 0.02, n)
    lit = rng.binomial(1, 0.01, n)
    logit = (
        2.2 * ext_od
        + 1.9 * salary
        + 0.3 * np.clip(delay - 2, 0, 12)
        + 2.5 * np.clip(0.9 - ratio, 0, 1)
        + 0.8 * np.clip(multi - 1, 0, 4)
        + 2.6 * lost
        + 2.4 * lit
        + 0.3 * np.clip(q1 - 1, 0, 5)
        + 0.05 * dpd
    )
    p = _calibrate(logit, 0.06)
    return pd.DataFrame(
        {
            "loan_id": _ids("L", n, 10),
            "customer_id": _cust(rng, n, "C6"),
            "obs_date": _month_end(_times(rng, n, with_time=False)),
            "outstanding_balance": bal,
            "mob": mob,
            "current_dpd": dpd,
            "repay_delay_days_avg_3m": delay,
            "payment_ratio_3m": ratio,
            "new_external_overdue": ext_od,
            "external_query_cnt_1m": q1,
            "multi_lending_cnt_new_3m": multi,
            "salary_inflow_drop": salary,
            "contact_lost": lost,
            "litigation_flag": lit,
            "warning_score": _score_high(rng, p, n),
            "monthly_revenue": np.round(bal * 0.01, 2),
            "label_default_6m": rng.binomial(1, p),
        }
    )


_GENERATORS: Dict[str, Callable[[np.random.Generator, int], pd.DataFrame]] = {
    "aml": _aml,
    "anti_telecom_fraud": _telecom,
    "pre_loan_fraud": _pre_fraud,
    "pre_loan_approval": _pre_approval,
    "credit_limit": _limit,
    "post_loan_warning": _post_loan,
}


def generate(scenario: Scenario, n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """生成某个场景的模拟数据，列顺序与上传模板一致。"""
    if scenario.key not in _GENERATORS:
        raise KeyError(f"场景 {scenario.key} 没有内置的示例数据生成器")
    rng = np.random.default_rng(seed)
    df = _GENERATORS[scenario.key](rng, int(n))
    return df[scenario.field_names]
