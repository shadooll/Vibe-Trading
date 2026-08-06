"""Tests for the memo parser used by pilot / agent_run (pilot.py)."""

from __future__ import annotations

from trade_tools.pilot import _extract_decision, parse_memo

SIGNAL_DATE = "2024-05-15"


class TestDecisionExtraction:
    def test_plain_buy(self) -> None:
        assert _extract_decision("### 结论\n买\n### 依据\n...") == "buy"

    def test_bold_buy_with_explanation(self) -> None:
        # Pilot 实测：agent 输出 **买** —— 说明
        assert _extract_decision("### 结论\n**买** —— R1 触发") == "buy"

    def test_final_decision_line_buy(self) -> None:
        # Pilot 实测：**决策：买入（...）**
        assert _extract_decision("依据若干\n**决策：买入（R1 成立）**") == "buy"

    def test_plain_no_buy(self) -> None:
        assert _extract_decision("### 结论\n不买\n") == "no_buy"

    def test_bold_no_buy(self) -> None:
        assert _extract_decision("### 结论\n**不买** —— 空仓等待") == "no_buy"

    def test_meta_statement(self) -> None:
        assert (
            _extract_decision("收到，不再调用工具，以现有已验证证据输出最终备忘录。")
            is None
        )


class TestParseMemo:
    def test_buy_memo_with_risk_fields(self) -> None:
        memo = (
            "### 结论\n**买** —— R1 触发\n\n"
            "### 操作计划\n"
            "止损：**21.44**\n目标：**23.42**\n仓位：**33.3%**\n"
        )
        plan, reason = parse_memo(memo, "601138.SH", SIGNAL_DATE, 22.10)
        assert plan is not None and reason == ""
        assert plan.decision == "buy"
        assert plan.stop_price == 21.44
        assert plan.target_price == 23.42
        assert round(plan.position_pct, 3) == 0.333

    def test_no_buy_memo(self) -> None:
        plan, reason = parse_memo(
            "### 结论\n不买\n理由充分", "000725.SZ", SIGNAL_DATE, 5.97
        )
        assert plan is not None and plan.decision == "no_buy"

    def test_meta_statement_no_decision(self) -> None:
        plan, reason = parse_memo(
            "收到，不再调用工具，以现有已验证证据输出最终备忘录。",
            "000725.SZ",
            SIGNAL_DATE,
            5.97,
        )
        assert plan is None
        assert "元话语" in reason

    def test_buy_without_risk_fields_fails_closed(self) -> None:
        plan, reason = parse_memo(
            "### 结论\n买\n没给操作计划", "000725.SZ", SIGNAL_DATE, 5.97
        )
        assert plan is None
        assert "缺字段" in reason

    def test_buy_low_rr_rejected(self) -> None:
        memo = "### 结论\n买\n止损：6.0 目标：6.2 仓位：50%"  # R:R < 2
        plan, reason = parse_memo(memo, "000725.SZ", SIGNAL_DATE, 6.1)
        assert plan is None
        assert "协议校验" in reason
