"""Tests for the Phase 2b as-of sandbox (agent_run.py).

Requires the agent package (``src.*``) — skipped when running the trade_tools
suite without ``PYTHONPATH=agent``.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("src.agent.tools", reason="agent 包需要 PYTHONPATH=agent")

from trade_tools.agent_run import _PinnedDateTool, build_asof_registry


class _EchoTool:
    """Minimal fake tool that echoes its kwargs back as JSON."""

    name = "echo"
    description = "echo"
    parameters = {"type": "object", "properties": {}}
    repeatable = True
    is_readonly = True

    def execute(self, **kwargs: object) -> str:
        return json.dumps(kwargs, ensure_ascii=False)


def test_pinned_tool_forces_end_date() -> None:
    """agent 传未来 end_date 会被强制覆盖成决策日（无前视保障）。"""
    pinned = _PinnedDateTool(_EchoTool(), end_date="2024-05-15")
    out = json.loads(pinned.execute(symbol="000725.SZ", end_date="2099-01-01"))
    assert out["end_date"] == "2024-05-15"
    assert "2099" not in out


def test_pinned_tool_keeps_other_args() -> None:
    pinned = _PinnedDateTool(_EchoTool(), end_date="2024-05-15")
    out = json.loads(pinned.execute(codes=["000725.SZ"], start_date="2024-01-01"))
    assert out["codes"] == ["000725.SZ"]
    assert out["start_date"] == "2024-01-01"
    assert out["end_date"] == "2024-05-15"


def test_asof_registry_only_has_safe_tools() -> None:
    """沙箱注册表只含 as-of 安全工具：无 screen_market/web_search/bash 等。"""
    reg = build_asof_registry("2024-05-15")
    names = sorted(reg._tools)  # type: ignore[attr-defined]
    assert names == ["get_market_data", "load_skill", "technical_indicators"]
