"""Regression coverage for identity and numeric grounding (#887, #886)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from src.agent.context import ContextBuilder
from src.agent.grounding import GroundingLedger
from src.agent.loop import AgentLoop, _is_tool_success
from src.agent.tools import BaseTool, ToolRegistry
from src.agent.trace import TraceWriter


def _resolver_payload(
    symbol: str = "562500.SS",
    *,
    candidates: list[dict[str, Any]] | None = None,
    query: str = "机器人ETF",
) -> str:
    rows = candidates
    if rows is None:
        rows = [
            {
                "symbol": symbol,
                "name": "机器人ETF",
                "market": "cn",
                "type": "ETF",
                "source": "yahoo",
                "also_from": ["eastmoney"],
            }
        ]
    return json.dumps(
        {
            "ok": True,
            "source": "symbol_search",
            "data": {
                "query": query,
                "count": len(rows),
                "candidates": rows,
                "sources": {"eastmoney": "ok", "yahoo": "ok"},
            },
        },
        ensure_ascii=False,
    )


def _market_payload(symbol: str = "562500.SS") -> str:
    return json.dumps(
        {
            symbol: [
                {
                    "trade_date": "2026-06-23",
                    "open": 1.141,
                    "high": 1.164,
                    "low": 1.121,
                    "close": 1.137,
                    "volume": 123456,
                },
                {
                    "trade_date": "2026-06-24",
                    "open": 1.137,
                    "high": 1.180,
                    "low": 1.110,
                    "close": 1.171,
                    "volume": 234567,
                },
            ],
            "_provenance": {
                symbol: {
                    "source": "yahoo",
                    "requested_source": "auto",
                    "detected_source": "yahoo",
                    "fallback_used": False,
                    "currency_conversion": "none",
                }
            },
        }
    )


class _ResolverTool(BaseTool):
    name = "search_symbol"
    description = "Resolve a company or instrument name."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    repeatable = True

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return self.result


class _MarketTool(BaseTool):
    name = "get_market_data"
    description = "Fetch OHLCV bars."
    parameters = {
        "type": "object",
        "properties": {"codes": {"type": "array", "items": {"type": "string"}}},
        "required": ["codes"],
    }
    repeatable = True

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return self.result


class _PrivateCompanySkillTool(BaseTool):
    name = "load_skill"
    description = "Load a skill."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    repeatable = True

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return json.dumps({"status": "ok", "content": "private company workflow"})


def _tool_call(call_id: str, tool_name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, name=tool_name, arguments=arguments)


def _build_direct_agent(
    tmp_path: Path,
    resolver_result: str,
) -> tuple[
    AgentLoop, _ResolverTool, _MarketTool, _PrivateCompanySkillTool, TraceWriter
]:
    resolver = _ResolverTool(resolver_result)
    market = _MarketTool(_market_payload())
    private_skill = _PrivateCompanySkillTool()
    registry = ToolRegistry()
    for tool in (resolver, market, private_skill):
        registry.register(tool)
    agent = AgentLoop(registry=registry, llm=SimpleNamespace(), max_iterations=5)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    agent._grounding = GroundingLedger(
        run_dir=run_dir,
        user_message="分析机器人ETF并给出买入价",
    )
    return agent, resolver, market, private_skill, TraceWriter(run_dir)


def test_ok_false_tool_envelope_is_failure() -> None:
    """Business failures must not be recorded as successful tool calls (#886)."""
    assert _is_tool_success('{"ok": false, "error": "upstream failed"}') is False
    assert _is_tool_success('{"success": false, "message": "denied"}') is False
    assert _is_tool_success('{"status": "failed"}') is False
    assert _is_tool_success('{"ok": true, "data": {}}') is True


def test_resolver_and_consumer_in_same_batch_cannot_race(
    tmp_path: Path,
) -> None:
    """A consumer sees the identity snapshot from before its whole LLM batch."""
    agent, resolver, market, _, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload(),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [
            _tool_call("resolve", "search_symbol", query="机器人ETF"),
            _tool_call(
                "prices-too-early",
                "get_market_data",
                codes=["562500.SH"],
                start_date="2026-06-23",
                end_date="2026-06-24",
            ),
        ],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )

    assert resolver.calls == 1
    assert market.calls == 0
    assert agent._grounding.authorized_symbols == {"562500.SS"}
    blocked = [json.loads(message["content"]) for message in messages]
    assert any(item.get("error_code") == "identity_required" for item in blocked)

    agent._process_tool_calls(
        [
            _tool_call(
                "prices-after-lock",
                "get_market_data",
                codes=["562500.SS"],
                start_date="2026-06-23",
                end_date="2026-06-24",
                source="auto",
            )
        ],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert market.calls == 1
    artifact = json.loads(
        (tmp_path / "run" / "artifacts" / "grounding_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["identity"]["status"] == "locked"
    assert any(
        record["field"] == "close"
        and record["value"] == 1.137
        and record["source"] == "yahoo"
        and record["currency"] == "CNY"
        and record["currency_conversion"] == "none"
        for record in artifact["evidence"]
    )


def test_market_sensitive_skill_waits_for_prior_identity_batch(
    tmp_path: Path,
) -> None:
    """Workflow selection cannot race the resolver in the same assistant turn."""
    agent, resolver, _, skill, trace = _build_direct_agent(
        tmp_path, _resolver_payload()
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [
            _tool_call("resolve", "search_symbol", query="机器人ETF"),
            _tool_call("workflow-too-early", "load_skill", name="valuation-model"),
        ],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )

    assert resolver.calls == 1
    assert skill.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_required"

    agent._process_tool_calls(
        [_tool_call("workflow-after-lock", "load_skill", name="valuation-model")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert skill.calls == 1


def test_bare_us_ticker_is_allowed_only_for_explicit_transport_contract(
    tmp_path: Path,
) -> None:
    """SEC's bare ticker consumes AAPL.US without permitting venue aliases."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 AAPL.US 的价格",
    )

    sec = ledger.authorize_tool_call(
        "get_sec_filings",
        {"ticker": "AAPL"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="sec",
    )
    market_alias = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["AAPL"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="prices",
    )

    assert sec.allowed is True
    assert market_alias.allowed is False
    assert market_alias.error_code == "identity_mismatch"


def test_stale_history_identity_does_not_unlock_new_subject(tmp_path: Path) -> None:
    """A previous turn's AAPL identity cannot authorize a SpaceX price request."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="SpaceX 在什么价格买入比较合适？",
        history=[{"role": "user", "content": "请分析 AAPL.US"}],
    )

    authorization = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["AAPL.US"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="stale-price",
    )

    assert ledger.authorized_symbols == set()
    assert authorization.allowed is False
    assert authorization.error_code == "identity_required"


def test_single_clean_not_found_source_is_not_enough_for_private_routing(
    tmp_path: Path,
) -> None:
    """A partial resolver outage cannot turn a public entity into private research."""
    resolver_result = json.dumps(
        {
            "ok": True,
            "source": "symbol_search",
            "data": {
                "query": "Acme",
                "count": 0,
                "candidates": [],
                "sources": {
                    "eastmoney": "ok",
                    "yahoo": "HTTP 429",
                },
            },
        }
    )
    agent, _, _, private_skill, trace = _build_direct_agent(tmp_path, resolver_result)
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="Acme")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("private", "load_skill", name="private-company-research")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert agent._grounding.identity_status == "invalidated"
    assert private_skill.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_required"


def test_explicit_symbol_and_resolver_suffix_alias_become_conflicting(
    tmp_path: Path,
) -> None:
    """A later .SH resolver result cannot silently replace explicit .SS identity."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 562500.SS 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "机器人ETF"},
        result=_resolver_payload("562500.SH"),
        call_id="resolver-alias",
        success=True,
    )

    authorization = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["562500.SS"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="prices",
    )

    assert ledger.identity_status == "conflicting"
    assert authorization.allowed is False
    assert authorization.error_code == "identity_conflict"


def test_locked_symbol_rejects_silent_exchange_suffix_rewrite(
    tmp_path: Path,
) -> None:
    """The consumer may not silently turn the resolver's .SS into .SH."""
    agent, _, market, _, trace = _build_direct_agent(tmp_path, _resolver_payload())
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="机器人ETF")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("wrong-suffix", "get_market_data", codes=["562500.SH"])],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert market.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_mismatch"


def test_listed_identity_blocks_private_company_workflow(
    tmp_path: Path,
) -> None:
    """Model memory cannot relabel a strongly resolved listing as private."""
    candidates = [
        {
            "symbol": "SPCX.US",
            "name": "SpaceX",
            "market": "us",
            "type": "equity",
            "exchange": "NMS",
            "source": "eastmoney",
            "also_from": ["yahoo"],
            "cik": "0001181412",
        }
    ]
    agent, _, _, private_skill, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload("SPCX.US", candidates=candidates, query="SpaceX"),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="SpaceX")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("private", "load_skill", name="private-company-research")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert private_skill.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_conflict"
    validation = agent._grounding.validate_final_answer(
        "SpaceX is a private company and is not publicly traded."
    )
    assert validation.valid is False
    assert any(
        issue["code"] == "listed_identity_relabelled_private"
        for issue in validation.issues
    )


def test_not_found_identity_allows_private_company_workflow(
    tmp_path: Path,
) -> None:
    """A clean multi-source not-found result keeps genuine private research usable."""
    agent, _, _, private_skill, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload(candidates=[], query="Acme Private Labs"),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="Acme Private Labs")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("private", "load_skill", name="private-company-research")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert private_skill.calls == 1
    assert agent._grounding.identity_status == "not_found"


def test_ambiguous_resolution_keeps_consumers_blocked(tmp_path: Path) -> None:
    """Multiple weak candidates must become first-class ambiguous state."""
    candidates = [
        {"symbol": "ABC.US", "name": "ABC Holdings", "source": "yahoo"},
        {"symbol": "ABC.HK", "name": "ABC Group", "source": "eastmoney"},
    ]
    agent, _, market, _, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload(candidates=candidates, query="ABC"),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="ABC")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("prices", "get_market_data", codes=["ABC.US"])],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert agent._grounding.identity_status == "ambiguous"
    assert market.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_conflict"


def test_final_numeric_gate_rejects_known_trace_contradiction(tmp_path: Path) -> None:
    """Known 1.11-1.18 evidence cannot become 0.88-0.91 in the answer."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 562500.SS 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={
            "codes": ["562500.SS"],
            "source": "auto",
        },
        result=_market_payload(),
        call_id="prices",
        success=True,
    )

    bad = ledger.validate_final_answer("""| 日期 | 开盘 | 最高 | 最低 | 收盘 |
|---|---:|---:|---:|---:|
| 2026-06-23 | 0.895 | 0.907 | 0.892 | 0.903 |

建议重仓买入价为 0.881。""")
    good = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）在 2026-06-23 的已观测开盘价为 1.141，收盘价为 1.137。"
    )

    assert bad.valid is False
    assert any(issue["code"] == "numeric_claim_conflict" for issue in bad.issues)
    assert good.valid is True


def test_numeric_gate_validates_derived_formula_and_provenance(tmp_path: Path) -> None:
    """A derived entry level must calculate correctly from observed evidence."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 562500.SS 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["562500.SS"], "source": "auto"},
        result=_market_payload(),
        call_id="prices",
        success=True,
    )

    bad_math = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）的推导买入价：(1.141 + 1.137) / 2 = 0.881。"
    )
    no_observed_input = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）的推导买入价：(0.88 + 0.90) / 2 = 0.89。"
    )
    good = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）的推导买入价：(1.141 + 1.137) / 2 = 1.139。"
    )
    missing_provenance = ledger.validate_final_answer(
        "2026-06-23 的已观测收盘价为 1.137。"
    )

    assert bad_math.valid is False
    assert no_observed_input.valid is False
    assert good.valid is True
    assert missing_provenance.valid is False
    assert {issue["code"] for issue in missing_provenance.issues} >= {
        "canonical_symbol_not_surfaced",
        "data_source_not_surfaced",
        "currency_not_surfaced",
    }


class _Response:
    def __init__(
        self,
        *,
        content: str = "",
        tool_calls: list[SimpleNamespace] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None
        self.has_tool_calls = bool(self.tool_calls)


class _CorrectingLLM:
    model_name = "grounding-test"

    def __init__(self) -> None:
        self.responses = [
            _Response(
                tool_calls=[
                    _tool_call("resolve", "search_symbol", query="机器人ETF"),
                    _tool_call(
                        "too-early",
                        "get_market_data",
                        codes=["562500.SH"],
                        start_date="2026-06-23",
                        end_date="2026-06-24",
                    ),
                ]
            ),
            _Response(
                tool_calls=[
                    _tool_call(
                        "prices",
                        "get_market_data",
                        codes=["562500.SS"],
                        start_date="2026-06-23",
                        end_date="2026-06-24",
                        source="auto",
                    )
                ]
            ),
            _Response(content="建议买入价为 0.881。"),
            _Response(
                content=(
                    "562500.SS（Yahoo，CNY）在 2026-06-23 的已观测收盘价为 1.137。"
                )
            ),
        ]

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
        on_reasoning_chunk: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> _Response:
        response = self.responses.pop(0)
        if response.content and on_text_chunk:
            on_text_chunk(response.content)
        return response

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Response:
        return _Response()


def test_agent_loop_rejects_then_corrects_ungrounded_final_answer(
    tmp_path: Path,
) -> None:
    """Rejected numeric drafts never become the returned or streamed answer."""
    resolver = _ResolverTool(_resolver_payload())
    market = _MarketTool(_market_payload())
    registry = ToolRegistry()
    registry.register(resolver)
    registry.register(market)
    events: list[tuple[str, dict[str, Any]]] = []
    agent = AgentLoop(
        registry=registry,
        llm=_CorrectingLLM(),
        max_iterations=4,
        event_callback=lambda event, data: events.append((event, data)),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)

    result = agent.run("请分析机器人ETF并给出买入价")

    assert result["status"] == "success"
    assert "1.137" in result["content"]
    assert "0.881" not in result["content"]
    assert market.calls == 1
    streamed = "".join(
        data.get("delta", "") for event, data in events if event == "text_delta"
    )
    assert "0.881" not in streamed
    assert "1.137" in streamed
    completed_thinking = "".join(
        data.get("content", "") for event, data in events if event == "thinking_done"
    )
    assert "0.881" not in completed_thinking
    artifact = json.loads(
        (run_dir / "artifacts" / "grounding_evidence.json").read_text(encoding="utf-8")
    )
    assert artifact["validations"][0]["valid"] is False
    assert artifact["validations"][-1]["valid"] is True


def test_agent_loop_grounding_disabled_accepts_draft_as_is(tmp_path: Path) -> None:
    """grounding_enabled=False: ungrounded draft is returned unchanged, no correction loop."""
    resolver = _ResolverTool(_resolver_payload())
    market = _MarketTool(_market_payload())
    registry = ToolRegistry()
    registry.register(resolver)
    registry.register(market)
    events: list[tuple[str, dict[str, Any]]] = []
    agent = AgentLoop(
        registry=registry,
        llm=_CorrectingLLM(),
        max_iterations=4,
        grounding_enabled=False,
        event_callback=lambda event, data: events.append((event, data)),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)

    result = agent.run("请分析机器人ETF并给出买入价")

    assert result["status"] == "success"
    # 未观测价草稿未被拒：直接成为最终答案。
    assert "0.881" in result["content"]
    # grounding 完全关闭：不产出 evidence artifact，无 answer_rejected 事件。
    assert not (run_dir / "artifacts" / "grounding_evidence.json").exists()
    assert not any(event == "answer_rejected" for event, _ in events)


_SHORTLIST_QUERY = "A股低价高增长股票"
_SHORTLIST_PAYLOAD = _resolver_payload(
    candidates=[
        {
            "symbol": "000543.SZ",
            "name": "皖能电力",
            "market": "cn",
            "source": "eastmoney",
        },
        {
            "symbol": "000727.SZ",
            "name": "冠捷科技",
            "market": "cn",
            "source": "eastmoney",
        },
    ],
    query=_SHORTLIST_QUERY,
)
_NARROWED_PAYLOAD = _resolver_payload(
    candidates=[
        {
            "symbol": "000543.SZ",
            "name": "皖能电力",
            "market": "cn",
            "source": "eastmoney",
        },
    ],
    query="000543.SZ",
)
_SCREENED_MARKET_PAYLOAD = json.dumps(
    {
        "000543.SZ": [
            {
                "trade_date": "2026-08-01",
                "open": 7.9,
                "high": 8.5,
                "low": 7.9,
                "close": 8.2,
                "volume": 100000,
            }
        ],
        "_provenance": {
            "000543.SZ": {
                "source": "tencent",
                "requested_source": "auto",
                "detected_source": "tencent",
                "fallback_used": False,
                "currency_conversion": "none",
            }
        },
    }
)


def _screened_ledger(tmp_path: Path) -> GroundingLedger:
    """Return a ledger that screened broadly, then locked one candidate."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="推荐A股低价高增长股票并给出买入价",
    )
    for call_id, query, payload in (
        ("shortlist", _SHORTLIST_QUERY, _SHORTLIST_PAYLOAD),
        ("narrow", "000543.SZ", _NARROWED_PAYLOAD),
    ):
        ledger.authorize_tool_call(
            "search_symbol",
            {"query": query},
            batch_authorized_symbols=ledger.authorized_symbols,
            call_id=call_id,
        )
        ledger.ingest_tool_result(
            tool_name="search_symbol",
            arguments={"query": query},
            result=payload,
            call_id=call_id,
            success=True,
        )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["000543.SZ"]},
        result=_SCREENED_MARKET_PAYLOAD,
        call_id="prices",
        success=True,
    )
    return ledger


def test_screening_shortlist_does_not_block_workflow_selection(tmp_path: Path) -> None:
    """A many-candidate screening result is an answer, not a stalled resolution (#955)."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="推荐A股低价高增长股票并给出买入价",
    )
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": _SHORTLIST_QUERY},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="shortlist",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": _SHORTLIST_QUERY},
        result=_SHORTLIST_PAYLOAD,
        call_id="shortlist",
        success=True,
    )

    assert ledger.identity_status == "ambiguous"
    skill = ledger.authorize_tool_call(
        "load_skill",
        {"name": "stock-selection"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="skill",
        batch_identity_status=ledger.identity_status,
    )

    assert skill.allowed is True


def test_narrowed_lock_retires_the_screening_shortlist(tmp_path: Path) -> None:
    """Locking a shortlisted candidate must unblock the run's final answer (#955)."""
    ledger = _screened_ledger(tmp_path)

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"000543.SZ"}
    prices = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["000543.SZ"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="prices",
        batch_identity_status=ledger.identity_status,
    )

    assert prices.allowed is True


def test_price_validation_ignores_symbol_date_and_quantity_digits(
    tmp_path: Path,
) -> None:
    """Ticker, calendar, holding-period, and position-cost digits are not prices (#955)."""
    ledger = _screened_ledger(tmp_path)

    for draft in (
        "000543.SZ 截至 8 月 3 日收盘价 8.20 CNY（source: tencent）",
        "000543.SZ 买入价 8.20 CNY（100 股成本 820 CNY；source: tencent）",
        "000543.SZ 建议持有 1–4 周，买入价 8.20 CNY（source: tencent）",
    ):
        result = ledger.validate_final_answer(draft)
        assert result.valid is True, (draft, result.issues)


def test_price_validation_still_rejects_a_quote_outside_observed_range(
    tmp_path: Path,
) -> None:
    """Masking non-price digits must not weaken the contradiction check (#955)."""
    ledger = _screened_ledger(tmp_path)

    result = ledger.validate_final_answer(
        "000543.SZ 收盘价 42.00 CNY（source: tencent）"
    )

    assert result.valid is False
    assert [issue["code"] for issue in result.issues] == ["numeric_claim_conflict"]


def test_screening_run_reaches_a_final_answer_through_the_agent_loop(
    tmp_path: Path,
) -> None:
    """End-to-end: screen, load a workflow skill, narrow, quote, and answer (#955)."""
    agent, resolver, market, skill, trace = _build_direct_agent(
        tmp_path, _SHORTLIST_PAYLOAD
    )
    market.result = _SCREENED_MARKET_PAYLOAD
    agent._grounding = GroundingLedger(
        run_dir=Path(agent.memory.run_dir),
        user_message="推荐A股低价高增长股票并给出买入价",
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    def batch(*calls: SimpleNamespace, iteration: int) -> None:
        agent._process_tool_calls(
            list(calls), ContextBuilder, messages, trace, react_trace, iteration
        )

    batch(_tool_call("shortlist", "search_symbol", query=_SHORTLIST_QUERY), iteration=1)
    batch(_tool_call("workflow", "load_skill", name="stock-selection"), iteration=2)
    assert skill.calls == 1

    resolver.result = _NARROWED_PAYLOAD
    batch(_tool_call("narrow", "search_symbol", query="000543.SZ"), iteration=3)
    batch(_tool_call("prices", "get_market_data", codes=["000543.SZ"]), iteration=4)
    trace.close()

    assert market.calls == 1
    validation = agent._grounding.validate_final_answer(
        "000543.SZ 买入价 8.20 CNY（100 股成本 820 CNY；source: tencent）"
    )
    assert validation.valid is True, validation.issues
