"""Ephemeral runtime-context blocks: request-only rider, never persisted."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.context import RequestContext
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ProviderConversationState
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_HISTORY_META,
    RuntimeContextBlock,
)
from tests.agent.runner_helpers import make_run_spec

PERSISTENT_TEXT = "persistent goal context"
EPHEMERAL_TEXT = "replies here are spoken aloud; keep to short plain prose"


def _make_loop(tmp_path: Path, blocks: list[RuntimeContextBlock]) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(max_tokens=4096)
    captured: dict[str, Any] = {}

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        captured["messages"] = kwargs["messages"]
        captured["provider_context"] = kwargs.get("provider_context")
        return LLMResponse(content="ok")

    provider.chat_with_retry = chat_with_retry
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    async def provide(_request: RequestContext) -> list[RuntimeContextBlock]:
        return blocks

    loop.register_runtime_context_provider(provide)
    loop._captured = captured  # type: ignore[attr-defined]
    return loop


@pytest.mark.asyncio
async def test_ephemeral_block_rides_on_request_but_never_persists(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, [
        RuntimeContextBlock(source="voice", content=EPHEMERAL_TEXT, ephemeral=True),
        RuntimeContextBlock(source="goal", content=PERSISTENT_TEXT),
    ])

    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="hello")
    await loop._process_message(msg)

    request_messages = loop._captured["messages"]
    assert request_messages[0]["role"] == "system"
    assert EPHEMERAL_TEXT in request_messages[0]["content"]
    user_rows = [m for m in request_messages if m.get("role") == "user"]
    assert user_rows
    assert all(EPHEMERAL_TEXT not in str(row["content"]) for row in user_rows)
    assert PERSISTENT_TEXT in str(user_rows[-1]["content"])

    session = loop.sessions.get_or_create("cli:c1")
    dumped = json.dumps(session.messages)
    assert EPHEMERAL_TEXT not in dumped
    assert PERSISTENT_TEXT in dumped
    user_row = next(m for m in session.messages if m.get("role") == "user")
    assert RUNTIME_CONTEXT_HISTORY_META in user_row


@pytest.mark.asyncio
async def test_ephemeral_block_never_enters_staged_provider_state(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, [
        RuntimeContextBlock(source="voice", content=EPHEMERAL_TEXT, ephemeral=True),
    ])
    loop.provider.can_resume_conversation_state.return_value = True
    session = loop.sessions.get_or_create("cli:c2")
    session.provider_state = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="test-model",
        version=1,
        payload={"items": []},
    )
    loop.sessions.save(session)

    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c2", content="hello")
    await loop._process_message(msg)

    provider_context = loop._captured["provider_context"]
    assert provider_context is not None
    state = provider_context.conversation_state
    assert state is not None
    assert state.pending_messages
    assert EPHEMERAL_TEXT not in json.dumps(state.pending_messages)
    assert EPHEMERAL_TEXT in loop._captured["messages"][0]["content"]


def test_build_request_kwargs_applies_spec_rider_without_mutating_caller_messages() -> None:
    provider = MagicMock()
    messages = [
        {"role": "system", "content": "base system prompt"},
        {"role": "user", "content": "hello"},
    ]
    spec = make_run_spec(
        provider,
        model="test-model",
        initial_messages=messages,
        tools=MagicMock(),
        max_iterations=3,
        max_tool_result_chars=20_000,
        ephemeral_runtime_context=EPHEMERAL_TEXT,
    )

    kwargs = AgentRunner()._build_request_kwargs(spec, messages, tools=None)

    assert kwargs["messages"][0] == {
        "role": "system",
        "content": f"base system prompt\n\n{EPHEMERAL_TEXT}",
    }
    assert kwargs["messages"][1] is messages[1]
    assert messages[0]["content"] == "base system prompt"
