"""Coverage tests for alpha_zero.rag.agents.workflow.

Targets uncovered paths including:
- WorkflowStep: duration_ms, success properties
- WorkflowContext: add_step, get_current_step, to_dict
- RAGAgentWorkflow: initialize, close, register_hook, _run_hooks, execute,
  _step_enhance_query, _step_generate_embedding, _step_retrieve_documents,
  _step_build_context, add_documents, query, async context manager
- WorkflowResult dataclass
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from alpha_zero.rag.agents.workflow import (
    RAGAgentWorkflow,
    WorkflowContext,
    WorkflowResult,
    WorkflowState,
    WorkflowStep,
)
from alpha_zero.rag.stores.base import Document, DocumentMetadata, SearchResult
from alpha_zero.utils.compat import UTC

# ---------------------------------------------------------------------------
# WorkflowStep
# ---------------------------------------------------------------------------


class TestWorkflowStep:
    def test_duration_ms_none_when_incomplete(self):
        step = WorkflowStep(name="test")
        assert step.duration_ms is None

    def test_duration_ms_calculated(self):
        step = WorkflowStep(name="test")
        step.started_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        step.completed_at = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
        assert step.duration_ms == pytest.approx(1000.0)

    def test_success_true(self):
        step = WorkflowStep(name="test")
        step.completed_at = datetime.now(UTC)
        assert step.success is True

    def test_success_false_with_error(self):
        step = WorkflowStep(name="test")
        step.completed_at = datetime.now(UTC)
        step.error = "something went wrong"
        assert step.success is False

    def test_success_false_not_completed(self):
        step = WorkflowStep(name="test")
        assert step.success is False


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------


class TestWorkflowContext:
    def test_defaults(self):
        ctx = WorkflowContext()
        assert ctx.state == WorkflowState.IDLE
        assert ctx.query == ""
        assert len(ctx.steps) == 0

    def test_add_step(self):
        ctx = WorkflowContext()
        step = WorkflowStep(name="step1")
        ctx.add_step(step)
        assert len(ctx.steps) == 1

    def test_get_current_step_empty(self):
        ctx = WorkflowContext()
        assert ctx.get_current_step() is None

    def test_get_current_step(self):
        ctx = WorkflowContext()
        ctx.add_step(WorkflowStep(name="s1"))
        ctx.add_step(WorkflowStep(name="s2"))
        assert ctx.get_current_step().name == "s2"

    def test_to_dict(self):
        ctx = WorkflowContext(query="test query")
        ctx.add_step(WorkflowStep(name="s1"))
        d = ctx.to_dict()
        assert d["query"] == "test query"
        assert d["num_steps"] == 1
        assert d["state"] == "idle"
        assert "workflow_id" in d
        assert "created_at" in d


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def test_defaults(self):
        ctx = WorkflowContext()
        result = WorkflowResult(context=ctx)
        assert result.success is True
        assert result.response == ""

    def test_failed_result(self):
        ctx = WorkflowContext()
        result = WorkflowResult(
            context=ctx,
            success=False,
            error="failed",
            total_time_ms=100.0,
        )
        assert result.success is False
        assert result.error == "failed"


# ---------------------------------------------------------------------------
# RAGAgentWorkflow
# ---------------------------------------------------------------------------


def _make_mock_store():
    store = AsyncMock()
    store.is_initialized = True
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.add_documents = AsyncMock(return_value=["id1"])
    return store


def _make_mock_embedder():
    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embedder.embed_documents = AsyncMock(side_effect=lambda docs: docs)
    return embedder


class TestRAGAgentWorkflowInit:
    def test_init(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        assert workflow._is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self):
        store = _make_mock_store()
        store.is_initialized = False
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        await workflow.initialize()
        assert workflow._is_initialized is True
        store.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        workflow._is_initialized = True
        await workflow.close()
        assert workflow._is_initialized is False
        store.close.assert_awaited_once()


class TestRAGAgentWorkflowHooks:
    def test_register_hook(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        cb = MagicMock()
        workflow.register_hook("pre_query", cb)
        assert cb in workflow._hooks["pre_query"]

    def test_register_hook_invalid_event(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        cb = MagicMock()
        workflow.register_hook("nonexistent_event", cb)
        # Should silently ignore

    @pytest.mark.asyncio
    async def test_run_hooks_sync(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        cb = MagicMock()
        workflow._hooks["pre_query"].append(cb)
        ctx = WorkflowContext()
        await workflow._run_hooks("pre_query", ctx)
        cb.assert_called_once_with(ctx)

    @pytest.mark.asyncio
    async def test_run_hooks_async(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        cb = AsyncMock()
        workflow._hooks["pre_query"].append(cb)
        ctx = WorkflowContext()
        await workflow._run_hooks("pre_query", ctx)
        cb.assert_awaited_once_with(ctx)


class TestRAGAgentWorkflowExecute:
    @pytest.mark.asyncio
    async def test_execute_not_initialized(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        with pytest.raises(RuntimeError, match="not initialized"):
            await workflow.execute("test query")

    @pytest.mark.asyncio
    async def test_execute_success_no_documents(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        workflow._is_initialized = True

        result = await workflow.execute("test query", top_k=3)
        assert result.success is True
        assert result.context.state == WorkflowState.COMPLETED
        assert result.total_time_ms > 0
        assert len(result.context.steps) == 4  # enhance, embed, retrieve, build

    @pytest.mark.asyncio
    async def test_execute_success_with_documents(self):
        store = _make_mock_store()
        doc = Document(content="Strategy guide")
        doc.metadata = DocumentMetadata(source="guide.txt")
        search_result = SearchResult(document=doc, score=0.95)
        store.search = AsyncMock(return_value=[search_result])

        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        workflow._is_initialized = True

        result = await workflow.execute("early game strategy", top_k=5)
        assert result.success is True
        assert len(result.sources) == 1
        assert "Strategy guide" in result.response

    @pytest.mark.asyncio
    async def test_execute_without_sources(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        workflow._is_initialized = True

        result = await workflow.execute("test", include_sources=False)
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        store = _make_mock_store()
        store.search = AsyncMock(side_effect=RuntimeError("DB error"))
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        workflow._is_initialized = True

        result = await workflow.execute("test query")
        assert result.success is False
        assert "DB error" in result.error
        assert result.context.state == WorkflowState.FAILED


class TestRAGAgentWorkflowDocuments:
    @pytest.mark.asyncio
    async def test_add_documents(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        workflow._is_initialized = True

        ids = await workflow.add_documents(
            texts=["doc1", "doc2"],
            sources=["src1", "src2"],
            metadata=[{"key": "v1"}, {"key": "v2"}],
        )
        assert ids == ["id1"]
        store.add_documents.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_documents_not_initialized(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        with pytest.raises(RuntimeError, match="not initialized"):
            await workflow.add_documents(texts=["doc"])

    @pytest.mark.asyncio
    async def test_query(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        workflow._is_initialized = True

        await workflow.query("test", top_k=3)
        embedder.embed_text.assert_awaited_once_with("test")
        store.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_query_not_initialized(self):
        store = _make_mock_store()
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)
        with pytest.raises(RuntimeError, match="not initialized"):
            await workflow.query("test")


class TestRAGAgentWorkflowContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        store = _make_mock_store()
        store.is_initialized = False
        embedder = _make_mock_embedder()
        workflow = RAGAgentWorkflow(store, embedder)

        async with workflow as w:
            assert w._is_initialized is True

        assert w._is_initialized is False
        store.close.assert_awaited_once()
