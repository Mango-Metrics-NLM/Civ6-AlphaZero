---
name: type-fortification-skill
description: >
  Guides the 'Type-Driven Fortification' methodology for Python codebases.
  Covers strict mypy compliance (mypy --strict), incremental cache recovery,
  logic preservation, and exhaustive unit test verification (85%+ coverage)
  to ensure reliability during major architectural migrations.
allowed-tools: Read, Edit, Bash
---

# Type-Driven Fortification Skill

## Purpose

Use this skill when performing massive structural changes, architectural migrations, or resolving legacy technical debt, particularly when the user requests "100% mypy stabilization" or "type-driven fortification."

The goal is to use the type checker (`mypy --strict`) as a compiler to safely guide refactoring, ensuring zero regressions in domain logic while achieving bulletproof type safety.

## Core Methodology

### 1. Incremental Cache Recovery

When changing core interfaces, `mypy`'s daemon or incremental cache can become corrupted or report stale errors. 
- If you see phantom errors that should have been fixed, run: `mypy --strict --no-incremental path/to/module`
- Clear the `.mypy_cache` directory if issues persist.

### 2. Logic Preservation

Type annotations must *never* change the runtime behavior of the code unless explicitly fixing a discovered bug.
- Use `typing.cast` sparingly and only when the type system cannot express a known runtime invariant.
- Prefer explicit `# type: ignore[error-code]` with a comment over unsafe `Any` proliferation if refactoring is impossible.
- Ensure all dictionary accesses, JSON parsing, and external API boundaries are strongly typed using `TypedDict` or Pydantic where applicable.

### 3. Exhaustive Unit Test Verification

Type safety is only half the battle. You must prove the logic holds via tests.
- Maintain at least 85% line coverage, the repository gate
  (`pytest --cov=your_module --cov-fail-under=85`).
- Ensure all unit tests pass after type fortification.
- If type changes require test changes, modify the tests carefully without altering their fundamental assertions.

## Execution Steps for Stabilization

1. **Baseline**: Run `make typecheck` or `mypy --strict target_dir/` to get a baseline error count.
2. **Iterative Fixes**:
   - Fix `Any` types, missing returns, and unannotated arguments.
   - Introduce `Protocol`s for duck-typing where concrete dependencies cause circular imports.
   - Use `TypeVar` and `Generic` for generic collections or builders.
3. **Validation**: Re-run `mypy --strict --no-incremental`.
4. **Testing**: Run `make test-all` or `pytest --cov=target_dir/` to ensure no runtime breakages.
5. **Coverage Check**: Assert the coverage gate has not been breached.

## Anti-Patterns to Avoid

- ❌ **`Any` propagation**: Do not silence `mypy` by adding `Any`.
- ❌ **Blind `type: ignore`**: Do not add `# type: ignore` without an error code (e.g., use `# type: ignore[attr-defined]`).
- ❌ **Runtime Type Checking overhead**: Avoid `isinstance` checks inside tight loops (e.g., MCTS rollouts) just to satisfy `mypy` if a `cast` or generic is faster and mathematically safe.
