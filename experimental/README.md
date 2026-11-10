# experimental/

Material that is **experimental, ungated and not part of the `alpha_zero` package**.

- Nothing here is built, linted, type-checked or tested by CI.
- Nothing here is installed by `pip install -e .` (the setuptools `exclude` list
  and ruff's `extend-exclude` both name this directory).
- It is kept for reference only; expect it to be incomplete or out of date.

| Directory | What it is |
|-----------|------------|
| `spacetimedb/server-module/` | Early SpacetimeDB server-module sketch (Rust + a Python AI service stub). |
| `spacetimedb/spacetimedb_module/` | SpacetimeDB reducers/tables for distributed coordination (Rust, never deployed). |

The maintained SpacetimeDB client code lives in `alpha_zero/spacetimedb/`.
