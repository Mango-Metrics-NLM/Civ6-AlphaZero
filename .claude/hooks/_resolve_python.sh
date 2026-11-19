#!/usr/bin/env bash
# Shared Python >= 3.11 interpreter discovery for `.claude/hooks/*.sh`.
# Resolves `$VIRTUAL_ENV`, real `python3`/`python` on PATH (skipping Windows Store
# `WindowsApps` stubs), or `${CLAUDE_PROJECT_DIR:-.}/.venv` regardless of cwd.

resolve_python() {
  local project_dir="${1:-${CLAUDE_PROJECT_DIR:-.}}"
  local cand
  for cand in \
    "${VIRTUAL_ENV:-}/Scripts/python.exe" \
    "${VIRTUAL_ENV:-}/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)" \
    "${project_dir}/.venv/Scripts/python.exe" \
    "${project_dir}/.venv/bin/python"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    case "$cand" in *WindowsApps*) continue ;; esac
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}
