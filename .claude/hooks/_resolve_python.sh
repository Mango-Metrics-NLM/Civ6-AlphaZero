#!/usr/bin/env bash
# Shared Python >= 3.11 interpreter discovery for `.claude/hooks/*.sh`.
# Search order: active `$VIRTUAL_ENV`, all non-WindowsApps `python3`/`python` hits on PATH,
# then `${CLAUDE_PROJECT_DIR:-.}/.venv` regardless of cwd.

_resolve_python_is_valid() {
  local cand="$1"
  [ -n "$cand" ] && [ -x "$cand" ] || return 1
  case "$cand" in *[Ww][Ii][Nn][Dd][Oo][Ww][Ss][Aa][Pp][Pp][Ss]*) return 1 ;; esac
  "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null
}

resolve_python() {
  local project_dir="${1:-${CLAUDE_PROJECT_DIR:-.}}"
  local cand path_hits
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    for cand in \
      "${VIRTUAL_ENV}/Scripts/python.exe" \
      "${VIRTUAL_ENV}/bin/python"; do
      if _resolve_python_is_valid "$cand"; then
        printf '%s\n' "$cand"
        return 0
      fi
    done
  fi
  path_hits="$(type -aP python3 python 2>/dev/null || true)"
  if [ -n "$path_hits" ]; then
    while IFS= read -r cand; do
      if _resolve_python_is_valid "$cand"; then
        printf '%s\n' "$cand"
        return 0
      fi
    done <<EOF
$path_hits
EOF
  fi
  if [ -n "$project_dir" ]; then
    for cand in \
      "${project_dir}/.venv/Scripts/python.exe" \
      "${project_dir}/.venv/bin/python"; do
      if _resolve_python_is_valid "$cand"; then
        printf '%s\n' "$cand"
        return 0
      fi
    done
  fi
  return 1
}
