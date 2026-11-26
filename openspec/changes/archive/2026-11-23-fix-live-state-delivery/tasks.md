# Tasks

## 1. GM-B7c-1: the transport keeps every state
- [x] 1.1 Tests first: a real `LogTailer` on a real file, ack and state in one batch
- [x] 1.2 `log_tailer.py`: sequence numbers, hand-out without clear-before-wait, skipped-state count
- [x] 1.3 `log_tailer.py`: the state already in the log at start, released by a live heartbeat
- [x] 1.4 `log_tailer.py`: bounded unread acks
- [x] 1.5 `live_bridge.py`: `require_new`, `backlog_state()`
- [x] 1.6 `bridge/state_diff.py` and `scripts/live_command_check.py`
- [x] 1.7 Docs: `docs/LIVE_CIV6.md`, `CHANGELOG.md`, `CLAUDE.md`, gap register (LF-H1)

## 2. Verification
- [x] 2.1 Planted defects are caught (clear-before-wait restored; backlog handed out without a heartbeat)
- [ ] 2.2 Maintainer, on the real game: after a command the next state arrives without a 30-second wait
