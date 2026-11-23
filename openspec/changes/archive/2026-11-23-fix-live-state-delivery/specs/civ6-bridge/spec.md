# civ6-bridge -- deltas

## ADDED Requirements

### Requirement: The log-tailing transport hands out every newest state once

`LogTailer.wait_for_state` SHALL return the newest state that has not been handed
out yet, without waiting when such a state has already been read, and MUST NOT
return the same state twice. A state read in the same batch of log lines as the
ack before it MUST still be returned to the next caller. States replaced by a
newer one before being read SHALL be counted.

#### Scenario: Ack and state arrive together

- **WHEN** an ack line and the state line behind it are read in one batch, and the caller waits for the ack and then for a state
- **THEN** the state is returned at once, not after a timeout

#### Scenario: No newer state

- **WHEN** a state has been handed out and no newer one has been read
- **THEN** waiting for a state ends in a timeout, not in the same state again

### Requirement: A start in mid-turn can see the state already written

At start the tailer SHALL keep the last complete state in the existing log and
make it available only when a heartbeat read after start shows the mod is
listening; a state read after start replaces it. Acks and heartbeats written
before start MUST NOT be used.

#### Scenario: Started while the mod waits for a command

- **WHEN** the log already holds a complete state and a heartbeat is read after start
- **THEN** that state is handed out as the current one

#### Scenario: Old heartbeat only

- **WHEN** the log holds a state and a heartbeat written before start, and nothing is written after start
- **THEN** no state is handed out and the heartbeat check stays false
