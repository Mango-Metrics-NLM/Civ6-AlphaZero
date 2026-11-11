# agent-harness — deltas

## MODIFIED Requirements

### Requirement: Prohibited-action guard

The harness SHALL block prohibited shell actions before they execute, via a
``PreToolUse`` hook (``.claude/hooks/pretooluse_guard.sh``) registered for both
the ``Bash`` and ``PowerShell`` tools that emits a ``permissionDecision`` of
``deny``. The guard MUST cover commits or pushes while checked out on a
protected branch, pushes whose destination is a protected branch in any refspec
form, deletion of a protected branch, tag pushes (``refs/tags/*``, ``--tags``,
``--follow-tags``), ``--all``/``--mirror`` pushes, ``git tag`` with any
non-listing argument, ``gh pr merge``, ``gh release``, ``gh repo edit``,
``gh ruleset``, ``gh api`` with a method or body field or the ``graphql``
endpoint, runs exceeding 1000 games without ``--approval_ref``, and force-pushes
(``--force`` / ``-f`` / a ``+``-refspec) in any argument position, while still
allowing ``--force-with-lease``. Protected branches MUST be the union of
``dev qa main``, the default branch and ``CIV6AZ_PROTECTED_BRANCHES``. Wrappers
(``pwsh -c``, ``powershell -Command``, ``cmd /c``, ``bash -c``, ``iex``, ``&``)
MUST be unwrapped before matching, and a payload no parser can read MUST be
denied.

#### Scenario: A force-push is denied

- **WHEN** a Bash tool call runs ``git push origin main --force``
- **THEN** the guard emits a deny decision

#### Scenario: A lease-guarded force-push is allowed

- **WHEN** a Bash tool call runs ``git push --force-with-lease origin feature``
- **THEN** the guard does not deny it

#### Scenario: A wrapped push to qa is denied

- **WHEN** a tool call runs ``pwsh -c "git push origin HEAD:qa"``
- **THEN** the guard emits a deny decision

#### Scenario: The old default stays protected after the default moves

- **WHEN** ``CIV6AZ_DEFAULT_BRANCH`` is ``dev`` and ``CIV6AZ_PROTECTED_BRANCHES``
  lists ``claude/civ6-alphazero-implementation-vMnun``
- **THEN** a commit while checked out on that branch is denied

#### Scenario: Tag listing is allowed and tag creation is denied

- **WHEN** a tool call runs ``git tag -l 'v2.1.0-rc.*'`` or ``git fetch --tags origin``
- **THEN** the guard does not deny it
- **WHEN** a tool call runs ``git tag v1``
- **THEN** the guard emits a deny decision

#### Scenario: A read-only gh api call is allowed and a write is denied

- **WHEN** a tool call runs ``gh api repos/o/r/pulls``
- **THEN** the guard does not deny it
- **WHEN** a tool call runs ``gh api repos/o/r/git/refs -f ref=refs/heads/x``
- **THEN** the guard emits a deny decision

### Requirement: Permission governance

The harness SHALL declare an explicit allow/deny permission model in
``.claude/settings.json``. The deny list MUST refuse edits to ``.env`` files so
that secret material is not written through the file-editing tools, and MUST
refuse ``gh pr merge``, ``gh release``, ``gh repo edit``, ``gh ruleset``,
``gh api`` with ``-X``/``--method``, and tag pushes for both the ``Bash`` and
``PowerShell`` tools. Project MCP servers MUST NOT be enabled wholesale
(``enableAllProjectMcpServers: false``).

#### Scenario: Editing .env is denied

- **WHEN** an ``Edit`` or ``Write`` tool call targets ``.env``
- **THEN** the operation is refused before execution

#### Scenario: Merging a PR is refused by rule

- **WHEN** a Bash or PowerShell tool call runs ``gh pr merge 1``
- **THEN** a ``permissions.deny`` rule refuses it, independently of the guard

## ADDED Requirements

### Requirement: Unverifiable runs ask for confirmation

The prohibited-action guard SHALL return a ``permissionDecision`` of ``ask``
(prompting the user) rather than allowing or denying a command that carries
``--approval_ref``, passes ``--preset production``, passes ``--config`` to a game
runner, or invokes a game runner that is not on the list of runners that price
themselves. A deny anywhere in the same command MUST take precedence over an ask.

#### Scenario: An approved over-cap run asks

- **WHEN** a tool call runs ``python -m alpha_zero.training_civ6 --num_games 5000 --approval_ref 123``
- **THEN** the guard emits an ask decision whose reason names ``--approval_ref``

#### Scenario: An unapproved over-cap run is denied

- **WHEN** a tool call runs ``python -m alpha_zero.training_civ6 --num_games 5000``
- **THEN** the guard emits a deny decision

#### Scenario: A priced dry run passes silently

- **WHEN** a tool call runs ``python -m alpha_zero.training_civ6 --dry_run``
- **THEN** the guard emits no decision
