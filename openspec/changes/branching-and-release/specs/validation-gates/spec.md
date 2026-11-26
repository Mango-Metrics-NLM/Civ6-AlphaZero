# validation-gates — deltas

## ADDED Requirements

### Requirement: Gating workflows record every protected-branch state

The ``ci``, ``codeql``, ``security-scan`` and ``scorecard`` workflows SHALL run on
push to every protected branch (``dev``, ``qa``, ``main`` and any branch in
``CIV6AZ_PROTECTED_BRANCHES``), and a newer push MUST NOT cancel an in-progress
run on a protected branch. The workflow-pinning regression suite MUST read the
protected list from the same source as the agent guard and fail when either
property is broken.

#### Scenario: A workflow drops a protected branch from its push trigger

- **WHEN** ``scorecard.yml`` no longer lists ``qa`` under ``on.push.branches``
- **THEN** ``test_gating_workflows_run_on_every_protected_branch`` fails

#### Scenario: A workflow cancels runs on a protected branch

- **WHEN** ``codeql.yml`` sets ``cancel-in-progress: true`` while pushing on ``dev``
- **THEN** ``test_runs_on_protected_branches_are_never_cancelled`` fails

#### Scenario: A push to qa is ratcheted against qa

- **WHEN** the coverage-floor ratchet runs on a push event with ``GITHUB_REF_NAME=qa``
- **THEN** its first base candidate is ``origin/qa``

### Requirement: Branch Direction gates protected merges

A pull request into ``dev``, ``qa`` or ``main`` SHALL be judged by
``scripts/check_branch_direction.py`` (job ``Branch Direction``) using base and
head read from the API: ``dev`` refuses ``main``, ``qa``, ``hotfix/*``,
``evidence/*`` and ``rcfix/*``; ``qa`` accepts only ``dev`` with ``origin/main``
as an ancestor, ``evidence/<rc tag>`` changing only ``docs/evidence/**``,
patch-equivalent ``rcfix/*`` and ``backmerge/*`` carrying only released commits;
``main`` accepts only ``qa`` and ``hotfix/*``. The head MUST be in the same
repository, path filters MUST NOT apply rename detection, and any API or git failure MUST
fail the check. The check is an accident net: the control for ``qa`` and ``main`` is that
only the maintainer can merge into them.

#### Scenario: Promotion after a squashed back-merge

- **WHEN** ``dev`` is proposed into ``qa`` and ``origin/main`` is not an ancestor of ``dev``
- **THEN** Branch Direction fails and reports the squashed or missing back-merge

#### Scenario: A hand-edited rc fix

- **WHEN** an ``rcfix/*`` commit is not patch-equivalent to a commit on ``dev``
- **THEN** Branch Direction fails

### Requirement: Required checks are real, blocking PR jobs

Every status-check context named in ``.github/rulesets/*.json`` SHALL be the
name of a job (matrix legs expanded) in a workflow that runs on every
``pull_request`` into ``dev``, ``qa`` and ``main``, without a path filter, and
that job MUST NOT be ``continue-on-error``.

#### Scenario: A required job is renamed

- **WHEN** the ``Security Scan`` job in ``ci.yml`` is renamed
- **THEN** ``test_every_required_check_is_a_real_blocking_pr_job`` fails

### Requirement: A release publishes only after its gates

``release.yml`` SHALL publish only from a job that needs ``verify-tag`` (version,
ancestry, final-tag rule) and ``wait-for-ci`` (push checks on the tagged
commit; ``Branch Direction``, and ``Release Evidence`` for a non-hotfix final,
on the head of the one merged PR whose merge commit is the tagged commit).
``scripts/maintainer/tag_release.py`` MUST apply the same rules before creating
a tag and MUST refuse a commit whose release workflow lacks ``wait-for-ci``.

#### Scenario: A final that changes code after its rc

- **WHEN** ``git diff <highest rc>..<final>`` touches a path outside ``docs/evidence/``
- **THEN** ``tag_release`` refuses and ``verify-tag`` fails

### Requirement: A release PR into main carries verified evidence

A ``Release Evidence`` job SHALL run on every open pull request (``opened``,
``synchronize``, ``reopened``, ``edited``), with no path filter and no
``continue-on-error``, in a workflow with no ``push`` trigger, and SHALL be a
required check of the ``main`` ruleset once ``main`` carries it. No
push-triggered workflow MAY define a job of that name. It MUST report n/a (exit
0) unless the base is ``main`` and the head is not ``hotfix/*``, reading the refs
from ``env``, never from an expression inside ``run:``. Otherwise
``scripts/verify_release_evidence.py`` MUST verify
``docs/evidence/release_vX_Y_Z/bundle.json`` (pyproject's version) and exit 0
(pass), 1 (fail) or 2 (missing): each attempt's ``run.json`` invocations equal
to section 7.1's registered commands at the attempt's commit; every component
re-hashed; each arm's counts,
``scoring.summarize`` and both intervals recomputed from its per-game records;
the verdict recomputed by ``a1_protocol.compute_verdict`` in section 7.1 order;
the rc commit's check runs read as ``wait-for-ci`` reads them; each component
run at most 1,000 games or carrying its own ``approval_ref``; ``rc_tag`` the
highest rc of the release; every attempt's ``git_sha`` an ancestor of
``rc_sha``, and for the judged attempt no change on
``scripts/release_play_path.txt`` from it to ``rc_sha`` or to the release head;
the protocol hash equal to section 7.1's; exactly one non-VOID attempt per role,
which is judged; ``learning_recorded`` true and ``learning_claimed`` true only
when the judged A1 and A1-R both PASS.

#### Scenario: A VOID records no learning result

- **WHEN** the bundle's only A1 attempt recomputes to VOID, recorded honestly
- **THEN** the verifier exits 1, and passes once a non-VOID re-run under a new ``card_id`` is listed beside it

#### Scenario: A negative A1 ships with the claim withheld

- **WHEN** the bundle's A1 verdict recomputes to FAIL and ``learning_claimed`` is false
- **THEN** the verifier exits 0

#### Scenario: A stored verdict that the records do not support

- **WHEN** the bundle stores PASS for an A1 whose primary arm recomputes to FAIL
- **THEN** the verifier exits 1

#### Scenario: The play path changed after the run

- **WHEN** ``git diff <run git_sha> <rc_sha>`` touches ``alpha_zero/core/mcts/``, or a later rc does while the bundle names the earlier one
- **THEN** the verifier exits 1
