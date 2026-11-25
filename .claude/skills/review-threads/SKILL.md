---
name: review-threads
description: Work through the review threads on a pull request -- read every thread, verify each finding against the code, fix a real one with a test that fails before the fix, reply with the commit and the test, and resolve; reply with what was traced when a finding is wrong. At most two review rounds; security findings stay open for the user; never approve or merge.
allowed-tools: Read, Edit, Write, Bash
---

# Review Threads Skill

A review thread is a claim about the code, not an instruction. This skill turns each
one into either a fix with a test that proves it, or a reply that shows why the claim
does not hold.

## Why this exists

Automated reviews (Copilot, `sdlc-*` role reviews) have found real defects here,
several of them in code that passed its author's checks (CLAUDE.md section 5.2 records
them). They have also been wrong. Accepting a finding without reading the code ships a
"fix" for a bug that is not there; dismissing one without reading it leaves a real bug
open. Both are cheaper to avoid than to undo.

## Steps

### 1. Read every thread

Read the threads with the GitHub MCP tool `pull_request_read`, method
`get_review_comments`: every thread, resolved and unresolved, with its file, line, the
commit it was written against and its replies. Do not reach for the GitHub GraphQL API
through `gh api`, or for a `gh api` call with a method or body field: the PreToolUse
guard denies both (they can write), and so do the deny rules in
`.claude/settings.json`. Read each thread in full. Do not start fixing until you have
read them all: two threads often describe one defect, and one fix can make another
thread stale.

### 2. Verify each finding against the code

For each thread, open the file at the line it names, on the PR's current head, and
trace the claim through the code: the call sites, the inputs that reach it, the test
that covers it. Decide one of three:

- **real**: the code does what the finding says, and that is wrong;
- **wrong**: the code does not do that, or what it does is intended and documented;
- **security**: the finding concerns secrets, credentials, permissions, a guard or
  hook, or anything that could fail open. Go to step 5 whatever you conclude.

### 3. A real finding: a failing test, then the fix

1. Write a test that **fails before** the fix and fails for the reason the finding
   gives. Run it and see it fail.
2. Make the fix. Run the test and see it pass, then run the suites that cover the
   module (`make verify-fast` at least).
3. If the test passed before the fix, it does not test the finding: rewrite it. The
   `negative-control` skill describes the check.
4. Commit on the PR's branch (never on `dev`, `qa` or `main`).

### 4. Reply, then resolve

Reply in the thread with the GitHub MCP tool `add_reply_to_pull_request_comment`,
before you resolve it:

- **real**: "Confirmed and fixed in `<short sha>`." Then what changed, in a sentence
  or two, and the test that would have caught it, by path and name.
- **wrong**: what you traced and why the finding does not hold: the file and line, the
  input that reaches it, the test that already covers it. No fix, no commit.

End the reply with the footer that the session's harness instructions set for GitHub
comments, when they set one. The footer comes from those instructions, not from this
skill.

Then resolve the thread with the GitHub MCP tool `resolve_review_thread`. A thread on
a line the fix moved is still answered, not silently left to go outdated.

**When no tool can resolve a thread** (the MCP tool is not connected, or it refuses):
reply all the same, leave the thread open, and list it in your report to the user as
answered but not resolved. Do not fall back to the GraphQL API or to a write-capable
`gh api` call.

### 5. Security findings stay open for the user

Reply with what you found and, if you fixed it, the commit and the test, but **do
not resolve** a security thread: leave it open for the user to read and resolve.
Say so in your report.

### 6. At most two review rounds

A round is one batch of findings and the replies to it. After two rounds on one PR,
stop: report the threads still open and what each would need, and leave the next step
to the user. A third round is a sign the change needs a design decision, not another
patch.

### 7. Never approve or merge

Do not approve the PR, submit a review on it, request a re-review in its place, or
merge it. Merging into `qa` or `main` is maintainer-only (CLAUDE.md section 3.2); into
`dev` it follows normal review. Your output is commits, replies and a report.

## Report

For each thread: its link, real / wrong / security, the commit and test (or what was
traced), and whether it was resolved. List the threads left open and why.
