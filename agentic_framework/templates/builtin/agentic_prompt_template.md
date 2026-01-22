# Agentic Coding System Prompt Template

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

**Template ID:** ${template_id}
**Generated:** ${generated_at}
**Project:** ${project_name}

-----

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: ${system_intent}
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
{%for criterion in success_criteria%}
- [ ] ${criterion}
{%endfor%}
```

### 1.3 Problem Description (The "Three Paragraphs")

```
${problem_statement:Describe the core problem and coordination logic here.}

${data_flows:Describe key data flows and state synchronization here.}

${failure_modes:Describe failure modes and invariants here.}
```

-----

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: ${language_runtime:Python 3.10+}
- Required Dependencies: ${required_dependencies:[]}
- Security: No hardcoded secrets, all inputs validated
- Compatibility: ${compatibility_requirements:[]}
{%for constraint in hard_constraints%}
- ${constraint.name}: ${constraint.value}
{%endfor%}
```

### 2.2 Soft Constraints (Preferences)

```
- Style: ${style_guide:Google Style Guide}
- Architecture: ${architecture_preference:Prefer composition over inheritance}
- Performance: ${performance_preference:Prefer async I/O, minimize allocations}
- Testing: ${testing_preference:pytest, >80% coverage on core logic}
{%for constraint in soft_constraints%}
- ${constraint.name}: ${constraint.value}
{%endfor%}
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure existing file organization
- Add dependencies not explicitly listed
- Refactor adjacent code for consistency
- Choose implementation patterns not specified
{%for permission in anti_constraints%}
- ${permission.description}
{%endfor%}
```

-----

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
{%for path in in_scope_paths%}
- ${path}
{%endfor%}

OUT OF SCOPE:
{%for path in out_of_scope_paths%}
- ${path}
{%endfor%}
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within scope
- Dependency installation
- Running tests
- Refactoring for consistency
{%for action in autonomous_actions%}
- ${action}
{%endfor%}

CONFIRM FIRST (ask before proceeding):
- Architectural changes affecting >3 modules
- Breaking API changes
- Deletions of >100 lines
{%for action in confirm_first_actions%}
- ${action}
{%endfor%}

PROHIBITED (do not attempt):
- Commits to main branch
- External API calls with side effects
- Modifications outside scope
{%for action in prohibited_actions%}
- ${action}
{%endfor%}
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: ${max_iterations:5}
- Max files to modify in single pass: ${max_files_per_pass:20}
- Time-boxed exploration: ≤${exploration_timeout_minutes:10} min on research before asking
```

-----

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:
1. [Lint]: `${lint_command:ruff check .}`
2. [Type check]: `${type_check_command:mypy src/}`
3. [Unit tests]: `${test_command:pytest tests/unit -v}`
4. [Integration tests]: `${integration_test_command:pytest tests/integration -v}`
{%for cmd in verification_commands%}
5. [${cmd.name}]: `${cmd.command}`
{%endfor%}
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
  → Fix automatically, re-run

ON TYPE ERROR:
  → Analyze error, fix types, re-run

ON TEST FAILURE:
  → Read failure output
  → Identify root cause
  → Fix implementation (not test, unless test is wrong)
  → Re-run

ON REPEATED FAILURE (same error 3x):
  → Stop and report analysis
  → Request human guidance
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass
2. Manual smoke test: `${smoke_test_command:}`
3. Generate brief summary of changes made
```

-----

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory (${memory_file:CLAUDE.md})

```markdown
Maintain/update the following in project root:

# ${memory_file:CLAUDE.md}
## Build Commands
- [command]: [purpose]

## Test Commands
- [command]: [purpose]

## Architecture Decisions
- [date]: [decision]: [rationale]

## Known Issues
- [issue]: [workaround]
```

### 5.2 Information to Preserve Across Sessions

```
- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale
- Gotchas discovered during implementation
```

### 5.3 Information That Can Be Re-derived

```
- File structure (can be scanned)
- Dependency versions (in lockfile)
- Current test status (can re-run)
```

-----

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```
1. Read ${memory_file:CLAUDE.md} if exists
2. Scan project structure: `${scan_command:find . -type f -name "*.py" | head -50}`
3. Identify entry points and test commands
4. Run existing tests to establish baseline
```

### 6.2 Implementation Order

```
1. Understand existing patterns (read before write)
2. Implement core logic (smallest working version)
3. Add error handling
4. Write tests
5. Run verification loop
6. Refactor if needed
7. Update ${memory_file:CLAUDE.md}
```

### 6.3 Completion Checklist

```
□ All success criteria met
□ All verification commands pass
□ ${memory_file:CLAUDE.md} updated with new commands/decisions
□ Summary of changes provided
□ Known limitations documented
```

-----

## SECTION 7: SUB-AGENT CONFIGURATION

### Available Sub-Agents

| Agent | Use For | Enabled |
|-------|---------|---------|
| **Explore** | Finding files, searching code, understanding codebase | ✓ |
| **Plan** | Designing strategies, identifying critical files | ✓ |
| **Bash** | Git operations, running commands | ✓ |
| **General Purpose** | Complex multi-step tasks | ✓ |
{%for agent in sub_agents%}
| **${agent.name}** | ${agent.description} | ${agent.enabled:✓} |
{%endfor%}

### Sub-Agent Usage Guidelines

```
When to use Explore agent:
- Open-ended codebase exploration
- Finding files by pattern (e.g., "src/**/*.tsx")
- Searching for keywords across multiple files
- Questions like "how does X work?"

When to use Plan agent:
- Complex tasks requiring architectural decisions
- Multi-step implementations
- Trade-off analysis

When to use Bash agent:
- Git operations
- Running shell commands
- System operations

When NOT to use sub-agents:
- Reading a specific known file (use Read directly)
- Simple grep for a single term (use Grep directly)
- Tasks that can be done in one step
```

-----

# USAGE NOTES

1. **Don't over-specify implementation**: If you find yourself writing pseudocode in the problem description, you're constraining too much. Describe the *what* and *why*, not the *how*.

2. **Make success criteria executable**: "Works correctly" is not verifiable. "All tests pass" is. "Handles edge cases" is not verifiable. "Returns 400 on malformed input" is.

3. **Anti-constraints matter**: Explicitly stating what the agent CAN do is as important as stating what it can't. Unlisted permissions default to "ask first" which slows everything down.

4. **Calibrate autonomy to trust**: Start with more confirmations, expand autonomy as you verify the agent's judgment matches yours.

5. **The feedback loop is the objective function**: If you can't specify how to verify success, you can't prompt for it effectively. Write the test specification before the implementation prompt.

-----

*Template Version: 1.0.0*
*Generated with Agentic Framework*
