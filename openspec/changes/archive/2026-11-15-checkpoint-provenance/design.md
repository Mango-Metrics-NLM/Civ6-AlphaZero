# Design notes -- checkpoint-provenance

## Primitive-only, validated at save

Metadata lives beside `config` and `state_dict` and must round-trip under
`weights_only=True`. `validate_primitive_metadata` raises `TypeError` at save
rather than letting a richer value push every later load onto the unsafe path
(CLAUDE.md §5.2, "Failed safe-load must not silently downgrade"). `policy_target`
is a string; `git_sha` is a string or `None`.

## git_sha is resolved once per run

`read_git_sha` reads `.git` directly (no subprocess); `train` calls it once and
reuses the dict for every save. The benchmark envelope shares the function.

## Only the horizon is a hard error

`max_turns` is the one field whose mismatch is silent. A `board_size` mismatch
already fails in `load_state_dict`; other fields describe how the data was made
and are provenance, not compatibility. A reader that supplies no horizon checks nothing.

## checkpoint_0 and the resume convention

`checkpoint_<N>.pt` is written after loop index N-1 and resumes at index N, so
`checkpoint_0.pt` means "from the top". It is written only when not resuming and
`start_iteration == 0`; otherwise a resume would overwrite the file it just read,
or put trained weights under the random-init name.

## ENV_VERSION

Starts at 2: version 1 is the pre-LF-B1 mask. No file carries version 1 (none
carried metadata), but the constant now distinguishes future rule changes from the mask fix.
