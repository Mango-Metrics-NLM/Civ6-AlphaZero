# Design notes -- fix-move-mask

## The mask is exactly the executor's predicate

`_is_valid_move` rejects off-board, mountain and any occupied tile. The mask now
applies the same three conditions. The occupancy rule covers own and enemy units
because the executor scans `self.units` without regard to owner. Cities do not
block movement in the executor, so they do not block it in the mask.

## Other rejections a mask can see

Audited against `_execute_action`: attack (needs an enemy at the target) and produce (city has no queue)
match their mask branches. Skip and found-city (settler, no city within
`MIN_CITY_DISTANCE`) are stricter in the mask: their bits sit inside the
`moves_remaining > 0` block, while the executor has no moves check for either.
That is the safe direction (the mask never offers a rejected action; it only
withholds some the executor would accept) and is left as is. Only move offered
actions the executor rejects.

## Fog of war

The mask builder receives `_get_game_state()` (true state), also under fog;
`get_observation` passes the unmasked state deliberately. This change adds no
information path. A fogged mask therefore reveals hidden blockers as illegal
moves, an existing leak recorded as LF-B6 and left for its own change.

## Verification

The oracle is `info["action_success"]`, the executor's own signal, not log text.
`test_mask_matches_executor_for_every_move_action` clones the env and steps every
move action at each state of a seeded game, so mask and executor are compared
action by action.
