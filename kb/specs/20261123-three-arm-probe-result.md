# The three-arm asymmetry probe: design, result and what may be claimed

Design, review and wording by sdlc-ml-scientist; test by sdlc-qa-sdet. Zero games; 2,430
searches; untrained networks. Test: `tests/unit/test_bridge_search_asymmetry_probe.py`
(the measurement is marked `slow`, so no CI job runs it; about 3.5 minutes with
`OMP_NUM_THREADS=2 python -m pytest tests/unit/test_bridge_search_asymmetry_probe.py -q -s`).

## Question

Since GM-B8b the acting player cannot win by elimination in the bridge's simulator and can
still lose by it. Does that make the search avoid contact with visible enemies?

## Design, fixed before the run

Three arms on the same roots, decision seeds and network initialisations: (a) the merged rule,
(b) the raw simulator rule, (c) a symmetric rule in which neither side's elimination ends the
game before the limit. 27 hand-built roots, 10 decision seeds, 3 initialisations. The verdict
classes, their thresholds and the controls were written in the test contract before any search
ran. The verdict is printed, never asserted.

## Result: vacuous

| Arm | Contact | Retreat | Attack | Mean root value | Own side eliminated in the tree |
|---|---|---|---|---|---|
| (a) merged | 0.083 | 0.583 | 0.075 | -0.014 | 0 |
| (b) raw simulator | 0.183 | 0.533 | 0.175 | +0.152 | 0 |
| (c) symmetric | 0.083 | 0.583 | 0.075 | -0.014 | 0 |

On none of the positions did any search line reach the loss of the player's last piece, so
the merged rule and the symmetric rule never behaved differently, and they chose the same
move in all 360 paired searches. The probe does not tell us whether losing-only endings make
the search avoid contact. It does show that the arms, their pairing and the controls work:
the raw simulator rule's win for destroying the last visible enemy was detected (root value
gap 0.32, attacks 126 against 54 on the control roots).

It was vacuous by construction, which is a flaw of the design: a full-health unit cannot be
killed inside a 24-simulation tree. The 100% agreement of (a) and (c) on the own-city roots
carries no information for the same reason and is not a passed check.

## Exploratory, not the registered comparison

On the same hand-built positions, removing the simulator's false win for destroying the last
visible enemy unit (the GM-B8b change) cut the share of searches that attacked or moved closer
from 18.3% to 8.3% (attacks 17.5% to 7.5%; mean root value +0.15 to -0.01). This is the
intended effect of the change, measured with an untrained network. It is not evidence about
whether losing-only endings make the search shy of contact, and it does not say whether 7.5%
is too few attacks.

## Why no follow-up probe is registered

Reading the code settles the question a follow-up would ask. The merged rule's loss fires only
when the simulator holds none of the player's pieces and no own unit was left out. Simulator
cities cannot be captured, so that state is reachable only when the player has no city and
every unit it owns is represented; losing every piece is then losing everything the player
has, and the loss is real. Whenever the loss would be false (something is left out), the
merged rule already takes it back. So there is no position where the two rules differ and the
merged rule's loss is false.

What stays open is not an asymmetry question: how faithful the simulator's combat is, and how
often the no-city, nothing-left-out case occurs at all. The second is counted, with no
verdict, by the exploratory addendum of the census
(`kb/specs/20261123-expressivity-census-preregistration.md`).

## Deviations from the design

- Three root families translate to 10 root actions, not 8 or fewer; 8 are sampled, the same 8
  in every arm for a given seed.
- In-tree "lines" are counted per tree step, so they cannot be compared across arms.
