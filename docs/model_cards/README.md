# Model cards

Only one artifact is published: the hex-crystalline spatial encoder
(`ianshank/civ6-hexcrystal-encoder`), documented in
[`hexcrystal_encoder.md`](hexcrystal_encoder.md). There are no cards for a
transformer policy/value network or a MuZero world model because no such
checkpoint has been published; a card for an unpublished model would describe
an artifact that does not exist. There is no dataset card for the same reason.

Cards follow the Hugging Face model-card front matter and the sections of
Mitchell et al. (2019): intended use, training, evaluation with its scope and
limitations, risks, and architecture. Evaluation numbers quote the score-rate
interval from `docs/EVALUATION.md` and state the search epoch of the artifact
they come from (`docs/evidence/README.md`).
