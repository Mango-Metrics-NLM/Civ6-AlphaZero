# Governance

Civ6-AlphaZero is maintained by a single maintainer, [@ianshank](https://github.com/ianshank),
who is the sole entry in [`CODEOWNERS`](.github/CODEOWNERS). That is a fact about the
project's size, not a policy goal; the documents below exist so that decisions are
recorded rather than remembered.

- **Scope, non-goals and invariants** live in [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md).
  Changes to them follow the amendment protocol in its §7 and are merged only by the
  maintainer.
- **Decision rights** for automated and human contributors are defined in
  [`CLAUDE.md`](CLAUDE.md) §3.2 (autonomous / confirm-first / prohibited). A change that
  alters deployed behaviour, deletes more than a hundred lines, or touches distributed
  coordination needs an explicit maintainer decision before it is implemented.
- **Substantive changes** are proposed under [`openspec/changes/`](openspec/changes/)
  before implementation and validated by `scripts/validate_openspec.py`.
- **Releases** are semantic; the maintainer pushes the `vX.Y.Z` tag after the changelog
  is rolled (see [`CONTRIBUTING.md`](CONTRIBUTING.md) → Releases).
- **Security reports** go through GitHub Security Advisories, never public issues
  (see [`SECURITY.md`](SECURITY.md)).
- **Conduct** questions go to the maintainer through the private channel named in
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
