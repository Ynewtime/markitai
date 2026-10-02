# Exact-version upstream license evidence

Packaging validates this directory offline before copying it. Package manifests,
published VCS records, original source notices and exact byte identities remain
part of the payload. This is mechanical evidence collection, not legal review.

All seventeen supplemental crate versions now carry complete license terms. Six
were collected directly from their exact upstream commits in R29. R30 adds five
Apache-2.0 options and selectors' MPL-2.0 terms from the official publishers
expressly linked by each pinned source notice. Three complete MIT originals have
separately verified historical provenance. The helper permits only the reviewed
package/version/commit and URL/digest combinations; arbitrary URLs, later versions
or unrelated historical licenses cannot inherit this classification.

R49 adds nom-language 0.1.0's original MIT text and tract-extra 0.23.8's
original workspace notice plus MIT and Apache-2.0 texts. Both child manifests
directly declare these options; neither is relabeled as a workspace-inherited
license. Their parent published originals agree with the pinned repository terms.
There are sixteen raw exact manifest matches and one explicitly reviewed
publication version stamp. For tract-extra alone, the pinned upstream version
`0.23.8-pre` becomes published `0.23.8`; all other manifest bytes and both Rust
files agree. The raw `dirty: true` VCS record and both manifests are retained.
The helper binds this exception to the complete fixed package/commit/checksum
and source identities. It does not normalize other manifests or trust arbitrary
dirty sources. These mechanical facts do not constitute legal review.

The unmodified nom-language and tract-extra crates.io source archives are also
included, making three pinned archives with selectors. Their seven and six
original regular files, respectively, are verified against the published source;
package VCS/original manifests and the same-commit Rust files remain in the
payload. `license-gaps-r49-provenance.json` records the fourteen actual anonymous
pinned-URL retrievals and source comparisons. Original R49 records reporting two
missing texts remain historical evidence; new collections derive their own state.

For objc2 0.6.4, objc2-encode 4.1.0 and objc2-foundation 0.3.2, the complete
original upstream MIT text, including Steven Sheldon's notice, is retained
unchanged. The 2016 original exactly matches the file removed by the 2025
licensing clarification. Target-anchored histories and both patches accompany
it. Each current pinned release still explicitly declares MIT; its original
notice, including the Apple SDK discussion, remains beside the historical text.

These three texts close the mechanical missing-text/source/copy gap. Their
`source_kind` remains `historical_notice`; they are not relabeled as files present
at the target commit. The collector reports `historical_text_packages: 3` and
`legal_review: not_performed` separately. It does not establish an exhaustive
contributor copyright inventory or resolve SDK licensing. No author, year or
license grant was invented. Raw R30 provenance retains its earlier classification
as a historical evidence record, not the current packaging decision.

The complete original selectors 0.38.0 source archive is included at
`source-archives/selectors-0.38.0.crate` (SHA-256
`8adfa1c298912827b8a28b223b3b874357397ae706e6190acd9bf28cee99114d`).
It is the unmodified crates.io archive, with matching VCS/manifest and all 22 files
checked against the published local registry source. The MPL license and original
notice are under `overlay/licenses/selectors-0.38.0/`. The manifest records the
canonical source download URL as well as this offline copy. See MPL sections
3.1–3.4 for source and notice terms; supplying these materials is not a legal
assessment of every dependency or distribution scenario.

The objc2-family Apple SDK discussion is preserved as written upstream. Apache
section 4 notice conditions and any other existing notices are not displaced by
adding the complete terms. The historical MIT evidence does not relicense the
MIT-only crates. Collection does not elect an alternative for other packages.

`inventory.json` covers every other file and detects unreviewed changes; the
source/build identity binds that inventory. `round30/provenance.json` and raw
response records retain successes and failed retrieval attempts. The generic MIT
publisher template was not substituted. Old R29 evidence and its recorded 404
remain intact. CLI archives, npm packages and Python wheels copy the validated
subset as well as the static-Go bundle; wheel paths are relative to its
`.dist-info/licenses/` attribution directory.
