# Offline token tariff provenance

This catalog contains ten exact model keys from the LiteLLM 1.106.0.dev2
backup data (a development pre-release: on 2026-10-09 the newest PyPI release
carrying the claude-haiku-5-5 row; the wheel file and digest are named in
provenance.json). The source file hash, distribution RECORD entry, exact source-byte
ranges and decimals are recorded in provenance.json. source-rows.json retains
each original object verbatim; surrounding JSON indentation is newly assembled.
Two additional exact Claude aliases are mapped to their dated tariff using the
official model-page evidence listed separately in provenance.json.
LiteLLM-LICENSE is the unmodified distribution notice. No generated license
template or runtime download is used.

Official verification links and the 2026-10-09 capture date identify this bounded
snapshot; they do not establish the source data's effective date or guarantee
future prices. Estimates use public list tariffs and actual response counters,
not negotiated invoices, credits or taxes. Only the two exact first-party
endpoints and listed models are priced. Unsupported categories, tiers, contexts,
models and proxies remain unknown. Reasoning tokens are already part of output;
cached input and cache writes are each charged once, at their own rates. Batch is classified per attempt and never applied
to an aggregate that might also contain Standard fallback requests.

The full workspace license review remains a separate task; this provenance record
is not legal advice or a claim that all project licensing gaps are resolved.
