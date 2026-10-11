# Anthropic Message Batches fixtures

Response bodies in the shapes of Anthropic's Message Batches API reference
(platform.claude.com/docs/en/api/messages/batches/{create,retrieve,results,list}
and build-with-claude/batch-processing, read 2026-10-11): the batch object, its
request counts, the four result types and the list page. IDs, times and texts
are synthetic. Custom IDs carry the fixture submission nonce
`0f8fad5b-d9cb-469f-a165-70867728950e` without hyphens, as markitai sends them.
No provider was contacted to make these files.

`recorded/` holds the bodies of one live Message Batch made on 2026-10-11 by this
build (`markitai inputs --llm --llm-batch`, `anthropic/claude-haiku-5-5`, two
synthetic Markdown files) through a counting loopback proxy that captured
bodies only, never headers: the create request and its response, the status
once the batch ended, and the results file. IDs are Anthropic's; nothing in
them identifies an account.
