# Pending Python markitai batches

Output directories that Python markitai 1.2.1 left with a pending Provider
Batch, used to test that this build collects them (`tests/python_batch.rs`).

- `openai/`: `markitai inputs -o out --llm --llm-batch --alt` with
  `openai/gpt-5.6-luna` (two documents and one image analysis).
- `anthropic/`: the same inputs without `--alt`, with
  `anthropic/claude-haiku-5-5` (Message Batches).

Made on 2026-10-11 with the 1.2.1 wheel (sha256
`5ac00c3665852bf99f012175d27a466c9e8e3c187327c0f886554093d24be196`) against a
loopback stub of the OpenAI Files/Batches and Anthropic Message Batches APIs
that accepted the create call and failed every status read, so Python kept
the state for a later collection. No provider was contacted; the key was a
fixture string. Inputs were authored for this test: `notes.md` (a link and a
code block) and `deck.pptx` (two slides, one generated PNG). Python's run
reports and checkpoints, which hold the generating machine's paths, were left
out. Everything else is byte for byte as Python wrote it.
