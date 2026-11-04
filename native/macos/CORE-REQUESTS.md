# Core requests from the macOS app

None of these block the app. Each was seen on the built app or its link step.

## 1. Build the static library for the app's minimum macOS

`cargo rustc -p markitai-app --release --lib --crate-type staticlib` without
`MACOSX_DEPLOYMENT_TARGET` compiles the C parts of dependencies for the build
machine's macOS. The app targets macOS 13 (DESIGN.md section 2), so the link
then warns that objects were "built for newer macOS". `build.py` sets
`MACOSX_DEPLOYMENT_TARGET=13.0` for its cargo call; a build of the same
`target/release` without it rebuilds everything (several minutes) and the next
`build.py` rebuilds again.

Request: set the deployment target where the crate is built for the apps (for
example `[env] MACOSX_DEPLOYMENT_TARGET = "13.0"` in `.cargo/config.toml`, or a
note in the crate's build instructions), so every build of the library agrees.

## 2. The `image_only` message names Python options

For an image with no text when neither AI nor OCR is on, `convert_json` answers
`image_only` with a message like:

> quadrants-121x81-orientation6.avif is an image file with no text to extract.
> Enable LLM (llm=True) or OCR (ocr=True) for content extraction.

The app shows this under Details. `llm=True` and `ocr=True` are the Python
API's spelling; the app's own sentence (`error.image_only`) already says which
settings to turn on.

Request: a message in plain words (for example "Turn on AI or OCR to read its
text"), or leave the option names to each front end.
