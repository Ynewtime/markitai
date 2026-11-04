# Core requests from the Windows app

## 1. An image with no text arrived as `conversion_error` (done)

The core answered `conversion_error` for an image with no text layer when
neither OCR nor AI was on, so the `image_only` row of DESIGN.md section 9 was
unreachable. `markitai-app`'s `convert_json` now answers the error code
`image_only` (test `an_image_without_text_is_image_only`), and DESIGN.md 4.1
step 4 maps every `skip_reason` to `skipped`. The Windows app follows both.
