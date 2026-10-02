# Local image OCR

On macOS 11 or later, local OCR uses the operating system's Vision framework
through Rust bindings. It runs in the current process without Python, Node,
Tesseract, a browser, downloaded OCR weights, provider credentials or a remote
recognition service. On Windows and Linux it uses the [portable
engine](#the-portable-engine-windows-and-linux): PaddleOCR's models, downloaded
once into the private Markitai home, run in process by a pure-Rust inference
engine. An x86-64 build running under Rosetta 2 on Apple
silicon fails recognition with an error that names Rosetta (Vision reports
failure there without an error of its own); use the arm64 build ([details](validation/macos-x86_64-rosetta.md)).
Most of this page describes Vision; the portable engine's languages, default
language and measurements have [their own section](#the-portable-engine-windows-and-linux),
and both share the [reading order](#bounds-and-reading-order) and [turned pages,
code and zeros](#turned-pages-code-numbers-and-table-cells) steps.
This page covers image inputs, including every page
of a multi-page TIFF and HEIF/AVIF images; scanned PDF pages use the same
recognizer as described in [PDF page OCR](pdf-ocr.md), and Office page images in
[Office rendering](office-rendering.md).

```sh
markitai scan.png --ocr                                        # English, Chinese, Japanese or Korean
markitai scan.png --ocr --config-json '{"ocr":{"lang":"zh"}}'  # Chinese (reads Latin too), nothing else
markitai scan.png --ocr -o out/                                 # keep the image asset
```

## Selection and output

`ocr.enabled=true` (`--ocr` in the CLI) selects local OCR when LLM enhancement is
disabled. The image reference remains in the base document and recognized text
is appended below it. Metadata records `ocr_used: true` and `ocr_path: vision`
(`paddle` for the portable engine).
A successfully processed image with no recognized text retains its image
reference and emits a warning. An invalid image, unsupported language or engine
failure fails the conversion instead of inserting error text into Markdown. (Under
the [default language](#the-default-language) only the English reading can fail
it: a Chinese, Japanese or Korean reading that this system lacks or that fails is
skipped.)

With LLM enabled, the existing image-to-model route remains the default.
Setting `MARKITAI_NO_VLM_OCR` to a nonempty value other than `0`, `false` or `no`
(case insensitive, ignoring surrounding whitespace), together with OCR enabled,
chooses local OCR followed by text enhancement. Image bytes
are not included in that model request. Enabling OCR does not enable LLM or
require a configured model. Disabling both still produces the existing image-only
error or CLI skip behavior.

OCR consumes the original raster bytes, independently of preview compression,
image width settings and embedded-image filters. Static SVG uses the existing
bounded in-process SVG rasterizer. JPEG, PNG, GIF, BMP, TIFF and WebP use the
enabled Rust decoders; animation uses its first image. HEIF and AVIF are decoded
by macOS ImageIO ([images](images.md#heif-and-avif)), and the primary image is
recognized. A multi-page TIFF is recognized page by page: each page keeps its
`<!-- Page number: N -->` marker and preview, followed by its text, and a page
without text adds a warning instead of ending the conversion.

## Language selection

`ocr.lang` defaults to `en`, which is a policy of its own: [the default
language](#the-default-language) reads English and, when that fails, Chinese,
Japanese or Korean. Every other value names one Vision language, and each call
asks the configured accurate Vision recognizer which languages its current OS
and request revision support, then requires a match. It never substitutes
English for an unsupported language, and never reads another language than the
one written. System revisions can change supported languages and recognition
results.

| Configuration spelling | Requested Vision language |
| --- | --- |
| `en` | `en-US`, then `zh-Hans`, `ko-KR` and `ja-JP` when the reading failed ([default language](#the-default-language)) |
| `en-US`, `en_US` | `en-US` only |
| `zh`, `zh_cn`, `zh-cn`, `cn`, `ch` | `zh-Hans` |
| `zh_tw`, `zh-tw`, `cht`, `chinese_cht` | `zh-Hant` |
| `ja`, `jp`, `japan` | `ja-JP` |
| `ko`, `korean` | `ko-KR` |
| `ar`, `arabic` | `ar-SA`, if installed Vision supports it |
| `fr`, `french`; `de`, `german` | `fr-FR`; `de-DE` |
| `es`; `it`; `pt` | `es-ES`; `it-IT`; `pt-BR` |

Case, surrounding whitespace and underscore/hyphen aliases are normalized.
Other short ISO codes may match the system's regional spelling. An explicit
region or script must match exactly, ignoring case. The reference's RapidOCR
model-family names such as `latin` and `cyrillic` are not Vision languages and
are rejected. Structural configuration acceptance does not imply runtime
language availability. `ocr.per_page_routing` has no effect on this image path.

A language the system lacks fails with an error that names it and lists the
languages the system reads. On macOS 27.0.1 (26A434) the accurate recognizer
(request revision 3) reads `en-US`, `fr-FR`, `it-IT`, `de-DE`, `es-ES`,
`pt-BR`, `zh-Hans`, `zh-Hant`, `yue-Hans`, `yue-Hant`, `ko-KR`, `ja-JP`,
`ru-RU`, `uk-UA`, `th-TH`, `vi-VT`, `ar-SA`, `ars-SA`, `tr-TR`, `id-ID`,
`cs-CZ`, `da-DK`, `nl-NL`, `no-NO`, `nn-NO`, `nb-NO`, `ms-MY`, `pl-PL`, `ro-RO`,
`sv-SE`, `fi-FI`, `hi-IN` and `mr-IN`. Greek and Hebrew are not among them:
`ocr.lang` `el` and `he` are rejected because this system's Vision cannot read
those scripts, not because of their spelling, and no setting reads them
locally. Thai (`th`), Arabic (`ar`), Hindi (`hi`), Vietnamese (`vi`) and the
others are read when written. (Vision returns Arabic lines with their letters
in reverse order, `لوسكلا بلكلا` for `الكلب الكسول`; they are passed on as
returned.)

## The default language

Vision's English recognizer reads Latin and Cyrillic text. For Chinese,
Japanese and Korean text it returns nothing, or a few symbols and Latin
look-alikes (`#Æ[× •` for a line of Chinese), without an error. Under `en`, the
default, an image is therefore read as `en-US` first, as it always was, and the
reading is judged by what Vision reports: confidence is 1.0 for nearly every
line of English it reads, and 0.3 or 0.5 for what it makes of other scripts.

| English reading | Next |
|---|---|
| three or more lines, fewer than a quarter of them below 0.9 | nothing: the English reading stands |
| one or two lines, all at 0.9 or more (*short*) | Chinese |
| at least a quarter of the lines below 0.9 | Chinese |
| no text, or at least half of the lines below 0.9 | Chinese, then Korean |

Short readings are read as Chinese because the English recognizer returns
confident Latin fragments for a line of Chinese with Latin words (`* PDF. Word,
Excel SEAT.` for `支持 PDF、Word、Excel 与图片转换。`), which only another reading
tells from English; it costs one more reading of an image of at most two lines.
The Chinese reading (`zh-Hans`) also reads Latin, Japanese and Traditional
Chinese: Vision returns the same text for `zh-Hans` and `zh-Hant`, so Traditional
is not read a second time. Each reading takes the
[aids](#chinese-japanese-and-korean-recognition-aids) of its language, exactly
as if `ocr.lang` named it.

A later reading replaces the English one only when it holds text of its script:
at least four letters of it (Han and kana for Chinese, Hangul for Korean), at
least a quarter of the reading's letters and digits, and line confidences that
add up to at least 3 over those letters (so ten letters at 0.3 are text and
three are not). A Chinese reading with at least 32 letters in lines of
confidence 0.5 or more settles the language; with fewer, Korean is read too and
the reading with more confidence-weighted letters of its own script is kept,
because Vision's Chinese recognizer makes confident Han and kana of Hangul
(never more than 30 such letters in 184 Korean images, while 27 of 416 Chinese
and Japanese images have fewer). A Chinese reading with at least eight kana,
a fifth or more of its letters, is read once more as `ja-JP`, which has its own
aids, and that reading replaces it unless it holds less text. A Vision language
this system lacks, and a reading that fails, are skipped: the English reading
stands, as it always did.

When no reading is better, the English reading is the output. If text was seen
and not read, the conversion warns `Local OCR could not read this image: ...`,
and of the English reading keeps only the lines it is sure of (0.9 or more):
what English makes of Greek, Hebrew, Thai or Arabic (`ApıOpoç троло 2026-0042`,
`üp Julügö 2026-0042`, `026-42`) is not text. Text was seen when the English
reading failed and holds a doubtful line of at least five characters, or when
another reading found such a line where English found none (Hebrew, which
English does not see at all). The warning names the image, TIFF page, PDF page
or Office page, says how to set `ocr.lang`, and lists the languages this
system's recognizer reads ([above](#language-selection)); an image warned this
way has no "no readable text" warning besides. A blank page or a photograph
without text has only the usual "no readable text" warning, and a picture
inside a PDF never warns. Vision reports the short cells of an English table
(`7`, `12`, `Q1`) at 0.5 as well, so such a table fails the English judgement
too, but is not taken for another script: two of the 46 rendered tables below
warned and lost no text before, and they now neither warn nor lose text. The
chalkboard account picture, which warned before, now does not (its one
doubtful line has four characters) and reads as before.

A written language is never replaced: `zh`, `ja`, `ko`, `fr`, `ar`, `en-US`,
`en_US` and every other value except `en` read that language alone, with output
identical to the policy's introduction (1,210 conversions of the corpora below
with `zh`, `ja`, `ko` and `en-US`, compared before and after); later aids
([spaces](#chinese-japanese-and-korean-recognition-aids), [tables and
numbers](#turned-pages-code-numbers-and-table-cells)) apply to every language.
The configuration fills in its
default before the recognizer sees it, so a default `en` and a written `en`
cannot be told apart: write `en-US` to read English only. (The reference's `en`
model also reads Chinese.)

Measured on macOS 27.0.1 (26A434), Apple silicon (Apple M5 Max), with `cargo build
-p markitai-cli --release --locked`, before (`1a86650`, 21,483,920 bytes) and
after (21,483,920 bytes), by whole conversions (`--ocr --no-llm`, an isolated
`MARKITAI_HOME`, `-o`, no `ocr.lang`). Character error rate is edit distance over
ground-truth length, with whitespace removed for Chinese, Japanese and Korean and
collapsed for English; the corpora are those above (synthetic renders, not scans
or photographs).

| Corpus (images) | Before | After | Written language |
|---|---:|---:|---:|
| English prose, numbers, two columns (126) | 0.07%, 0.00%, 0.30% | identical text on every image | `en-US`: identical |
| R45 Chinese (24) | 94.73% | 0.97% | `zh` 0.97% |
| Held-out Chinese (150), dense pages (4) | 91.87%, 84.04% | 1.82%, 1.61% | `zh` 1.82%, 1.61% |
| Traditional Chinese (24) | 100.00% | 1.41% | `zh` 1.41% |
| Japanese (150), full pages (4) | 99.63%, 98.64% | 0.11%, 0.09% | `ja` 0.12%, 0.04%; `zh` 0.20%, 0.09% |
| Korean (150), full pages (4) | 99.67%, 99.71% | 0.91%, 0.87% | `ko` 0.91%, 0.87% |
| 60 DPI Chinese, Japanese, Korean (30 each) | 94.50%, 99.25%, 99.84% | 9.83%, 1.17%, 4.54% | `zh` 9.83%, `ja` 0.85%, `ko` 1.78% |

Before, 420 of the 600 Chinese, Japanese and Korean images came out empty and
the rest as symbols, none with a Chinese, Japanese or Korean character; after,
every one has text. The 202 Chinese images (Simplified and Traditional), the 154
Korean images and the thirty 60-DPI Chinese images read exactly as the written
`zh` or `ko` reads them, and 149 of 154 Japanese images as `ja` reads them. The
one Korean image that reads worse than with `ko` is a 60-DPI one that the
Chinese recognizer turns into confident Han. Of 75 other images (English,
French, German and Spanish; Russian; code, a table and symbols; Arabic, Hebrew,
Greek, Thai and Hindi; Chinese, Japanese and Korean at 14 and 24 pixels, alone
and mixed with English; macOS's desktop and account pictures; a blank page and
noise) nine changed, all with Chinese, Japanese or Korean text in them
(`i-cjk-300.png` among them) and now read in that language; the other 66 are
unchanged, including every English, Latin and Cyrillic image, and four of them
(Greek, Hebrew, Thai and a chalkboard picture) have the warning.

Whole conversions, one at a time, before and after alternating, median of four
rounds (two for the others) per image, mean per corpus, in milliseconds:

| Images | Before | After | Written language, after |
|---|---:|---:|---:|
| English prose (72), numbers (36), two columns (18) | 169.5, 172.9, 193.1 | 169.8, 173.5, 192.4 | |
| Held-out Chinese (150) | 273.8 (`zh`) | 364.7 | 274.5 |
| Traditional Chinese (24) | 277.8 (`zh`) | 357.8 | 278.2 |
| Japanese (150) | 246.7 (`ja`) | 423.3 | 247.1 |
| Korean (150) | 218.6 (`ko`) | 468.2 | 219.8 |

The 126 English images differ from before by 0.25 ms on average (5th to 95th
percentile −5.1 to +6.0 ms) and none is more than 10% slower: an English image
that reads as sound costs only the check of its confidences. A Chinese,
Japanese or Korean image takes the English reading, one to three more readings
and the models they load: 90 ms (Chinese), 177 ms (Japanese, with its second
reading) and 248 ms (Korean, after the Chinese reading) more than the written
language, in a process that converts one image. A document or batch converts in
one process, which loads each model once. A picture without text costs two more
readings: a blank 1600×1200 page 94 to 288 ms, small photographs about 85 to
255–300 ms, 3000×3000 photographs 380–430 to 730–850 ms, and `sample.jpg`,
whose 11-pixel text Vision does not read in any language, 90 to 645 ms. An
English image of one or two lines costs one more reading: `Hello World` 102 to
193 ms, two lines of German 124 to 254 ms. The same corpora with a written
language took the same time before and after.

Limits. English among three or more lines that reads as confident Latin is not
read again, so Chinese, Japanese or Korean words among English lines stay unread
(`Markitai converts documents to Markdown.` above a Chinese line that reads as
`I PDF, Word 5EA, W Markdown X4.`): set `ocr.lang` to `zh`, which reads Latin
too, for such pages. Scripts that no reading covers (Arabic, Thai,
Devanagari) still need `ocr.lang`, and Greek and Hebrew cannot be read on this
system at all; the warning says so when Vision found text there. A single
confident line of another script (Hindi read as `fộ-4 415 2026`) is short, not
failed, and is kept without a warning. Korean text at 60 DPI that the Chinese
reading turns into confident
Han can be kept as Chinese (1 of 184 Korean images).

## Bounds and reading order

Encoded input is limited to 64 MiB. Decoded images must fit 32 million pixels,
with a 256 MiB decoder allocation limit. EXIF orientation is applied before OCR;
alpha is composited over white. A bounded PNG buffer of at most 128 MiB transfers
the normalized pixels to Vision in memory. Source paths, user configuration
files and provider keys are not passed to the engine.

An internal renderer entry point also accepts upright RGB pixels already
composited on white. It rechecks nonzero dimensions, the same 32-million-pixel
limit and an exact three-byte-per-pixel buffer before bounded PNG encoding.
It does not rotate, resize or recolor those pixels. Language selection, Vision
recognition and reading-order assembly are shared with encoded-image OCR;
the encoded-image path still applies its original format, orientation and alpha
rules. This entry point alone does not establish PDF routing or rendering support.

The request uses accurate recognition with language correction. At most 10,000
observations and 8 MiB of recognized text are accepted. Text and line rectangles
are checked for finite, valid confidence and geometry. Internal rectangles use
top-left pixel coordinates after orientation correction, and mean confidence
uses nonempty recognized lines. These diagnostics are not new fields in the
public conversion JSON.

Lines are ordered top to bottom; observations sharing at least 60% of the shorter
vertical band form a left-to-right row. A vertical gap greater than 80% of the
taller adjacent row separates paragraphs. Text columns are read column by column:
the widest gutter, at least one line height wide, between the lines narrower
than three fifths of the text splits the page when both sides hold at least
three lines of prose (a median of 12 characters) side by side. Lines crossing
the gutter, such as a title, separate sections read left column first, and
nested columns are found in each side. Short cells side by side, such as a
receipt's items and prices, stay rows. Two lines of a row are joined without a
space where a Chinese or Japanese letter or full-width mark meets the other
line less than 0.15 line heights away. The reference's table, vertical-writing
and marginal-note reconstruction is not reproduced. On a rendered corpus
([R45](validation/ocr-quality-round45.md)) English and number text is read more
accurately than by the reference's RapidOCR, two-column pages in order, and
Chinese, with the [aids below](#chinese-japanese-and-korean-recognition-aids), at 1.7 to 2.6 times
its character error rate; Japanese is read more accurately than by RapidOCR on
a corpus written for that check. Vision inference itself has no wall-clock cancellation deadline
in this slice, and recognition quality is not guaranteed for handwriting, small
text or every supported language.

## Turned pages, code, numbers and table cells

These steps apply to every language, after the reading is chosen.

- **Turned pages are read upright.** Vision reads text turned a quarter or a
  half and gives each line's corners. When three quarters of the letters (in
  lines of three or more) run the same other way, down, up or right to left,
  the lines' rectangles are turned upright and ordered and joined as on an
  upright page; the text is not read again, and rectangles are reported in
  the upright page's pixels. A sideways label on an upright page stays as it
  is. An image turned a quarter clockwise (`img_rot90.png`) came out as one
  paragraph in reverse line order and an upside-down one in reverse line
  order; both now read as four lines in order.
- **Code keeps its indentation.** At least three consecutive rows of one line
  each in a fixed pitch (every line of three or more characters within 15% of
  the median width per character), one indented by one and a half characters
  or more, and a third holding brackets, `=`, `;` or a closing `:` are code:
  they are fenced (with a fence longer than any backtick run inside) and each
  is indented by its left edge in characters. Text in proportional fonts,
  receipts and tables in fixed-pitch fonts (their rows all start at one edge)
  and unindented code stay text. `img_code.png`, `def fib(n):` and its body,
  read as five flush lines and now as the code with its four- and
  eight-space indents.
- **Zeros read as letters are mended.** In a token of digits and number marks
  only (`.,:;/-+%$€£¥#()[]'"`), with at least two digits, a run of `O`, `o`,
  `ø` or `Ø` between two digits is read as zeros: `2ø26` is 2026, the
  scanned fixture's slashed zero below. Words and formulas (`Fe2O3`, `CO2`),
  the ends of numbers (`10O`) and tokens of one digit stay as read. Other
  look-alikes (`ł`, `ą`) are not mapped to digits; the regions holding them
  are read again (next item).
- **Table cells and garbled numbers are read again.** Vision often drops a
  lone short cell, most often a single digit, from a table's row, and reads
  some numbers as letters of other alphabets (`1,200,00łł`, `Q2 202łąłą`,
  `З` for 3, `183•33`). Three or more rows of cells at least a row height
  apart are a table; its columns are those of the rows with the most cells
  (two or more such rows), and a row with fewer cells, each in one column,
  misses the cells of its other columns whose cells hold at most 12
  characters (by median). The missing cell's region is read again: with the
  row's previous cell, enlarged two and then three times, and alone inside a
  white margin of one row height, enlarged twice, until a reading places
  upright text inside the cell's column and row. Vision reads a lone digit
  upside down at times (6 as 9), and the previous cell shows which way up the
  row stands. A reading that is only a rule (`|`, `l`), longer than twice the
  column's longest cell or, in a Latin reading, holds letters of other
  alphabets, adds nothing. In a reading of Latin script a line whose token
  holds a digit and a letter of another alphabet, a bullet or a noncharacter
  is read again, enlarged two and three times and with the margin, and its
  text is replaced only when that reading repeats every other token and puts
  digits and number marks, no longer than before, where the garbage was
  (Latin letters touching the garbage, as in `72.7zął`, count with it). A
  cell of a column of numbers (at least half of its cells are) that holds
  only a Cyrillic letter drawn like a digit (`З`, `О`, `б`), which the English
  recognizer reads, is that digit; elsewhere such letters are words
  (Ukrainian `з`) and stay. At most 12 regions and 24 readings per image; a
  turned page and a page warned as unread are not read again, and an image
  without a table or garbled number costs nothing more.

Measured on macOS 27.0.1 (26A434), Apple silicon, release builds of
`09d8713` (before, 21,946,800 bytes) and after (21,996,400 bytes; 41,280
bytes more machine code for this section's steps, the spaces aid and the
warning together), by whole conversions (`--ocr --no-llm`, an isolated
`MARKITAI_HOME`, `-o`, no `ocr.lang`). The tables are 46 rendered screenshots
written for this check (Helvetica, Arial, Times New Roman, Georgia, Verdana
and Menlo at 20 to 36 pixels, with and without rules, four columns of items,
quantities of one or two digits and prices, twelve with an empty cell; six
grids of single digits and percentages; four with Chinese headers), with the
four table probes of the quality inventory (`img_t2.png`, `img_t3.png`,
`img_t4.png`, `img_table.png`).

| Corpus (images) | Before | After |
|---|---:|---:|
| Rendered tables (46) | 2.80% (139 edits) | 0.91% (45 edits) |
| Inventory table probes (4) | 4.96% | 0.71% |
| Inventory turned, code, receipt and other probes (13) | 16.43% | 1.76% |
| English prose, numbers, two columns (126) | 0.07%, 0.00%, 0.30% | identical text on every image |
| Chinese: R45, held-out, full pages, Traditional (202) | 0.97%, 1.82%, 1.61%, 1.41% | the same (whitespace removed) |

Of the 62 nonempty cells missing from the first readings of these 50 tables,
the three readings above recover 48 in a Swift probe of the same requests
(one wrongly, `QI` for `Q1`); the cell alone, enlarged two and three times,
recovers 27, and other enlargements (1.5 and 2.5 times, cubic filtering) no
more. In the conversions no cell was filled wrongly, and no image of the
corpora above lost or changed a line except as described here. The 45
edits left are 17 cells still missing (lone `3`, `0` and `6` mostly), three
numbers whose second readings repeat the garbage (`183•33`, `72.7zął`, a
noncharacter for a point) and two prices that Vision reads confidently as
others (`50.30` for `56.36`). No image of the 328 corpus images is fenced as
code; of the 98 other images, `img_code.png` is. Two of the 46 tables warned
as unread before and none does now; the seven images that warn are the
Greek, Hebrew, Thai and Arabic ones, which now carry no garbage.

Whole conversions were timed one at a time after a warm-up, before and after
alternating, median of three rounds (two for Chinese and the other images)
per image, mean per corpus, in milliseconds:

| Images | Before | After | Per image, 5th to 95th percentile |
|---|---:|---:|---:|
| English prose (72), numbers (36), two columns (18) | 160.6, 161.1, 184.2 | 160.7, 160.5, 184.3 | −8.1 to +6.3 |
| R45 Chinese (24), held-out Chinese (150), Traditional (24) | 287.6, 326.6, 299.4 | 288.1, 326.8, 299.0 | −17.7 to +17.1 |
| Full Chinese pages (4) | 1,120.6 | 1,139.7 | −55.5 to +64.5 |
| Rendered tables (46) | 188.5 | 252.1 | −4.8 to +154.5 |
| Inventory probes (21) | 217.7 | 229.0 | −4.6 to +67.1 |
| Other images: languages, symbols, code (29); text-free pictures (34) | 282.9, 376.5 | 287.4, 377.5 | −22.6 to +55.0 |

A table costs the readings of its missing cells, about 15 to 25 ms each and
three for a genuinely empty cell, within the budget of 24; an image without a
table or garbled number costs only the geometry. Four of about 5,000 timed
conversions, two of each build, waited more than 120 seconds in Vision with
almost no processor time and were run again; repeated five times, those
images converted in 0.17 to 0.32 seconds.

## Chinese, Japanese and Korean recognition aids

These steps run only when the requested language is `zh-Hans`, `zh-Hant`,
`ja-JP` or `ko-KR`, or when the [default language](#the-default-language) reads
the image in one of them. English and every other language take exactly the path
described above. Each step measures the script's full-width letters: Han
characters for Chinese; Han characters and kana for Japanese (small kana such
as ゃ and the prolonged sound mark ー are drawn narrower, but Vision boxes them
as wide as the other letters); Hangul syllables for Korean. Vision's Korean
recognizer does not read Hanja: in Korean text they come out as other
characters, which no step corrects.

- **Small text is read again, enlarged** (Chinese and Japanese). When the
  median height of the recognized lines that hold letters is below 24 pixels
  (12-point text at 96 DPI or smaller), the image is enlarged with Lanczos
  filtering toward 32-pixel lines, by 1.25 to 4 times and within the
  32-million-pixel limit, and that second reading replaces the first.
  Rectangles stay in the original image's pixels. The first reading is
  discarded, so such an image takes about twice as long. Korean is not
  enlarged: Vision reads Hangul at 96 and 72 DPI as well at its own size, and
  enlarged copies changed vowels (에 read as 메) and dropped periods and hyphens.
- **Text missed entirely is read again, enlarged** (all three). Vision
  sometimes returns no text at all for small print that it reads at other
  sizes. When the reading is empty, a fast reading (Vision's Latin
  recognizer) supplies the sizes of the text lines it finds, never their
  text. Lines at least three times as wide as they are tall count, so that a
  speck in a photograph does not; when their median height is below 24 pixels
  the image is read again enlarged, as above. A blank or text-free image costs
  that one fast reading more.
- **Dropped characters are recovered** (all three). Vision sometimes drops a
  character, most often 的 in Chinese or a kanji in Japanese (箱 after
  段ボール), and stretches its neighbour's box over the gap. In a line with at
  least four letters, a letter box wider than 1.6 times the line's median
  letter box marks a suspect; in Korean the first syllable of a line does not,
  because Vision boxes it up to about twice as wide (27 of the 36 wide Korean
  boxes measured, none hiding a dropped syllable). The region of the suspect
  and up to four recognized neighbours on each side (word spaces are skipped)
  is read again.
  One letter is inserted only when that reading places it directly before or
  after the suspect and repeats up to two neighbours on each side exactly; any
  other reading, including a failed one, leaves the line unchanged. At most 32
  regions are read again per image.
- **Spaces the image shows are kept** (Chinese and Japanese). Between a Han
  character or kana and a Latin letter or digit, Vision's Chinese recognizer
  dropped 385 of the 982 spaces of the held-out, R45, Traditional and full-page
  texts, and never put one where the text has none (12 such places, and 3,101
  between two Latin letters). Its letter boxes cover the line without gaps, so
  the space is measured in the pixels of the image read: where a reading has
  no space at such a junction, the widest run of ink-free columns between the
  two letters' centres is measured in the line's band, and a run of 0.22 line
  heights or more is a space. Where the text has none the runs were 0.02 to
  0.12 line heights; where it has one, 0.09 to 0.51, a twentieth below 0.2.
  Spaces the reading has are kept. Between two Latin letters runs overlap
  (0.30 without a space, 0.15 with one), so `CLIJSON` for `CLI JSON` stays.

Measured on macOS 27.0.1 (26A434), Apple silicon, `cargo build -p markitai-cli
--release --locked`, before (r18, 21,731,920 bytes) and after (21,748,496
bytes). Character error rate is edit distance over ground-truth length with
whitespace removed; the reference is RapidOCR (PP-OCRv6) in the reference's
virtual environment.

| Corpus / variant | Reference | Before | After |
|---|---:|---:|---:|
| R45 Chinese 300 DPI / 150 DPI / scan-like | 0.34% / 0.52% / 0.52% | 0.86% / 1.03% / 1.55% | 0.69% / 0.86% / 1.37% |
| Held-out Chinese 300 / 150 / scan-like | 1.60% / 2.17% / 2.57% | 1.44% / 1.28% / 1.56% | 1.28% / 1.16% / 1.52% |
| Held-out Chinese 96 DPI / 72 DPI | 0.96% / 1.68% | 2.21% / 8.82% | 1.93% / 3.21% |
| Full Chinese pages 150 / 96 DPI | 1.92% / 2.89% | 1.19% / 2.97% | 1.16% / 2.06% |
| Traditional check 150 / 96 / 72 DPI | not run | 0.30% / 0.60% / 5.44% | 0.00% / 0.30% / 3.93% |

The R45 English, number and two-column text (126 images) is identical before
and after. Of 150 held-out images, 61 changed: 167 fewer and 12 more edits; the
added edits are single characters read differently from the enlarged copies.
The held-out corpus renders 30 paragraphs of the reference's Chinese guides
and changelog (disjoint from R45's text) in Heiti TC Light and Medium, Hiragino
Sans GB, Songti SC and Arial Unicode at 12 points: 300 and 150 DPI, a
scan-like copy (as in R45) and 96 and 72 DPI screenshots. The full pages are
two A4-proportioned pages in Hiragino Sans GB and Songti SC at 150 and 96 DPI;
the Traditional check is eight short paragraphs written for it, in Heiti TC
and Songti TC. All are synthetic renders, not scans or photographs.

Whole conversions (`--ocr --no-llm`, one at a time, after a warm-up, median of
five runs of the mean per image; the quieter of two alternating rounds) are
unchanged for English (153 ms) and within 5 ms for ordinary Chinese images
(207 to 236 ms) and a full 150 DPI page (656 to 661 ms). Images enlarged for a
second reading take longer: held-out 96 and 72 DPI paragraphs 202 to 268 ms
and 191 to 269 ms, a full 96 DPI page 584 to 1,284 ms. The other round, on a
busier machine, showed the same pattern with more spread.

Tried with a Swift probe of the same Vision request and not adopted: other
settings (language correction off, `zh-Hans` with `en-US`, request revision 3:
identical output; `en-US` first: unusable); enlarging text that is not small
(no gain at 150 and 300 DPI); reading every line again in its own region (the
held-out error rate rose from 3.1% to 8.7%); voting across three readings at
different scales (2 to 3 fewer edits of 1,746 for three times the time);
restoring the case of Latin words inside Chinese lines from an English reading
of their line (6 and 5 fewer edits on R45 and held-out text, 4 more on the full
pages, and up to 0.8 seconds more per page); normalizing full-width symbols
such as ＃ to ASCII (the held-out text itself uses the full-width ／); and
lexical replacements such as 己 to 已 outside 自己 (two cases in R45, none held
out). Vision never put a space between two Han characters; the spaces it drops
between Han and Latin letters are the aid above. Remaining Chinese errors are mostly lookalike characters
(界/果, 器/嚣, 已/己), 的 read as another character, and Latin letters inside
Chinese lines (`l`/`I`, the case of `o`, `s` and `c`, `--` read as `-`).

Japanese and Korean, and the reading of missed text, were measured on the
same system with release builds of `df40ecc` (before, the r19 build, 21,748,672
bytes) and with these aids (21,748,672 bytes; 4,544 bytes more machine code,
absorbed by segment alignment). The text is 30 Japanese and 30 Korean
paragraphs written for this check: kanji, hiragana and katakana with small
kana, ー, Latin words and digits; Hangul with word spaces, a few Hanja in
parentheses, Latin words and digits. Japanese is drawn in Hiragino Sans W3
and W6, Hiragino Mincho ProN, Hiragino Maru Gothic ProN and Arial Unicode,
Korean in Apple SD Gothic Neo Regular and Bold, AppleMyungjo, AppleGothic and
Arial Unicode, with the same sizes and scan-like copy as the held-out Chinese
text, two full pages each, and a 60 DPI copy (10 pixels per em) of these and of
the held-out Chinese paragraphs. The reference's Korean model is not installed
in its environment (RapidOCR would download it), so Korean has no reference
score.

| Corpus / variant | Reference | Before | After |
|---|---:|---:|---:|
| Japanese 300 DPI / 150 DPI / scan-like | 0.42% / 0.38% / 0.38% | 0.14% / 0.05% / 0.05% | 0.14% / 0.00% / 0.05% |
| Japanese 96 DPI / 72 DPI | 0.33% / 0.99% | 0.33% / 0.42% | 0.19% / 0.23% |
| Japanese full pages 150 / 96 DPI | 1.34% / 0.53% | 0.11% / 0.18% | 0.07% / 0.00% |
| Korean 300 DPI / 150 DPI / scan-like | not run | 0.81% / 0.81% / 0.76% | unchanged |
| Korean 96 DPI / 72 DPI | not run | 7.78% / 4.27% | 0.97% / 1.19% |
| Korean full pages 150 / 96 DPI | not run | 0.93% / 0.85% | 0.89% / 0.85% |
| 60 DPI Chinese / Japanese / Korean | not run | 11.27% / 1.13% / 1.94% | 9.83% / 0.85% / 1.78% |

Every image of the corpora above (English, numbers, two columns, Chinese,
Traditional Chinese: 328) is identical before and after. Japanese text changed
in 14 of 154 images: 15 fewer edits and one more (春 read as 音 from an
enlarged 72 DPI copy); the recovered characters are kanji such as 箱, 類, 確, 環
and 資. Before, Vision returned no text for 3 of the 60 Korean 96 and 72 DPI
paragraphs; they are now read, apart from their Hanja with one error (a
missing period); one dropped syllable (릉) was recovered on a page. At 60 DPI
one Chinese paragraph that read as empty is now read; Japanese changed in 16
images (10 fewer edits, 4 more) and Korean in 3: three dropped syllables were
recovered, and in one of these images a syllable was inserted where 올 had been
dropped but read as 윤. Most remaining Korean errors are the Hanja: without
them, Korean text has 18 errors in about 14,000 characters before and after.
With word spaces counted, Korean error rates move the same way (96 DPI 7.64%
to 0.76%). Judged by the first readings' boxes, recovery reads a region again
in 12% of the Japanese and 5% of the Korean images (21% of the held-out
Chinese ones). On 55 text-free pictures (macOS's own desktop and account
pictures, a blank page and noise) the Chinese and Korean output is identical
before and after.

Whole conversions were timed interleaved (`--ocr --no-llm`, one at a time
after a warm-up, before and after alternating three times per image, median
per image), while other builds ran on the machine. English, Chinese, and
Japanese and Korean 150 DPI paragraphs are unchanged within 4 ms (147, 250 to
254, 209 and 210 ms). Images read again enlarged take 1.3 to 1.9 times as
long: Japanese 96 and 72 DPI paragraphs 205 to 275 ms and 191 to
268 ms, a full Japanese 96 DPI page 669 to 1,294 ms, and the Korean paragraphs
that were read as empty 138 to 237 ms and 196 to 299 ms. Recovery adds 65 and
79 ms to full 150 DPI Japanese and Korean pages (731 to 796 ms, 587 to 666 ms).
A text-free picture read as Chinese, Japanese or Korean costs the fast reading
more: 5 ms for a blank 1600×1200 page, 16 ms for a 3000×3000 photograph,
28 ms for a 4000×4000 wallpaper and 69 ms for a 4000×3000 noise picture (533
to 603 ms); English is unchanged.

Tried and not adopted: enlarging Korean (Hangul errors 18 to 25), enlarging
Japanese at 150 DPI (no change), a 20-pixel threshold for Japanese (2 more
errors), wide boxes from 1.4 (no change) or 1.8 times (3 more errors), and
counting only Han characters in Japanese lines (18 instead of 15 errors). For
Chinese: reading the region again without language correction, letting a wide
Latin letter or punctuation box be a suspect (no change in errors on any
corpus), and wider regions (padding of 0.3 or 0.6 median widths, or six
neighbours: on the held-out text 2 to 3 fewer and 4 to 8 more errors).

The spaces were measured with the builds of [table cells](#turned-pages-code-numbers-and-table-cells)
on the same system, by whole conversions without `ocr.lang` (and with `zh`,
which reads the same). The error rates above remove whitespace and do not
change; with spaces kept (runs of whitespace as one space), they fall from
4.06% to 2.62% on the held-out Chinese text (150 images; 961 instead of 765 of
its 1,091 spaces), from 3.91% to 2.40% on the full pages, and from 2.39% to
1.22% on R45 Chinese; the Traditional check has no such junctions. On a set
written for this check (three lines of Chinese with Latin words and digits, in
Arial Unicode, Hiragino Sans GB, Songti and STHeiti at 20, 28 and 40 pixels,
with and without spaces), the 12 unspaced images read as before, without a
space added, and the nine spaced ones that the default reads as Chinese read
99 instead of 59 of their 108 spaces (6.64% to 1.53% with spaces kept; the
other three read as English, a limit of the default language above).
`i-cjk-300.png` (R45's `cjk-04`, whose text is `在 CLI JSON、报告`) keeps its
spaces and still reads `CLIJSON`. The measurement decodes the image read once
per reading that has such a junction: Chinese paragraphs take the same time
as before within 1 ms on average, full pages 19 ms more
([timing](#turned-pages-code-numbers-and-table-cells)).

## The portable engine (Windows and Linux)

Windows and Linux read images with PaddleOCR's PP-OCR models, run in the
conversion process by the pure-Rust [`tract`](https://github.com/sonos/tract)
inference engine (`tract-onnx` 0.23.8): no Python, ONNX Runtime, Tesseract or
system service. A macOS build has it only with the `portable-media` feature
(`cargo build -p markitai-cli --release --features portable-media`); Vision
stays that build's default, the portable engine is chosen when Vision cannot
run (macOS before 11, or an x86_64 build under Rosetta), and
`MARKITAI_OCR_BACKEND=paddle` or `vision` chooses one explicitly (any other
value is a configuration error; `vision` is an unsupported-capability error
where Vision is absent). Metadata records `ocr_path: paddle`, and `markitai
doctor` names the engine in its `rapidocr` check.

### Models

The model files are not part of the executable. A manifest compiled into it
([`models.json`](../crates/markitai-core/src/ocr/paddle/models.json)) names
each one with its official URL, size, SHA-256 and license (Apache-2.0;
provenance in [`licenses/paddleocr`](../licenses/paddleocr/README.md)): the
ONNX files RapidOCR 3.9.2 publishes on its ModelScope mirror, the same the
reference downloads. A model lives at
`MARKITAI_HOME/models/ocr/<name>-<first 8 digits of its SHA-256>/<file>`.

| Role | Model | Bytes |
| --- | --- | ---: |
| Text detection | PP-OCRv6 small (`ppocrv6-det-small`) | 9,929,594 |
| Line direction (0° or 180°) | PP-OCR mobile v2.0 classifier (`ppocr-cls-mobile-v2`) | 585,532 |
| Recognition, default and Latin/Chinese/Japanese languages | PP-OCRv6 small multilingual (`ppocrv6-rec-small`) | 21,234,383 |
| Recognition, Korean | PP-OCRv5 mobile (`korean-ppocrv5-rec-mobile`) | 13,488,748 |
| Recognition, other scripts | PP-OCRv5 mobile: Arabic, Thai, Greek, East Slavic, Cyrillic, Latin, Devanagari, Tamil, Telugu | 7.8–8.1 MB each |

`markitai doctor --fix` installs the files the selected portable engine and
configured `ocr.lang` need: the default set above is 45.2 MB; an explicit
language needs the detector, classifier and its recognizer. It names each file
installed. A normal OCR downloads missing models as needed, with a notice once
per process on standard error (`Local OCR: downloading …`). It does not
necessarily download all four default files before reading its first image.
The default macOS Vision engine does not inspect or install portable models.

`markitai doctor` checks paths, size and SHA-256 with bounded, no-follow reads;
it creates no model state, downloads nothing and does not parse or run ONNX.
A verified file is not a recognition-accuracy test. The four model states are:

| State | Ordinary OCR | `doctor --fix` |
| --- | --- | --- |
| Missing | Downloads the model when needed; an offline error gives its URL, size, digest and manual path. | Installs it without overwriting a file that appeared meanwhile. |
| Ready | Verifies bytes when loading the model; graphs are reused within the process. | Keeps it. |
| Corrupt | Does not overwrite it; required-model failures name `doctor --fix`. | Can replace a safe damaged managed file after verifying the new size and digest. |
| Unsafe | Rejects an unsafe path in the selected model set before loading or downloading. | Refuses it and names the path reason before downloading. |

Unsafe entries include symbolic links, Windows junctions/reparse points,
special files, multiply linked files, foreign ownership and non-private managed
model directories/files. The Markitai home container may have ordinary readable
permissions such as 0755 on Unix, but must belong to the current user and not
be writable by others; its managed model descendants must be private. These
checks do not require every system ancestor to be private.

Privacy follows the existing platform contract: Unix checks owner and permission
bits; Windows checks the process user's SID (or its token's default owner) and
inherits the parent ACL. It does not tighten or audit every Windows ACL entry.
Use a private user directory for the managed home; model hashing and identity
checks are separate from filesystem access permissions.

Downloads use the page-fetching proxy settings and HTTPS-only redirects. Under
a stable, validated installation lock, repair observes the whole selected set
again before the first request. Each download is staged beside its destination,
verified against its published size and SHA-256 both as received and from the
staged file, and rechecks directory, lock and named/held file identities before
publication. A missing file is published without clobbering; explicit repair
atomically replaces only a safe private, current-user-owned, single-link
managed model. The old damaged file stays open during replacement, including
on Windows. File/directory synchronization and a final model verification follow
publication. Failed download or verification keeps the existing damaged model;
stage removal is best effort. If synchronization fails after publication, the
replacement may already exist: run `doctor` again to inspect its actual state.
The optional Korean default reading can fall back to the first reading if its
model cannot load or infer; it still never repairs corruption implicitly.

### How an image is read

1. **Detection.** The image is scaled so that its long side is at most 1,600
   pixels (the measured quality and memory tradeoff below), and a small image (short side under 400
   pixels) is enlarged up to 1.5 times; a strip eight times wider than tall is
   padded above and below with its border color. The detector returns a map of
   text probability; its connected regions above 0.3 become rotated rectangles,
   kept when their mean probability is at least 0.5 and grown back to the
   text's extent (by area × 1.6 / perimeter, as the model was trained).
2. **Cut-out.** Each region is cut out of the original pixels as an upright
   rectangle (a perspective resampling); one at least 1.5 times taller than wide
   is text running down and is turned a quarter. The classifier flags lines it
   finds upside down; a flagged line is read both ways and the more confident
   reading kept (it flags a few upright lines of ordinary prose).
3. **Recognition.** Each line is scaled to 48 pixels high on a width in steps
   of 32 pixels (at least 320) and read one line at a time, on up to eight
   threads per image. Extra helper threads share a process-wide processor-count
   budget; the calling conversion threads also participate, so this is not a
   limit on the total number of runnable threads. The recognizer's
   per-column probabilities collapse into text (repeats merged, blanks dropped)
   with the model's own dictionary; a line's confidence is the mean of its
   characters' probabilities, and lines below 0.5 are dropped, as the reference
   does. Right-to-left recognizers' text is put in reading order.
4. **Layout.** The lines then take the steps Vision's do: [turned
   pages](#turned-pages-code-numbers-and-table-cells) (from each line's
   direction), columns, rows and paragraphs, code fences and mended zeros.
   Character positions and image gaps preserve spaces between Chinese and Latin
   text. Vision's enlarged second readings, recovered characters and table-cell
   readings are not applied; lone cells are kept.

### Languages

| `ocr.lang` | Recognizer |
| --- | --- |
| `en` (default) | multilingual, then Korean for the lines it cannot read ([below](#the-portable-default-language)) |
| `en-US`, `zh`, `zh_cn`, `cn`, `ch`, `zh-Hans`, `zh_tw`, `cht`, `chinese_cht`, `zh-Hant`, `ja`, `jp`, `japan`, `ja-JP`, `fr`, `de`, `es`, `it`, `pt`, `vi`, `tr`, … (the reference's PP-OCRv6 list) | multilingual only |
| `ko`, `korean`, `ko-KR` | Korean |
| `ar`, `arabic`, `fa`, `ur`, `ar-SA` | Arabic |
| `th`; `el`; `ta`; `te` | Thai; Greek; Tamil; Telugu |
| `ru`, `uk`, `be`, `eslav`, `ru-RU` | East Slavic |
| `cyrillic`, `bg`, `mk`, `kk`, `ky`, `mn`, `tg` | Cyrillic |
| `latin` | Latin (PP-OCRv5) |
| `hi`, `mr`, `ne`, `devanagari` | Devanagari |

Case, surrounding whitespace and underscores are normalized as for Vision; a
region or script after a known language is accepted (`pt-BR` reads as `pt`).
Any other value fails with an error that lists these. Vision's spellings are
accepted so that one configuration works on every system.

### The portable default language

The multilingual recognizer reads Latin script, Simplified and Traditional
Chinese and Japanese, the scripts the reference's default reads. For a line of
Hangul it returns nothing, a doubtful reading, or a confident fragment (the
line's final period alone). Each line it is not sure of, below 0.9 confidence,
empty, or with fewer than 0.4 characters per line height of its length, is
read again from the same cut-out with the Korean recognizer, whose reading
replaces it when at least a third of its letters are Hangul and it is more
confident. If the optional Korean model cannot load or its inference fails,
the first reading stands. When half the lines or more remain unread and at
least one is shaped like text, the conversion keeps confident lines and warns
that the default could not confidently read all text. Possible causes include
an unsupported script, low image quality or an unavailable optional Korean
model; the warning does not identify the cause with certainty. It lists
`ocr.lang` choices and `markitai doctor --fix`. A blank image or noise with no
recognized text receives the ordinary no-text warning; the partial-reading
rule does not treat a table of short cells as unsupported prose.

### Measured quality

The frozen r4 release CLI reran 636 whole conversions on macOS 27.0.1,
Apple M5 Max, with `portable-media`, `MARKITAI_OCR_BACKEND=paddle`, no explicit
`ocr.lang`, isolated `HOME`/`MARKITAI_HOME` and verified models. Every conversion
succeeded; normalized text, edit counts and number checks matched the
previous 636-image check. These are synthetic rendered corpora from the
sections above, including degraded variants, not a survey of real scans or
photographs. Character error rate is edit distance over ground-truth length,
whitespace removed for Chinese, Japanese and Korean and collapsed for English.
The portable column below is confirmed by r4; Vision and reference columns are
previously recorded comparisons, not new r4 runs. Parallel quality-run wall
times were collected while other work was active and are not performance data.
See the [validation record](validation/portable-ocr-w2.md) for binary identities,
methods and platform coverage.

| Corpus / variant | Portable engine (r4) | Vision (earlier) | Reference (earlier) |
|---|---:|---:|---:|
| English prose 300 / 150 DPI / scan-like | 0.01% / 0.03% / 0.08% | 0.04% / 0.11% / 0.07% | 5.82% / 3.73% / 4.36% |
| Numbers 300 / 150 / scan-like | 0.00% / 0.03% / 0.58% | 0.00% / 0.00% / 0.00% | 0.00% / 0.03% / 0.87% |
| Number tokens exact (of 298 each) | 298 / 296 / 270 | 298 / 298 / 298 | 298 / 296 / 259 |
| Two columns 300 / 150 / scan-like | 0.03% / 0.05% / 0.11% | 0.16% / 0.40% / 0.35% | 0.03% / 0.11% / 0.05% |
| R45 Chinese 300 / 150 / scan-like | 0.17% / 0.52% / 0.52% | 0.69% / 0.86% / 1.37% | 0.34% / 0.52% / 0.52% |
| Held-out Chinese 300 / 150 / scan-like | 0.32% / 0.60% / 1.28% | 1.28% / 1.16% / 1.52% | 1.60% / 2.17% / 2.57% |
| Held-out Chinese 96 / 72 DPI | 0.92% / 1.20% | 1.93% / 3.21% | 0.96% / 1.68% |
| Full Chinese pages 150 / 96 | 0.22% / 1.34% | 1.16% / 2.06% | 1.92% / 2.89% |
| Traditional Chinese 150 / 96 / 72 | 0.00% / 0.30% / 1.81% | 0.00% / 0.30% / 3.93% | not run |
| Japanese 300 / 150 / scan-like | 0.28% / 0.38% / 0.33% | 0.14% / 0.00% / 0.05% | 0.42% / 0.38% / 0.38% |
| Japanese 96 / 72 | 0.33% / 0.66% | 0.19% / 0.23% | 0.33% / 0.99% |
| Japanese full pages 150 / 96 | 0.49% / 0.53% | 0.07% / 0.00% | 1.34% / 0.53% |
| Korean 300 / 150 / scan-like | 0.97% / 0.92% / 2.21% | 0.81% / 0.81% / 0.76% | not run |
| Korean 96 / 72 | 1.13% / 2.70% | 0.97% / 1.19% | not run |
| Korean full pages 150 / 96 | 1.08% / 1.20% | 0.89% / 0.85% | not run |

The 150 Korean paragraph images have 1.59% character error as a whole.
Linux x86-64 in the Rosetta-translated Ubuntu guest and Windows 11 ARM64 in
UTM each ran 16 real CLI conversions: eight selected images under the default
and the same eight with explicit languages. All succeeded; default normalized
text matched the frozen macOS portable readings, and inputs/models were
unchanged. This is platform smoke/fixture coverage, not a rerun of all 636
images or a Linux/Windows performance comparison.

Earlier supplementary WIP checks, **not rerun on the final r4 binary**, reported:

| Earlier check | Portable engine | Vision |
| --- | ---: | ---: |
| 60 DPI Chinese / Japanese / Korean | 2.33% / 0.80% / 2.86% | 9.83% / 0.85% / 1.78% |
| Rendered tables (46) | 0.02% (651 of 652 numbers) | 0.91% |

The following explicit-language corpus, spacing and language-inventory results
also belong to those earlier WIP checks; they are not final-r4 acceptance:

With `ocr.lang` written (`zh`,
`zh_tw`, `ja`, `ko`) every image of these corpora reads with the same error
as under the default. Kept spaces (runs of
whitespace as one space): R45 Chinese 0.58% (Vision 1.22%), held-out 1.33%
(2.62%), full pages 1.14% (2.40%); the spacing set's spaced lines 0.00% with
144 of 144 spaces (Vision 1.53%), its unspaced lines 0.00% with no space
added. Of the inventory's language probes, English, French, German, Spanish,
code, a table, Chinese, Traditional Chinese, Japanese and Korean read
correctly, and so do lines mixing English with Chinese, Japanese or Korean,
which Vision's default reads as English; Arabic, Hebrew, Russian, Thai and
Greek warn (Greek keeps one partly read line), as do a line of mathematical
symbols (kept as read); a blank page and noise warn that no text was found.

### Time and size

The frozen r4 timing check used eight particular 150-DPI images on macOS
27.0.1 arm64, Apple M5 Max, Rust 1.99.0, release CLI builds. Each engine had
one warm-up conversion per image, followed by five paired conversions with
alternating engine order: 96 separate CLI processes in total. Runs were serial,
with other local/guest builds and benchmarks paused; models and filesystem
bytes were warmed. Each new process still pays model verification, parsing and
planning. The table is the median of the five measured whole CLI conversions;
RSS is the largest process peak across those five (`time -l`), in MiB.

| Selected image | Portable median ms | Vision median ms | Portable peak RSS MiB | Vision peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| English prose | 729.53 | 154.76 | 400.00 | 61.48 |
| Numbers | 643.69 | 150.02 | 463.81 | 60.91 |
| Two columns | 904.77 | 175.96 | 507.25 | 65.20 |
| Held-out Chinese | 686.61 | 273.73 | 253.94 | 106.05 |
| Full Chinese page | 2551.38 | 844.69 | 1119.81 | 201.30 |
| Japanese | 702.48 | 366.55 | 331.02 | 140.39 |
| Korean | 1170.71 | 417.65 | 503.47 | 130.58 |
| Traditional Chinese | 648.91 | 259.31 | 257.66 | 101.98 |

These eight samples do not establish full-corpus latency, batch throughput or
Windows/Linux performance. The first Vision warm-up took **23.53 seconds**;
it is recorded separately and excluded from the warmed medians. First use of a
new executable/system recognition cache can have a substantial cold cost
([Vision cache details](#validation-fixtures)). The portable English warm-up
was 0.75 seconds with models already installed: it does not include the initial
45.2 MB model download. Korean's default can pay for a second recognizer.
Detection admits at most two images at once within a process; the RSS values
above are observations, not a memory limit or a guarantee for larger pages.

Retained release executables have these actual sizes; their full SHA-256 and
source/compiler evidence are in the [validation record](validation/portable-ocr-w2.md):

| Build | Bytes | Comparison |
| --- | ---: | --- |
| macOS arm64 default baseline before W2 | 22,443,136 | Frozen pre-W2 release |
| macOS arm64 default (Vision) r4 | 22,459,664 | +16,528 bytes against that baseline |
| macOS arm64 `portable-media` r4 | 32,291,152 | Includes portable PDF rendering and OCR |
| Linux x86-64 r4 | 41,495,120 | Built/run in the Rosetta-translated Ubuntu guest |
| Windows 11 ARM64 r4 | 40,787,968 | Built/run in the UTM guest |

The macOS portable/default difference includes the portable PDF renderer and
OCR together; it is not an isolated OCR size delta. Model weights are external
to all these executable sizes. Windows x86-64 has been cross-type-checked,
including with Rust 1.92.0, but no native x86-64 executable size is reported here.
The Rust 1.92.0 workspace/all-targets/locked checks passed for macOS default,
macOS portable and Windows x86-64 using the existing cross-check driver;
the last checks types, not native linking or execution. Native Linux and Windows
ARM64 tests, strict Clippy, release builds and the 16-conversion OCR fixtures
have separate successful evidence. These are W2 candidate checks, not release
packaging or final product acceptance.

### Limits

- On the eight warmed macOS samples above, the portable engine took about
  0.64–1.17 seconds for paragraphs/columns and 2.55 seconds for the selected full
  Chinese page, versus 0.15–0.42 and 0.84 seconds with Vision. Peak RSS was higher.
  Other images, cold caches and concurrent conversions can differ substantially.
- Korean Hanja are not read by the Korean model, which has no Han characters.
- On the synthetic scan-like number set, some spaces between numbers and words
  are dropped (`+1(584)`, `2031target`): 270 of 298 number tokens are exact,
  against all 298 at 300 DPI.
- The default does not reliably read Cyrillic, Greek, Arabic, Thai or Devanagari:
  set `ocr.lang`. (Vision's default reads Cyrillic.) A partial-read warning can
  also result from low image quality or an unavailable optional Korean model.
- The 636-image final quality rerun and paired timing were on Apple silicon.
  Linux and Windows ARM64 have the native build/test and 16-conversion coverage
  described above; there is no comparable full-corpus or fair performance run
  on those guests or on physical x86-64 hardware.

## Validation fixtures

The authored [English PNG](../crates/markitai-core/src/ocr/fixtures/english.png)
contains three lines drawn from simple bitmap glyphs, without textual PNG
metadata. Its independent [expected text](../crates/markitai-core/src/ocr/fixtures/english.txt)
is `MARKITAI OCR`, `LOCAL TEXT ONLY` and `2026`. The macOS test requires every
token, valid confidence and at least three line rectangles. The same test checks
a white image, undecodable input and an unavailable language. Pure tests cover
language aliases, bounds, alpha handling, row/paragraph assembly and malformed
geometry. These are focused checks, not an OCR accuracy corpus or a claim that
results match RapidOCR on arbitrary documents.

Two macOS tests draw sentences adapted from this repository's Chinese changelog with
the in-process SVG renderer in Hiragino Sans GB: at 13 pixels per em the text
must be read exactly from an enlarged copy with rectangles in the original
pixels, while 25-pixel Chinese and 13-pixel English are read once; at 25 pixels
a 的 that this system's recognizer drops must be recovered. Two more draw
sentences from the Japanese and Korean corpora: Japanese in Hiragino Sans is
read exactly from an enlarged copy at 16 pixels, and at 17 pixels a 環 that the
recognizer drops is recovered; Korean in Arial Unicode MS, which the accurate
recognizer misses entirely at 16 pixels, is read exactly from a copy enlarged
by the fast reading's line heights, with rectangles in the original pixels,
while at 15 pixels it is read once, at 11 pixels a dropped 돗 is recovered, and
a blank image stays blank. They depend on the installed recognizer, like the
English fixture. Pure tests cover each language's script and letters
(including small kana, ー and the iteration marks, and excluding the katakana
middle dot, half-width katakana, jamo and Hanja in Korean), the enlargement
threshold, factor and pixel limit (with Korean never enlarged for small text),
the Lanczos copy's size and rounding, the wide-box suspects and their regions,
and the anchored insertion, including ambiguous, other-script and mismatched
readings.

Three macOS tests cover the [default language](#the-default-language) with the
same drawn sentences and no `ocr.lang`: the Chinese sentences (13 and 25 pixels),
the Japanese ones (16 and 24 pixels) and the Korean ones (16 and 24 pixels) must
be read exactly, by Chinese, Japanese and Korean, and as the written `zh`, `ja` and
`ko` read them; a line of Chinese with three Latin words, for which `en-US`
alone returns confident Latin fragments, must be read whole; and English in
three, one and a short line must stay as `en-US` and `en_us` read it, Chinese
written with `en-US` must not be read, and a blank image must stay blank
without a warning. Pure tests cover the judgement of an English reading (sound,
short, doubtful, failed), the letters, share and strength that make a reading
text of a script, the kana that mean Japanese, the warning condition, and that
only a bare `en` is the default. They depend on the installed recognizer, like
the English fixture.

Pure tests cover the turned-page vote and the upright rectangles of each turn,
the zeros mended and the words, formulas and number ends kept, the table
holes found (and none in prose, two columns of prose, complete, two-row and
spanning tables), the order and context of their readings, which readings
fill a cell (not rules, long text or other alphabets, but a look-alike digit
in a column of numbers), which garbled numbers are read again and mended,
the fixed-pitch code run with its indentation (and receipts, prose and
proportional fonts left as text), the ink-free gap measured as a space on
light and dark grounds, the junctions measured, lines joined without a
space, the unread judgement of a table of numbers, the warning's language
list, and the enlarged copy of a region inside its margin. The scanned PDF
fixture's exact transcript (`2ø26` mended to `2026`) runs in the ordinary
conversion tests.

Renderer-entry tests compare the same fixture's normalized PNG bytes and Vision
observations with the encoded-image path. Additional checks reject zero-sized,
oversized and excess-storage RGB layouts without allocating a maximum-sized
image, and verify that ordinary RGB rows and colors remain unchanged.

The portable engine's pure tests (they run wherever it is built: Windows and
Linux, and macOS with `portable-media`) cover the detection layout (long-side
cap, small-image enlargement, padded strips, BGR input planes), the
probability-map regions (thresholds, eight-connected runs, rotated
rectangles and their growth, specks), the classifier's decision, the
dictionary, CTC collapsing and character positions, the right-to-left order,
the perspective cut-out and quarter turn, bucketed line inputs, the measured
Chinese–Latin space, the default language's sure, Hangul and unread rules,
the `ocr.lang` table (reference, Vision and region spellings, and the
error), the manifest (official HTTPS downloads, digests, licenses, the
default set) and installation against a loopback server (exact size and
digest, owner-only modes, nothing installed from a wrong, oversized or
unreachable download, the error naming `doctor --fix` and the manual path, a
changed file refused on load). An ignored test reads the authored English
fixture with the installed models, upright and upside down, and a blank
image (`MARKITAI_HOME=… cargo test -p markitai-core --features
portable-media --lib -- --ignored ocr::paddle`); earlier runs passed on macOS
arm64 and in the x86-64 Linux guest. Final-r4 whole-CLI quality and native
Linux/Windows ARM64 fixtures are recorded separately above. Installation
regressions cover same-length corruption, explicit repair, failed-download
preservation and stage cleanup, changed stage/lock identities and concurrent
installation; Windows tests also reject managed-directory junctions before a
request or external write. `doctor` tests distinguish ready, missing, corrupt,
unsafe and invalid-language states, without loading an ONNX graph.

The [frozen release check](validation/native-backends-round16.md) observed a
25.897-second first image OCR call and much shorter subsequent calls. The cause
is Vision compiling its recognition models for the device on first use: it
caches them under `~/Library/Caches/<executable name>/com.apple.e5rt.e5bundlecache/`
per system build, for the executable that ran: `markitai` for the CLI, the host
process (`node`, `python3.13`, a Go program's name) for a binding, and each test
binary under its hashed name. The cache is about 136 KB, lies outside
`MARKITAI_HOME` (Vision chooses it; neither `MARKITAI_HOME` nor `HOME` moves it)
and can be deleted at any time. The first OCR after installing
or updating Markitai, or after switching between two builds, takes about 25–45
seconds; later calls take about 130 ms per image. A process that may not write
that cache directory (a sandbox limited to its output folder) fails with
Vision's `missingError`. A separate small native CPU-only probe does not
justify changing the default compute policy. These observations are not an
OCR speed guarantee.

The command-line executables link Foundation, CoreFoundation, CoreGraphics,
ImageIO and Vision delay-initialized: from macOS 15, dyld initializes them on
first use instead of at launch, which saves about 1 ms in every process that
needs none of them (HTML, Office and text PDF conversions). OCR, HEIF/AVIF
decoding and PDF rasterization open their framework explicitly before first
use, about 1 ms more once per process; earlier systems initialize them at
launch as before. The language bindings link them the same way; see
[macOS system frameworks](bindings.md#macos-system-frameworks) for what each
host saves.

Implementation references: Apple's [text-recognition guide](https://developer.apple.com/documentation/vision/recognizing-text-in-images),
[recognition request](https://developer.apple.com/documentation/vision/vnrecognizetextrequest)
and [in-memory request handler](https://developer.apple.com/documentation/vision/vnimagerequesthandler/init(data:options:)),
plus the maintained [objc2 Vision bindings](https://docs.rs/objc2-vision/0.3.2/objc2_vision/).
