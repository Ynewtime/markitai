# iwork-rs

Read and write Apple iWork documents — **Pages**, **Numbers** and **Keynote** —
from Rust, with no Apple software involved.

All three apps share one file format. It has never been documented by Apple,
so [`FORMAT.md`](FORMAT.md) writes down what it actually is, derived from real
documents and checked by the tests in this repository.

```
cargo add iwork
```

```rust
// A spreadsheet from nothing: no template, no Apple software.
let mut doc = iwork::Document::new_spreadsheet("Sales", "Q1", 4, 2)?;
let mut q1 = doc.table_mut("Q1")?;                 // by name, sheet+name or id

q1.set_block("A1", &[vec!["Region", "Units"]])?;   // a block, in one pass
q1.set("A2", "Zürich")?;                           // a string is a string
q1.set("B2", 1_240)?;                              // a number is a number
q1.set("B3", 980)?;

// A real formula, registered in the calculation engine — Numbers recalculates
// it when a figure above it changes. The value is what the cell shows until it
// does: this crate writes formulas and evaluates none of them.
q1.formula("B4", "=SUM(B2:B3)", 2_220)?;
q1.format("B4", &Format::Number { decimals: Some(0) })?;
q1.column_width(0, Some(140.0))?;

doc.save("Sales.numbers")?;
```

Every refusal carries a **reason**, not just a sentence. Refusing precisely is
what this crate does instead of guessing, and a caller doing bulk work needs to
tell "skip this cell" from "stop":

```rust
use iwork::Refusal;

match table.set(cell, value) {
    Err(e) if e.refusal() == Some(Refusal::Merged) => continue,      // covered by a merge
    Err(e) if e.refusal() == Some(Refusal::OutOfBounds) => grow()?,  // table is too small
    Err(e) if e.refusal() == Some(Refusal::HoldsFormula) => leave(), // deliberately
    other => other?,
}
```

The sentence is unchanged and is still what `Display` prints; the reason is that
sentence as a value — `Merged`, `HoldsFormula`, `WrongSlot`, `NoDonorFormat`,
`Organised`, `Patched`, `NotDrawn`, `Ambiguous` and the rest.

**A sheet is not a grid.** This is where a Numbers document parts company with
the shape a spreadsheet library usually assumes: a sheet is a *canvas* holding
any number of tables, with charts, shapes and images beside them. `doc.sheets()`
reports the sheets and what is drawn on each, `sheet.tables(&doc)` narrows that
to the tables, and `doc.tables()` reports every table in the document whatever
holds it — which is also how a table on a Pages page or a Keynote slide is
reached. A table is named by a name that is unique, by its sheet *and* name, or
by its identifier: `numbers-pivot.numbers` has a `Sales` on each of its two
sheets, and a bare `"Sales"` is refused rather than guessed at.

Pages and Keynote are addressed the same way — by the thing, not by its object
id:

```rust
let mut body = doc.body_mut()?;                    // the Pages document body
body.append("A new paragraph")?;
body.replace(40..55, "different words")?;          // UTF-16 code units

let mut slide = deck.slide_mut(0)?;                // by position; ids are spelled
slide.title("Quarterly review")?;                  // the layout's title placeholder
slide.notes("The numbers are provisional")?;
slide.transition("dissolve")?;
slide.add_text_box("Aside", Frame { x: 100.0, y: 120.0, width: 600.0, height: 120.0 })?;
```

**A slide is not a page with a title slot.** What a slide can hold is decided by
the *layout* it is built on: a title is a placeholder that layout defines. A
deck made from nothing has a layout that defines none, so `slide.title(…)`
refuses by name and says why, rather than putting a text box on the slide and
calling it a title.

```rust
let mut doc = iwork::Document::open("Report.pages")?;
println!("{} document", doc.kind().as_str());      // "Pages"

for storage in doc.text_storages() {
    println!("{}: {}", storage.identifier, storage.text);
}

// Editing text remaps everything anchored into the storage: style runs,
// hyperlinks, list levels, anchored drawables, comment anchors. Indices are
// UTF-16 code units, and an edit that would split a surrogate pair or delete
// the character an object hangs off is refused by name.
doc.insert_text(6083, 12, "eingeschoben ")?;
doc.delete_text(6083, 40..55)?;
doc.set_text(6083, "A new headline")?;                                 // a full-range replace

// Text styles, by copy-and-adjust
for style in doc.text_styles() {
    println!("{} {} {:?}", style.identifier, style.kind.as_str(), style.name);
}
let kicker = doc.create_text_style(3712, "Kicker")?;                    // copy one that works
doc.set_text_style_property(kicker.identifier, style::property::FONT_SIZE,
                            Some(Value::Fixed32(18f32.to_le_bytes())))?;
doc.apply_text_style(6083, 0..8, kicker.identifier)?;                   // UTF-16 code units

doc.save("Report-edited.pages")?;
```

A new document is made two ways. From nothing at all — no Apple software
anywhere, which is what the first line of this file promises — for all three
apps:

```rust
let mut doc = iwork::Document::new(iwork::Kind::Pages)?;
doc.append_paragraph("Aus dem Nichts")?;
doc.append_paragraph("A second paragraph, and the first one keeps its style")?;
doc.save("Made.pages")?;                       // Pages opens it, and resaves it

// A spreadsheet, sized as you like, with every cell writable from the start.
let mut sheet = iwork::Document::new_spreadsheet("Sales", "Q1", 4, 3)?;
sheet.set_cell("Q1", 0, 0, iwork::table::CellValue::Text("Region".into()))?;
sheet.save("Sales.numbers")?;

let deck = iwork::Document::new(iwork::Kind::Keynote)?;   // one slide, one master
deck.save("Deck.key")?;
```

Thirteen objects go into the Pages document, 82 into the spreadsheet, 42 into
the deck — against 569, 590 and 1063 in the blank documents the apps themselves
write. The difference is measurement: a document each app made was reduced one
field at a time, and where deletion could not answer, the app was *asked* —
every one of them narrates its document loading to the unified log, naming the
message it cannot parse, the field it is missing and the method it cannot send.
FORMAT.md §14 has the method and everything it found. `iwork create pages
Made.pages` does the same from the shell, and `examples/` writes a report and a
spreadsheet from Rust data.

Or out of a template, the way the apps make one:

```rust
// A template bundle is a document package; what a new document needs is an
// identity of its own, and a document from a template gets a *whole* new one,
// lineage included. Two calls give two documents.
let doc = iwork::Document::from_template(
    "/Applications/Pages.app/Contents/SharedSupport/Templates/08_Journal_Newsletter/ISO.template",
)?;
doc.save("Newsletter.pages")?;
```

Either shape of package is read and written, and a document keeps the shape it
was in — `File > Advanced > Change File Type` writes a directory instead of a
ZIP, and `Package::write_as` converts on request:

```rust
let package = iwork::Package::read("Huge.numbers")?;         // file or directory
println!("{}", package.form.as_str());                        // "single file"
package.write_as("Huge-as-a-package.numbers", iwork::Form::Directory)?;
```

Tables are read the same way, and are cross-app — a Numbers sheet, a Pages page
and a Keynote slide all hold the same `TST` archives:

```rust
let doc = iwork::Document::open("Budget.numbers")?;
for table in doc.tables() {
    println!("{} — {}×{} on {:?}", table.name, table.rows, table.columns, table.sheet);
    for cell in table.cells() {
        println!("  r{} c{}  {}  [{}]", cell.row, cell.column,
                 cell.value.to_text(), cell.format);
    }
}
```

A chart carries its data **twice**, and the two are different questions:

```rust
for chart in doc.charts() {
    println!("{} on {}", chart.type_label(), chart.placement.as_str());
    // The private copy — what the chart draws, and all a Pages or Keynote
    // chart has.
    for series in chart.series() {
        println!("  {:?}: {:?}", series.name, series.values);
    }
    // The live references — Numbers only. The grid above is then a cache of
    // what these last evaluated to.
    if let Some(references) = &chart.references {
        for reference in &references.data {
            println!("  fed by {}", reference.to_text());   // Sales!B2:D2
        }
    }
}
```

**The private copy can be written, and a chart can be put where there was
none.** `Document::set_chart_data` rewrites the grid — new names, new numbers,
and a shape the chart did not have before, extra series taking their colour
from the theme's six, which cycle. A row that stays keeps the UUID that makes
it the same series through the edit; only a new one is minted, and a blank
stays a blank rather than shifting its row. Told to draw three series over four
categories where it had two over three, Keynote read the chart back and wrote
it out unchanged, and so did Pages.

`Document::add_chart` puts a **copy** of a chart the document already has on a
slide, a sheet or a page — a copy rather than an invention, because a dozen
objects of theme properties stand behind a chart and this crate decodes none of
them. What is copied and what is shared follows the archive's own distinction:
the *style* half of each pair is the theme's and is shared, the *non-style*
half carries this chart's own state and is copied and renumbered. Location
would not serve — a Keynote deck keeps its non-styles in the slide's stream and
Pages keeps them in `ObjectContainer`.

**A chart fed by a table is refused, both ways.** Writing numbers into a cache
of formulas makes the chart disagree with its table the moment Numbers
recalculates, and a copy of such a chart would claim to follow a table while
holding numbers of its own. Nothing here evaluates a `TSCE` formula, so nothing
here writes that cache.

## CLI

The crate ships a binary as well as a library, so `cargo install iwork` puts
`iwork` on your path. It is the tool the format was worked out with: it reports
what is in a document and, for text, changes it.

```
cargo install iwork          # or: cargo install --path .

iwork inspect   Report.pages              # package form, previews, components,
                                          # media, object census
iwork text      Report.pages              # every text storage, with its object id
iwork storages  Report.pages              # …and every attribute table each one carries
iwork links     Report.pages              # hyperlinks and smart fields, with their text
iwork set-text  Report.pages 6083 "…" out.pages
iwork insert-text Report.pages 6083 12 "…" out.pages   # at character 12
iwork delete-text Report.pages 6083 12 30 out.pages    # characters 12..30
iwork objects   Budget.numbers 2001       # every object of one message type
iwork dump      Talk.key 1                # one object, field by field
iwork check     Report.pages              # look for a broken object graph
iwork extract   Report.pages ./media      # embedded media, byte-identical
iwork roundtrip Report.pages out.pages    # decode and re-encode every object

iwork drawables Talk.key                  # every placed object: geometry, style,
                                          # media, non-destructive edit state
iwork media     Talk.key                  # every media file, its digest and its users
iwork set-geometry Talk.key 2652464 250 300 400 120 out.key
iwork replace-media Talk.key 2652622 new.png out.key
iwork add-text-box Report.pages "page 2" "Randnotiz" 380 120 160 90 out.pages
iwork add-shape Talk.key 2652176 ellipse "" 200 200 300 200 out.key
iwork add-image Report.pages "page 3" photo.jpg 380 500 120 90 out.pages
iwork add-table Report.pages "page 2" Preise 4 3 72 400 out.pages

iwork tables    Budget.numbers            # every table: size, headers, merges, geometry
iwork cells     Budget.numbers Zellarten  # every cell, with its type and data format
iwork cells     Budget.numbers 904769 --raw   # …and the cell record behind each one
iwork csv       Budget.numbers Zellarten  # one table as CSV
iwork organise  Budget.numbers            # sort rules, filters, categories,
                                          # pivots, highlighting, custom formats
                                          # — with each rule's condition as a formula
iwork formulas  Budget.numbers            # every formula: cell, text, cached value
iwork set-cell  Budget.numbers Zellarten B3 n:43 out.numbers
iwork set-cell  Budget.numbers Zellarten 2 1 n:43 out.numbers   # the same cell
iwork set-cells Budget.numbers Zellarten A2 rows.csv out.numbers  # a CSV block
iwork set-formula Budget.numbers Zellarten C9 "=SUM(B2:B8)" n:1234 out.numbers
iwork set-format Budget.numbers Zellarten B3 percent:1 out.numbers
iwork set-width  Budget.numbers Zellarten 0 210 out.numbers
iwork set-height Budget.numbers Zellarten 3 33  out.numbers
iwork insert-row Budget.numbers Zellarten 8 out.numbers   # an empty row before index 8
iwork insert-column Budget.numbers Zellarten 1 out.numbers   # …and a column
iwork delete-row Budget.numbers Zellarten 8 out.numbers      # and back out
iwork delete-column Budget.numbers Zellarten 1 out.numbers
iwork merge   Budget.numbers Zellarten B2 1 3 out.numbers    # B2:D2 as one cell
iwork unmerge Budget.numbers Zellarten B2 out.numbers

iwork charts    Budget.numbers            # every chart: type, placement, the data
                                          # it carries, and the table ranges it
                                          # follows — the two are not the same
iwork set-chart-data Talk.key 2654690 sales.csv out.key
iwork add-chart Report.pages "page 1" 3493939 sales.csv 100 100 400 300 out.pages

iwork slides    Talk.key                  # every slide: layout, placeholders and
                                          # their text, presenter notes, the
                                          # transition with its per-effect
                                          # parameters, builds; and for the show,
                                          # playback, soundtrack, recording and
                                          # live-video cameras
iwork layouts   Talk.key                  # the theme's slide layouts, and which
                                          # slides are built on each
iwork set-notes Talk.key 2652498 "…" out.key
iwork effects                             # all 44, by name and by identifier
iwork set-transition Talk.key 2652176 "object cube" 1.5 out.key
iwork add-build Talk.key 2652176 2652501 in out.key
iwork skip-slide   Talk.key 2652498 out.key   # leave it out of the show
iwork unskip-slide Talk.key 2652498 out.key
iwork move-slide Talk.key 2652498 0 out.key   # to position 0
iwork duplicate-slide Talk.key 2652498 out.key

iwork sections  Report.pages              # sections, their text ranges, page
                                          # numbering, headers and footers
iwork structure Report.pages              # mode, paper, page templates, threads,
                                          # contents lists, footnotes, columns,
                                          # change-tracking switches

iwork metadata  Report.pages              # the two plists, the identity, the build
                                          # history, locale, template, custom formats
iwork annotations Report.pages            # authors, comments and their anchors,
                                          # tracked changes
iwork add-comment Report.pages 1732539 5 12 "Prüferin" "Hier bitte prüfen." out.pages
iwork duplicate Report.pages copy.pages   # a copy with a *new* document identity
iwork new "/Applications/Pages.app/Contents/SharedSupport/Templates/08_Journal_Newsletter/ISO.template" \
          Newsletter.pages                # a document from a template bundle
iwork strip-previews Report.pages out.pages   # drop the stale thumbnails

iwork styles       Report.pages           # every text style, with its object id
iwork style        Report.pages 3712      # one style, field by field, and what uses it
iwork new-style    Report.pages 3712 Kicker out.pages
iwork set-style    Report.pages 3801 font-size=f32:18 out.pages
iwork set-style    Report.pages 3801 11.3=f32:18       out.pages   # the same field
iwork apply-style  Report.pages 6083 0 8 3801 out.pages
iwork delete-style Report.pages 3801 3712 out.pages    # 3712 replaces it
iwork paragraphs   Report.pages 6083      # ranges, list level, style and bullet
iwork properties                          # every named style property, and its evidence
```

## How it works

The format is four layers deep, and this crate gives you each of them:

| Layer | What it is | Module |
|---|---|---|
| 1 | package — a ZIP with every entry *stored*, or a directory of the same entries | [`package`](src/package.rs) |
| 2 | `Index/*.iwa` — raw Snappy blocks, 64 KiB each | [`iwa`](src/iwa.rs) |
| 3 | flat stream of length-delimited protobuf objects | [`iwa`](src/iwa.rs), [`pb`](src/pb.rs) |
| 4 | an object graph whose shape depends on the app | [`document`](src/document.rs), [`pages`](src/pages.rs) |

Apple does not publish the `.proto` definitions, and a numeric message type is
the only thing identifying a payload's schema. So this crate works at the
protobuf **wire level**: objects decode to fields and re-encode in place. Two
consequences worth knowing:

- **Nothing is lost.** An object this crate has no idea about is carried through
  untouched, so editing a headline cannot corrupt a chart.
- **Nothing is touched either.** `save` re-encodes only the streams whose
  objects actually changed; the rest keep their original bytes exactly. Editing
  one style in a 97-stream Numbers document rewrites one stream, and a save with
  no edits reproduces every entry byte for byte. That is not just cheaper — a
  save that re-compresses everything moves every Snappy block boundary, and then
  nothing in the file distinguishes the edit you meant from the noise you did
  not. `iwork` prints which streams it rewrote.
- **Names are advisory.** [`registry`](src/registry.rs) maps type numbers to
  names like `TSWP.StorageArchive`, tagged `Confirmed`, `Inferred` or
  `Unverified`. It feeds human-readable output only; a wrong name cannot
  break parsing.

## Text styles

A `TSWP.StorageArchive` holds no formatting. It holds *attribute tables* —
lists of `{character_index, reference}` entries, each run reaching to the next —
and the objects those references land on are the styles. Field 5 points at
**paragraph** styles, field 7 at list styles and field 8 at **character** styles
— the opposite of what the field order suggests, and of what this crate assumed
until a probe document settled it.

[`style`](src/style.rs) gives you CRUD over them, and it splits along exactly
that line:

|  | How it works |
|---|---|
| **Read** | enumerate styles, their names, their fields, and every run that uses them |
| **Create** | **copy** an existing style, allocate an identifier above `PackageMetadata` field 1, list the copy wherever the template was listed |
| **Update** | rewrite a field by path (`11.3`) or by name (`font-size`), or hand the decoded archive to a closure |
| **Delete** | re-point or drop the runs, unlist it, refuse if a reference would be left dangling |
| **Apply** | point a character range at a style, splitting the run table and restoring what followed |

Everything that decides *which text gets which style* works on the attribute
tables, whose shape is asserted by the test suite. There are twenty-two of them
and they are not all alike — `iwork storages` lists what a document has, and
[FORMAT.md](FORMAT.md) has the inventory with what each one is anchored to,
which is what decides where its entries go when the text under them changes.

What is *inside* a style is a weaker kind of knowledge, and the split matters:
a wrong name in the registry only prints wrong, while a wrong field number
writes wrong bytes into your document. So the fields are addressed by path, and
`iwork style` prints the tree with a path for every one — the numbers come from
the document in front of you. A handful have been pinned down by comparing 654
styles against names the app itself assigned, and those have names:

```
iwork style     Report.pages 2857                        # what is in there
iwork set-style Report.pages 2857 font-size=f32:18 out.pages
```

`iwork properties` lists them, with what backs each name:

```
  bold                     11.1         measured in an imported document
  underline-width          11.30        observed changing alongside a measured one
  outline-level            12.27        name only, not observed here
```

Most were established by experiment rather than by correlation. A document was
built in which every paragraph differed from a baseline in exactly one property
— 37pt, `#123456`, 175%, 17pt — and imported into Pages; diffing each resulting
style against the baseline's leaves one changed field per probe. All four probe
colours came back byte-exact, which is what turned the colour field from a guess
into a fact. `bold` and `italic` are *toggles*, independent of the font's own
weight.

Some properties come in pairs — the value, and a boolean saying it is
deliberately *none*. Removing a field means "inherit from the parent style";
setting the companion means "explicitly nothing". Those are different documents.

**A style does not have one text colour, it has up to four.** The font colour,
the fill drawn inside the glyphs, and the underline and strikethrough colours
that follow the text — Pages writes all of them together, and **the fill is what
gets drawn**. Setting only `red`/`green`/`blue` leaves the fill behind, and the
text renders in its old colour. Use `set_text_style_color`, or `iwork set-color`,
which writes every one the style keeps:

```
iwork set-color Report.pages 2857 0.85 0.10 0.10 out.pages
```

**Setting a property whose container is missing fails, deliberately.** A style
with no colour cannot simply be given one: a colour is `{model, r, g, b, a,
space}`, and a container this crate invents would hold only the channels it was
asked for. Pages crashes on opening such a document — confirmed, not theorised.
Get the container from a style that has one, with `create_text_style` or
`copy_text_style_property`, then change the channels on the copy.

Creating by copying is the same rule [`FORMAT.md`](FORMAT.md) gives for whole
documents, for the same reason: the Pages sample spends 313 objects on its
stylesheet, and a style that already works is a better starting point than a
synthesised one. A copy is listed in the stylesheet the template names — and
only there. Bare style references elsewhere can be positions rather than
memberships (a Keynote slide's five outline levels are exactly that), and adding
an entry to one of those corrupts it.

### Tables

A table's cells are the one part of the format that is not protobuf. They are
fixed-layout byte records concatenated into a `bytes` field, sliced by an array
of signed 16-bit offsets, and every record holds *keys* rather than content:
its text is a number pointing into the table's string list, and so are its
format, its style and its formula. `doc.tables()` resolves all of that;
[`FORMAT.md`](FORMAT.md) §Tables writes down the layout.

Three things are worth knowing before trusting a table reader, including this
one:

**A Numbers document has no text storages at all.** `iwork text` reads nothing
out of a spreadsheet whose cells the app reads 2711 values from. Pages and
Keynote tables are the other way round, and their cells point at
`TSWP.StorageArchive`s.

**A value without its format is only half the cell.** `0.25` shown as `25%`,
`19.99` shown as `CHF 19.99`, `TRUE` shown as a checkbox — and the difference
between a cell that *holds* a number and a cell the user *made* a number is one
bit in the record. Value and format are read together.

**The app is the oracle.** `tests/tables.rs` asks Numbers, through AppleScript,
for the value, the data format and the formula of every cell of every table of
three spreadsheets and compares them with what this crate decoded: 2943 cells,
all agreeing. Run it with `IWORK_APP_CHECK=1 cargo test`. It found real bugs —
the format model above is what survived it.

**Cells are addressed by index; everything layered on them is not.** Sort
rules, filters, categories, conditional highlighting and pivot tables name
rows and columns by UUID, because a sort or a filter moves an index and a UUID
survives it. `iwork organise` reads all of it — and the fixtures that exercise
it are documents made from Apple's own templates, because Numbers' scripting
interface has no command that sorts, filters, categorises, highlights or pivots
anything.

**A cell can be written, one at a time.** `iwork set-cell` puts text, a number,
a boolean, a date or a duration into a cell that already exists, and Numbers
opens the result and reports the new value. Three things make that safe rather
than merely possible. The record is **edited, never rebuilt** — the encoder is
the decoder's exact inverse on every record in the corpus, so a cell keeps
its style keys, its control definition, its conditional-highlighting keys and
the bytes nobody has decoded. The interned string and format lists are
**refcounted both ways**: a string another cell already holds is shared, and one
nobody points at any more is removed, which is what the app itself does — and
emptying a cell gives back **every** key the deleted record held. And the whole
write is **planned before a byte moves**, so a write that turns out to be refused
leaves the document byte for byte as it was, rather than half-applied.

Editing a number rewrites **one** of a Numbers document's 97 package entries.
Writing a cell the value it already holds rewrites none.

What it refuses, rather than writing something plausible: a formula cell (taking
a formula out means editing `TSCE`), a rich-text cell, a cell covered by a
merge, a row with no stored cells, an ambiguous table name (write by identifier
to say which), and any object it would rewrite that carries version patches.
A formula that *reads* an edited cell keeps its stale cached value — Numbers
recalculates it on open, so the app is right and a reader trusting the cache is
not.

**An empty row can be inserted, into a plain table.** `iwork insert-row` grows a
plain rectangular table held in one tile: the row count is bumped, the tile's
row indices and the row-header bucket shift down past the insertion, and the
`ColumnRowUIDMap` gains a fresh per-table-unique row UUID — rebuilt sorted by
UUID, the way the app keeps it. Numbers opens the result, shows one more row with
the new one empty, and reads every row below the insertion back with its value,
its data format and its control (a checkbox, a rating, a slider…) intact. The
new row is genuinely empty — it has no cell storage of its own — so filling it
needs the *first-cell-in-a-row* write `set-cell` does not do yet.

**A shift can cross a tile boundary, and now does.** A row's absolute index is
`tileid * 256 + tile_row_index`, so the last row of tile 0 moving down one
becomes the *first* row of tile 1: its `TileRowInfo` has to leave one object and
join another. The insert therefore gathers every row of every tile by absolute
index, shifts, and lays them back out into whichever tile each now belongs to.
Verified on a 300-row table this crate made — Numbers opened it, and wrote it
back with row 255 in the second tile where the insert had put it. A table that
*fills* every tile it has is still refused: that row would need a tile of its
own, a new object and a new component both.

Everything the insert cannot maintain safely is refused **by name**: a
categorised, filtered or pivoted one, a table with conditional highlighting,
hidden or collapsed rows or footer rows, a merge at or straddling the insertion,
and — the subtle one — any table whose formulas reference it at or below the
insertion point, where an unshifted `TSCE` reference would silently compute the
wrong answer. A whole-column reference is unaffected and allowed; a relative
reference that moves together with its host is allowed; a bounded range the
insertion would cross is refused.

**A column can be inserted too, and it is not a row turned sideways.** A row is
an object — a `TileRowInfo` of its own — so inserting one shifts whole objects
and the new row has none. A column is not an object at all: it is one entry in
*every row's* offset array, so `iwork insert-column` rewrites every row of every
tile, slicing each into its per-column records, opening a gap and laying it back
out. Which is also why it is not limited to a single tile the way the row insert
is: the work is per row, and a tile boundary is a row boundary — verified on the
301-row fixture, which has two. The offset array keeps the length it arrived
with, because Numbers pads it to 255 entries and that padding is what a reader
steps through, so a table already that wide is refused rather than losing a
column out of the back. Numbers opens the widened table, reports one more
column, reads every value back one column over — compared against *its own*
earlier reading, since the app prints `1.2345678E+4` where this crate says
`12345.678` — and writes the document out again. The refusals are the row
insert's along the other axis: a categorised table (a category *is* a column), a
pivot, a filter, conditional highlighting, hidden columns, a merge at or
straddling the insertion, and any formula whose reference would shift.

### Drawables, geometry and media

A drawable is anything placed on a page, a sheet or a slide — an image, a
shape, a text box, a line, a movie, a group, a table, a chart. `TSD` is
cross-app in the strongest sense: the type ids live in the *common* registry, so
one table of names serves all three apps.

```rust
let doc = iwork::Document::open("Talk.key")?;
for drawable in doc.drawables() {
    let mask = drawable.mask().and_then(|id| doc.drawable(id));
    let frame = drawable.frame(mask.as_ref());          // what the app reports
    println!("{} {} at {},{} {}×{} — {}", drawable.identifier,
             drawable.kind.as_str(), frame.x, frame.y, frame.width, frame.height,
             drawable.placement.as_str());
}
```

**The geometry is not at a fixed depth.** `super` is field 1 and is a
submessage, so a Keynote title placeholder is four levels of nesting before its
geometry and an image is one. Nothing here assumes a depth: the walk follows
field 1 until it finds a geometry, and every read and write goes through the
path it returns.

**What the app calls an object's rectangle is not what the archive says**, and
both corrections came from asking the app rather than reading a schema:

- **A masked image is reported as its mask.** The mask is a separate object
  whose geometry is in the *image's* coordinate space, so the rectangle is
  `image.position + mask.position` by `mask.size`. Pages reports 60 × 123,
  475 × 383 for a photo whose own geometry says 33.86 × 66.28, 511.86 × 466.13.
- **A rotated object is reported at the corner of its rotated bounding box**,
  and at its *unrotated* size. A 220 × 180 shape turned 30° at 100 × 100 comes
  back as 470 × 57, still 220 × 180.

`iwork drawables` prints the corrected rectangle, and `IWORK_APP_CHECK=1 cargo
test` compares every one of them with what Keynote and Pages report.

**Object styling is a separate object, and it inherits.** Fill, stroke,
opacity, shadow and reflection live in a `TSD.ShapeStyleArchive` — or a
`TSD.MediaStyleArchive`, which has no fill and therefore *numbers everything
one lower*, a difference that is silent if you get it wrong. Told to set a
shape to 50% opacity, Keynote wrote a new variation style carrying nothing but
the opacity and the reflection; everything else comes from its parent, so
resolution walks the chain.

**A drawable can be moved and resized, app-verified.** `iwork set-geometry`
takes the rectangle the app reports and converts it back. Three things travel
with a resize because the app moves them too: an *unmasked* media object's
`originalSize` (a masked image's is left alone — the app does not put the
picture's size there and the corpus does not agree on what it does put); a
shape's **path source**, whose natural size and every baked point Keynote
rewrites — a document with only the geometry changed opens with the app still
reporting the old size, and a curve's three control points all move, not just
the first; and a masked image's whole assembly, scaled by one factor so the
frame lands where it was asked to. Against the document Pages itself wrote for
the same resize, the mask comes out byte-identical and the image differs in the
last two ulps of one float.

**Media is refcounted, digested and easy to falsify.** A drawable never carries
pixels: it carries a reference into `TSP.PackageMetadata.datas`, whose entries
name files under `Data/` and carry a **raw SHA-1** of their bytes. Replace an
image in Keynote twice and the first replacement's registry entry and file are
both gone — nothing points at them any more.

`iwork replace-media` swaps the bytes, the digest, the byte length, the recorded
pixel size and every drawable's `naturalSize` and traced outline, and marks the
image as replaced — which is what Keynote does when it replaces one itself.
**And it refuses when it would be lying.** An image can carry a crop, a shaped
mask, an Instant Alpha path, tone adjustments, a stored thumbnail or other
cached renderings of the old pixels, or a traced outline of them; none of that
is in the new file and none of it can be recomputed. Swapping bytes underneath
produces a document that opens, reports the same geometry, passes every
structural check and draws the wrong thing — so the replacement is refused by
name instead. An *identity* mask is not an objection: its window is the whole
drawn picture at the origin — measured against the image's own geometry, not the
`originalSize` field the app fills with the mask window — which is what the app
installs when it replaces an image, and it hides nothing.

An honest limit, worth stating plainly: an app round trip proves the document
opens and that the picture is still where it was. It cannot prove the pixels
drawn are the new ones — nothing on a locked screen can see what is rendered.

**A drawable can be added where there was none.** `Document::add_text_box` and
`add_shape` put a box, an ellipse or a line on a Keynote slide, a Numbers sheet
or a Pages page:

```rust
let mut doc = iwork::Document::new(iwork::Kind::Pages)?;
doc.add_text_box("page 1", "Aus dem Nichts", (72.0, 300.0), (400.0, 100.0))?;
doc.add_shape("page 1", Outline::Line, "", (72.0, 700.0), (400.0, 0.0))?;
doc.save("Drawn.pages")?;
```

One archive serves all three apps; what differs is **who holds it**. A slide
owns its drawables and is named as the shape's parent, a sheet the same — and a
Pages page owns *nothing*: the page group in `TP.FloatingDrawablesArchive` names
the shape, the shape names no parent, and it carries the text-wrap archive
because Pages is the only app that flows text around anything. Three outlines,
because these are the three whose path can be written and checked: a rectangle
(with iWork's redundant closing `moveTo`), a line, and an ellipse drawn as four
Bézier arcs.

**Nothing is invented.** The style the box is drawn with, its stylesheet, its
paragraph style and its list style all come from the document it lands in — a
text shape already on it, or the theme's text-box preset — so a box added to a
themed deck looks like the theme, and a document that has none of them is
refused by name rather than given a style it never defined. Verified past
opening: Pages and Keynote both *resaved* a document with an added box, keeping
its geometry, its path and its words.

**A picture can be placed the same way.** `Document::add_image` copies the
bytes into the package, registers a `TSP.DataInfo` with their raw SHA-1 and
their length, and declares the identifier in the object's own
`data_references` — the last of which is what makes the app load the picture
rather than draw a hole. PNG and JPEG only, because the registry records the
pixel size and the drawable's `naturalSize` has to agree with it, and this
crate reads headers rather than decoding pictures; anything else is refused
rather than registered with a guess. Pages, Numbers and Keynote all open the
result, and Pages and Keynote resave it with the bytes intact.

**A table is a drawable too**, so it can be placed the same way:
`Document::add_table_at` puts one on a Numbers sheet, a Keynote slide or a
Pages page, and the whole difference between the three is the containment. It
still borrows the styles of a table the document already has — a document with
none is refused rather than given an invented table style — so this grows a
report that has a table, not a Pages file made from nothing. Pages reads the
new table's cells back and resaves the document with it still floating where it
was put.

One thing that fell out of asking Keynote: **a data reference and an object
reference are the same bytes**, `{1: identifier}`, and the two identifier
spaces overlap. Resaving a deck, Keynote gave a slide thumbnail the *data*
identifier 1041 while object 1041 was the image on that slide — so a structural
walk reads the thumbnail as a cross-component reference to the image. The
object's own `MessageInfo.data_references` is what settles it, and the
reference checks here subtract it first.

## What is verified

Everything below is asserted by `cargo test` when you supply fixtures.

| | Pages | Numbers | Keynote |
|---|---|---|---|
| Open, identify, decode every object | ✅ | ✅ | ✅ |
| Object streams survive re-encode byte for byte | ✅ | ✅ | ✅ |
| Components resolve to real streams | ✅ | ✅ (96 of them) | ✅ (29) |
| Media registry resolves | ✅ | — (no media in samples) | ✅ (33) |
| Text extraction | ✅ | ✅ | ✅ |
| Edit text, leave every other object alone | ✅ | ✅ | ✅ |
| All 22 attribute tables inventoried; an unknown field refuses the edit | ✅ | ✅ | ✅ |
| Attribute tables point at styles of the matching kind | ✅ | ✅ | ✅ |
| Entries increase, fit the text, and start at 0 | ✅ | ✅ | ✅ |
| Paragraph entries sit at paragraph starts — over all 901 bundled templates | ✅ | ✅ | ✅ |
| Insert, delete and replace a range; everything anchored moves with it | ✅ | ✅ | ✅ |
| **Ten edits Pages made, reproduced entry for entry** | ✅ | — | — |
| A paragraph created or destroyed keeps the bookkeeping exact | ✅ | ✅ | ✅ |
| An edit inside a surrogate pair, or over an anchor, is refused by name | ✅ | ✅ | ✅ |
| **The app reads back inserted, deleted and replaced text** | ✅ | ✅ | ✅ |
| Hyperlinks: read, and their target changed | — (none exist) | ✅ | — (none exist) |
| Smart-field run extents, terminated and unterminated | — | ✅ | — |
| List level and list style per paragraph | ✅ | — | — |
| A run resolved to its named style plus its local overrides | ✅ | — | — |
| Copy a style: one new object, text untouched | ✅ | ✅ | ✅ |
| Apply a style, leave every other stream alone | ✅ | ✅ | ✅ |
| A copy keeps the template's kind (named vs variation) | ✅ | ✅ | ✅ |
| Tables: names, sizes, header/footer counts, freeze flags | ✅ | ✅ | — (no fixture) |
| Cell values: text, number, boolean, date, duration, currency, rich text | ✅ | ✅ | — |
| Data formats and control cells (checkbox, rating, slider, stepper, pop-up) | ✅ | ✅ | — |
| Write a data format: number, percent, scientific, currency, date pattern | — | ✅ | — |
| A format goes in the slot the value uses; any other slot is refused | — | ✅ | — |
| **The app draws the written format** — percent, €, decimals, date pattern | — | ✅ | — |
| Write money: a currency *cell*, its amount and the currency's own format | — | ✅ | — |
| The record is the app's own: type 10, `extras 0x0802`, no number-slot key | — | ✅ | — |
| One format entry per currency, reused; a number written over money clears it | — | ✅ | — |
| **The app draws two currencies in one table** — `CHF 184300.00`, `€ 1234.50` | — | ✅ | — |
| Write a column's width and a row's height; the frame is left alone | — | ✅ | — |
| **The app reports both back exactly** | — | ✅ | — |
| Merged ranges | — (none) | ✅ | — |
| Every cell record consumed to the byte (every one in the corpus) | ✅ | ✅ | — |
| **Every cell agrees with the app** (2943 compared) | — | ✅ | — |
| Sort rules; filter sets with their rules and on/off switch | — | ✅ | — |
| Write a table's sort rules, and take them away | — | ✅ | — |
| Turn a filter off and on, and switch it between all and any | — | ✅ | — |
| **Switching a filter off does not un-hide its rows** — the hiding is stored | — | ✅ | — |
| Change what a conditional highlight compares against, in all four copies | — | ✅ | — |
| **The app keeps all three through a save of its own** | — | ✅ | — |
| Hidden rows and columns, with *why* (user vs filter) | — | ✅ | — |
| Categories: source column, groups, rows per group, SUM summaries | — | ✅ | — |
| Pivot tables: source, row/column/value fields, summary functions | — | ✅ | — |
| Conditional highlighting rules; custom cell formats | — | ✅ | — |
| A save leaves an organised document byte-identical | — | ✅ | — |
| Every cell record re-encodes to the bytes it came from | ✅ | ✅ | — |
| Version patches: the view state and a too-new chart carry them; no table archive does | ✅ | ✅ | ✅ |
| Every list key resolves, every refcount matches, every cell count adds up | ✅ | ✅ | — |
| A cell is named `"B3"` or `(2, 1)`, and reads back the same either way | ✅ | ✅ | ✅ |
| A text storage is edited through a handle, not an object id | ✅ | ✅ | ✅ |
| A slide is addressed by position or identifier | — | — | ✅ |
| A span is a range (`"B2:D2"`), a frame is named, a duration is a field | ✅ | ✅ | ✅ |
| Every refusal carries a reason a program can match on, not just a sentence | ✅ | ✅ | ✅ |
| **The app does not draw a page-layout document's body** — so it is refused | ✅ | — | — |
| A title the layout does not define is refused by name, not invented | — | — | ✅ |
| A sheet's drawables, and the tables among them, in the sheet's own order | — | ✅ | — |
| A table name two sheets share is refused, with both sheets named | — | ✅ | — |
| Write a cell: text, number, boolean, date, duration, empty | ✅ | ✅ | — |
| Give a row with no storage its first cell, in the shape the app writes | — | ✅ | — |
| **The app reads back a value in a row that had no storage at all** | — | ✅ | — |
| Write many cells in one pass: one decode per tile, list and bucket | ✅ | ✅ | — |
| A batch and the same single writes produce the same document, byte for byte | ✅ | ✅ | — |
| A refused cell leaves the whole batch unwritten | ✅ | ✅ | — |
| **The app reads back a block written across three columns and two rows** | — | ✅ | — |
| A written cell keeps its styles, format and undecoded bytes | ✅ | ✅ | — |
| Writing a cell what it already holds changes no byte | ✅ | ✅ | — |
| **The app reads back the written value** | ✅ | ✅ | — |
| Insert an empty row into a plain single-tile table; refuse the rest by name | — | ✅ | — |
| **The app reads back the extra row, empty, with the rows below unmoved** | — | ✅ | — |
| Insert an empty column, into any number of tiles; refuse the rest by name | — | ✅ | — |
| Every cell right of the insertion keeps its value *and* its data format | — | ✅ | — |
| **The app reads back the extra column, empty, and the values one over** | — | ✅ | — |
| A row inserted below a tile boundary crosses it; a full table is refused | — | ✅ | — |
| **The app resaves a cross-tile insert with the row in its new tile** | — | ✅ | — |
| A table made from nothing carries the UUID map an insert needs | — | ✅ | — |
| Delete a row or a column, with every reference it held given back | — | ✅ | — |
| A deleted line's cells leave every count in the document adding up | — | ✅ | — |
| A formula that names the line, or a range across it, refuses the delete | — | ✅ | — |
| **The app reads back the table with a row and a column gone** | — | ✅ | — |
| Merge cells: the range node is **byte for byte the app's own**, all four | — | ✅ | — |
| The covered cells are emptied through the cell writer, references and all | — | ✅ | — |
| **The app reports the two merged cells under one name** | — | ✅ | — |
| Pages mode: word processing vs page layout, and the app agrees | ✅ | — | — |
| Sections: name, text range, page numbering, background, switches | ✅ | — | — |
| **Every section's text agrees with the app, character for character** | ✅ | — | — |
| Three headers and three footers per section template, every time | ✅ | — | — |
| Page templates exist exactly in page-layout documents | ✅ | — | — |
| Linked text boxes: the thread, its storage and its boxes in order | ✅ | — | — |
| Table of contents: both settings archives, its rules and its entries | ✅ | — | — |
| Columns: equal and non-equal, as fractions that add up to one | ✅ | — | — |
| A page number's format is on the attachment, not on the section | ✅ | — | — |
| Footnotes: marks on `U+000E`, notes as kind-2 storages, made via the UI | ✅ | — | — |
| Bookmarks: nameless UUID archives; terminator entries are not bookmarks | ✅ | — | — |
| Write a header or footer; only the touched stream is rewritten | ✅ | — | — |
| **Pages opens the edited document, saves it, and the header is still there** | ✅ | — | — |
| Deleting a section break is refused by name | ✅ | — | — |
| Drawables: geometry, rotation, lock, parent, z-order, containment | ✅ | ✅ | ✅ |
| Shapes, text boxes, lines: path source and its natural size | ✅ | — | ✅ |
| Masked images: the crop, and the frame the app reports | ✅ | — | ✅ |
| Object styles: fill, stroke, opacity, shadow, reflection, inheritance | ✅ | ✅ | ✅ |
| Media registry: digest is the SHA-1 of the bytes, every stored file | ✅ | — | ✅ |
| Non-destructive edit state detected: crop, shaped mask, Instant Alpha, adjustments, derived renderings | ✅ | — | ✅ |
| Movies and live video: read and named (galleries, 3D and pencil are registry names only) | — | — | ✅ |
| Every geometry re-encodes to the bytes it came from | ✅ | ✅ | ✅ |
| **Every rectangle agrees with the app** | ✅ | — | ✅ |
| Move and resize a drawable; only the touched stream is rewritten | ✅ | — | ✅ |
| **The app reads back the moved rectangle** | — | — | ✅ |
| Replace an image's bytes; registry and drawables stay in step | — | — | ✅ |
| A replacement is refused when edit state would make it a lie | ✅ | — | ✅ |
| Add a text box, an ellipse or a line where there was none | ✅ | ✅ | ✅ |
| A page's box has no parent; a slide's and a sheet's do | ✅ | ✅ | ✅ |
| A shape with no size, or a container that is not there, is refused by name | ✅ | ✅ | ✅ |
| **The app reads back the words in an added box** | ✅ | ✅ | ✅ |
| **The app resaves a document with an added box, and keeps it** | ✅ | — | ✅ |
| Place a picture: bytes, registry entry, digest and declaration | ✅ | ✅ | ✅ |
| A picture whose pixel size cannot be read is refused | ✅ | ✅ | ✅ |
| A new object claims none of its neighbour's media | ✅ | ✅ | ✅ |
| **The app resaves a document with an added picture, bytes intact** | ✅ | — | ✅ |
| Float a table on a Pages page, with every cell writable | ✅ | — | — |
| **The app reads the new table's cells and resaves it in place** | ✅ | — | — |
| Formulas: the AST, its 40 node types and 48 function ids in this corpus | ✅ | ✅ | — (no fixture) |
| Every formula archive re-encodes to the bytes it came from (every one in the corpus) | ✅ | ✅ | ✅ |
| Every formula validates field by field against the 15.3.1 schema | ✅ | ✅ | ✅ |
| The reference model: absolute/relative per axis, whole row, whole column | ✅ | ✅ | — |
| Number literals from their decimal128, not from the double beside it | ✅ | ✅ | — |
| Cross-table references resolve by identity — **proven by a renamed table** | — | ✅ | — |
| Write a formula from its text: operators, precedence, functions, ranges | — | ✅ | — |
| Every node matches the shape the app wrote for the same formula, byte for byte | — | ✅ | — |
| The written cell is registered in the engine, so the app recalculates it | — | ✅ | — |
| A table made from nothing carries the two `TSCE` owners a formula needs | — | ✅ | — |
| Both owners are indexed in the engine's tracker, by id and by reference | — | ✅ | — |
| **The app recalculates a formula in a document built from nothing** | — | ✅ | — |
| **The app prints the written formula back in its own spelling, and its value** | — | ✅ | — |
| Header-name references, with quoting, scoping and ambiguity | ✅ | ✅ | — |
| A stored `#REF!`, made by deleting a column a formula pointed at | — | ✅ | — |
| `LET`/`LAMBDA`: bindings, continuations, symbols — the 14.4 shape of fields 34–37 | — | ✅ | — |
| Filter and conditional-highlighting conditions read as formulas | — | ✅ | — |
| **Every formula matches the app's text, character for character** (273 of 273 outside pivots) | — | ✅ | — |
| Charts: type, placement, rectangle, series direction, 33 of them | ✅ | ✅ | ✅ |
| 22 of the 28 chart types, every 3-D family but the donut | — | ✅ | ✅ |
| The chart model found at extension 10000 of every chart drawable | ✅ | ✅ | ✅ |
| The private grid: row and column names, series, blank ≠ zero | ✅ | ✅ | ✅ |
| **Every value of an 18-chart zoo is the number the app was told to plot** | — | — | ✅ |
| Rewrite a chart's private grid, including its shape | ✅ | — | ✅ |
| A row that stays keeps its identity; a new one gets a new one | ✅ | — | ✅ |
| A blank stays blank, and does not shift the row | ✅ | — | ✅ |
| Copy a chart: theme styles shared, non-styles copied and renumbered | ✅ | — | ✅ |
| A chart fed by a table is refused, written and copied | — | ✅ | — |
| **Make a chart follow a table**: the mediator, its formulas and its owner | — | ✅ | — |
| The entity id and its owner uid are the same bytes, byte-reversed | — | ✅ | — |
| **The app recalculates a chart from the formulas this crate wrote** | — | ✅ | — |
| **The app resaves a rewritten and a copied chart, data intact** | ✅ | — | ✅ |
| Which table and which ranges feed a chart, through function 175 | — | ✅ | — |
| A chart with no mediator has private data and nothing to follow | ✅ | — | ✅ |
| Interactive chart: the data set it is showing, in the model not the view state | — | ✅ | — |
| Sparse series arrays sized by `count`, not by their entries | ✅ | ✅ | ✅ |
| A chart too new for an old reader carries a down-level type patch | — | ✅ | — |
| Every chart-domain archive re-encodes to its bytes (every one in the corpus) | ✅ | ✅ | ✅ |
| Both plist forms in the package read; a binary one survives a rewrite | ✅ | ✅ | ✅ |
| Identity agrees in all three places it is written | ✅ | ✅ | ✅ |
| Locale, creation locale, document language, template id, custom-format list | ✅ | ✅ | ✅ |
| A copy gets four new UUIDs and keeps the lineage; a plain save keeps all five | ✅ | ✅ | ✅ |
| A copy's object streams are the original's, byte for byte | ✅ | ✅ | ✅ |
| **Pages saves the re-identified copy twice and moves only the version** | ✅ | — | — |
| A password-protected package is refused by name, hint and all | ✅ | (same shape) | (same shape) |
| Comments anchor through highlights at the selected characters, with a real author | ✅ | — | — |
| Tracked changes: insertions and a deletion, anchored, in one session | ✅ | — | — |
| An edit through a tracked change is refused by name | ✅ | ✅ | ✅ |
| Write a comment: its author, its anchor, and the run it ends | ✅ | — | — |
| The anchor table starts at 0 and the bare entry stops the run | ✅ | — | — |
| One author however many comments; an overlap is refused | ✅ | — | — |
| **The app resaves an authored comment, author and words intact** | ✅ | — | — |
| Keynote builds: in/out told apart by `animation_type`, effects by stored id | — | — | ✅ |
| Alt text (`accessibility_description`) read — 59 of them, in twelve fixtures | ✅ | ✅ | ✅ |
| The show: theme, slide size, slide tree, layouts in the app's order | — | — | ✅ |
| Every slide's layout is one the theme lists; every layout has a name | — | — | ✅ |
| Placeholder kinds match the fields that name them (title/body/number/object) | — | — | ✅ |
| Presenter notes are the kind-4 storages, and nothing else is | — | — | ✅ |
| Text roles per slide: title, body, number, notes, text box | — | — | ✅ |
| A skipped slide has no number and the rest count past it | — | — | ✅ |
| "Title showing" is ownership — membership of `owned_drawables` | — | — | ✅ |
| Transitions read by the identifier the app's dictionary lists — **all 44** | — | — | ✅ |
| Each effect's `custom_*` parameters, diffed across 44 otherwise identical slides | — | — | ✅ |
| The parameters follow the effect, not the duration, delay or automatic flag | — | — | ✅ |
| Magic Move's whole surface: fade unmatched, acceleration, text granularity | — | — | ✅ |
| No transition anywhere carries a parameter the 15.3.1 schema does not name | — | — | ✅ |
| The app writes no transition direction — absent in seven decks and 182 themes | — | — | ✅ |
| Give a slide a transition, by name or by identifier; refuse one that is not | — | — | ✅ |
| "None" is written as the app writes it, not as the field's absence | — | — | ✅ |
| The `custom_*` parameters follow the effect, and go when it changes | — | — | ✅ |
| **The app resaves a written transition, and a removed one stays removed** | — | — | ✅ |
| Animate a drawable on or off; a drawable the slide does not own is refused | — | — | ✅ |
| **The app resaves a written build, on a text box also written here** | — | — | ✅ |
| Playback: loop, play on open, restart-when-idle, and its **minutes-vs-seconds** trap | — | — | ✅ |
| Presentation type and the two self-playing delays sit at their defaults, written | — | — | ✅ |
| A soundtrack in every deck, empty in all of them; its track list is a data-id list | — | — | ✅ |
| Builds: `keynote-builds.key` carries eight, measured from the Animate inspector; the other six decks and all 182 themes have none | — | — | ✅ |
| No recorded presentation exists; one live-video camera per deck, and it is the default | — | — | ✅ |
| **The app agrees about every slide** — 34 layout names and 14 slides, nine fields each | — | — | ✅ |
| Skip and unskip a slide; unskipping restores the bytes exactly | — | — | ✅ |
| Reorder slides: a permutation of the slide tree, nothing else touched | — | — | ✅ |
| Write presenter notes through the Phase 4 remapper | — | — | ✅ |
| Duplicate a slide: a new component, stream, node, metadata entry and declaration | — | — | ✅ |
| A copy shares no object with its original, and copying twice gives two slides | — | — | ✅ |
| The same duplicate twice produces the same file, byte for byte | — | — | ✅ |
| A copied image slide shares the media instead of growing the package | — | — | ✅ |
| **Keynote reads back the copy's title and notes, and saves the deck itself** | — | — | ✅ |
| The package form: a directory reads as the same entries, and saves back as a directory | ✅ | ✅ | ✅ |
| **The app opens a package this crate wrote**, and writes one file back over it | ✅ | ✅ | ✅ |
| An entry name that would escape the package is refused; a symlink is not an entry | ✅ | ✅ | ✅ |
| A new document carries a calculation engine and the styles a table needs | ✅ | ✅ | ✅ |
| **The app opens a table added to a document made from nothing** | ✅ | ✅ | — |
| A slide made from nothing can be given presenter notes — two objects | — | — | ✅ |
| **The app reads those notes back** | — | — | ✅ |
| A document from a template: a new identity, its own lineage, the template recorded | ✅ | ✅ | ✅ |
| **All three apps open a document made from a template, save it, and leave the identity alone** | ✅ | ✅ | ✅ |
| An edit leaves every `preview*.jpg` byte for byte | ✅ | ✅ | ✅ |
| **A document with its previews removed still opens in the app** | ✅ | ✅ | ✅ |
| Thousands of mutated documents fail rather than panic | ✅ | ✅ | ✅ |

Keynote is the gap in that block for one reason only — neither AppleScript nor
any bundled theme will put a table on a slide, so there is no fixture. The
archives are the same ones Pages uses.

Developed against one real Pages document (a 15 MB German magazine article,
485 objects, two TIFFs and two charts) and two Numbers spreadsheets from
[numbers-parser](https://github.com/masaccio/numbers-parser)'s test suite
(738 and 647 objects, 97 and 37 streams). The style work was checked against
four further Pages documents and one Keynote deck — 654 styles in all.

### Keynote status

Keynote is verified against seven decks — 1 to 46 slides, 17 slide layouts each —
and against its own scripting dictionary, which is the richest of the three:
`scripts/slide-oracle.sh` asks the app for the slide count, the layouts by name,
the four playback settings, and every slide's number, base layout, skipped flag,
title, body, presenter notes and transition, and every one of those is compared.
The effect table is compared too — the app names an effect in English, the
document names it by identifier, and all 44 pairings are checked against the
app rather than transcribed from the dictionary.

Layers 1–3 are exactly as predicted — same stored ZIP, same Snappy framing,
same object stream, text in the same `TSWP.StorageArchive`, styles in the same
attribute tables. Layer 4 is a *show*, and it is [§13 of
FORMAT.md](FORMAT.md#13-keynote-structure--kn). Four things there are worth
repeating here.

**Numbers and Keynote both number their document archive `1`.** The app-level
archives are numbered per app, so the root object's type cannot tell those two
apart, and a `.key` read by type alone came back as a spreadsheet. `Kind`
detection goes by components — `Index/Tables/` for Numbers, `Index/Slide*` for
Keynote — and [`registry`](src/registry.rs) entries carry the app they belong
to, so type 1 resolves to `TN.DocumentArchive` or `KN.DocumentArchive`
depending on the document, and to neither when the kind is unknown.

**Each slide is its own component**, whose identifier is the slide archive's
own, and whose node lives with the show in `Index/Document.iwa`. Nothing else
in the format splits one user-visible thing across two components, and it is
why `iwork duplicate-slide` writes a stream, a metadata entry, a node, a slide
tree entry and an external-reference declaration rather than one object.

**"Title showing" and "slide numbers showing" are not the fields they sound
like.** The first is whether the slide owns the placeholder; the second is a
flag on every slide's node, not on the show.

**A transition's parameters belong to its effect.** `keynote-transitions` is 44
blank slides carrying all 44 effects the dictionary lists, identical in every
other respect, and diffing them is how the `custom_*` block was read: eleven of
the effects bring a parameter and thirty-three bring none. Keynote writes the
ones the effect *has*, false values included, so they are read as optionals —
`apple:scale` carries `custom_bounce` = false, which is not the same as having
no bounce at all.

**A transition can be given, changed and taken away.** `Document::set_transition`
writes the effect, its duration, its delay and its automatic flag, and updates
the node's `has_transition` beside it — a deck with one but not the other plays
what its outline does not show. "No transition" is not the field's absence:
`transition` is a *required* field of `KN.SlideArchive`, and Keynote says so by
name in the unified log before failing to load the slide at all, so what the app
writes — and what this writes — is the same message with the effect `"none"` in
it. The direction is left alone unless asked for, because Keynote's own writer
never emits one.

Three things about a transition cannot be reached from a script and are marked
as such in [§13](FORMAT.md#13-keynote-structure--kn): the **direction**, whose
eight wire values were got by handing Keynote a patched PowerPoint file and
reading back what its importer wrote; the **presentation type** and the two
self-playing delays, which have no scripting term; and the **soundtrack**, which
has none either — `make new audio clip` is accepted by Keynote and then does
nothing at all. Builds went from the largest gap to a measured fact when the
screen was unlocked: no dictionary and no theme will make one, but the Animate
inspector will, and `keynote-builds.key` carries eight. A build-in and a
build-out are told apart by `animation_type` (`"In"`/`"Out"`), and the menu's
"Disappear" stores the identifier `apple:bc-appear`. Action builds, motion
paths and by-bullet delivery remain schema-only — see
[§13](FORMAT.md#13-keynote-structure--kn).

## Testing

No iWork documents are committed — they are other people's files. Supply your
own:

```
cp ~/Documents/Anything.pages tests/fixtures/
cargo test

# or point at a directory you already have
IWORK_FIXTURES=~/Documents cargo test
```

With no fixtures the integration tests skip and say so, so a fresh clone is
green. Fixtures are found recursively, so a whole directory tree of them works.
The unit tests always run: they build synthetic archives in memory and cover the
parts that are easy to get subtly wrong — varint round-trips, repeated-field
ordering, and objects straddling a Snappy block boundary.

If you have Pages, Numbers and Keynote, you can have the apps write you a corpus
instead:

```
scripts/make-fixtures.sh          # into tests/fixtures/generated/, gitignored
scripts/make-fixtures.sh --ui     # + the six only the menus can make
```

`--ui` adds the fixtures nothing scriptable can produce — footnotes, comments,
tracked changes, bookmarks, Keynote builds, hand-hidden rows — by driving the
apps' menus, which works only while the screen is unlocked; the script probes
and skips cleanly when it is not.

Twenty-seven documents (thirty-three with `--ui`) that between them cover plain and styled text, non-Latin
text including emoji, a table and an image, lists, sections and facing pages, a
table of contents, a page-layout document with linked text boxes, page
numbering, columns, a password-protected document, two sheets of typed cells and
formulas, a 300-row imported table, seven decks — presenter notes, a skipped
slide, charts, all 44 transition effects, the playback settings, eight builds — and a slide
carrying one of every drawable a script can make: a shape, a rotated shape, a
shape at half opacity with a reflection, a text box, a line, an image, a locked
shape and an image Keynote itself cropped. Five spreadsheets are built from
templates Apple ships, carrying a category with a summary row, two pivot tables,
a filter that hides rows, columns hidden by hand, conditional highlighting, a
custom cell format and a sort rule; a sixth is Apple's `.nmbtemplate` bundle
renamed, for the hyperlink fields nothing else here can make. Existing files are
left alone unless `--force` is given.

Keynote builds the drawable fixture because it is the only one of the three
apps that will create a drawable from a script: Pages and Numbers answer `make
new shape` with "Don't know how to create TMAScriptShapeInfoProxy". `TSD` is
cross-app, so that deck is the shape fixture for all three. `make new group` and
`make new movie` are accepted and then do nothing, so groups and movies stay
read-only here, exercised by the themes that ship with them.

The four spreadsheets come from templates for a reason: Numbers' scripting dictionary
has no sort, filter, category, highlight or pivot command, and the menu items
that do need a document window and therefore an unlocked screen. The generator
names them by template `id`, which is the path inside the app bundle and the
same on every Mac; the localised template *name* is never used.

And the check the rest of the suite cannot make — does the app open it?

```
scripts/app-check.sh out.pages "A new headline"   # exit 0 if Pages agrees
scripts/app-check.sh --self-test Report.pages     # prove it fails when it should
scripts/table-oracle.sh Budget.numbers            # every cell, as Numbers reads it
scripts/drawable-oracle.sh Talk.key               # every rectangle, as the app reports it
scripts/section-oracle.sh Report.pages            # every section's text, as Pages reads it
scripts/resave.sh out.pages                       # have the app open it and write it out again

IWORK_APP_CHECK=1 cargo test                      # every fixture, through the app
```

`app-check.sh` opens the document in the app that owns its extension, reads back
body text, cell values, slide text and presenter notes, looks for a string if
you give it one, and closes without saving. `--self-test` corrupts a copy of a
document it has just accepted and checks that the app refuses it, because a
harness that always says yes is worse than none.

Every one of these gives the app two attempts, killing it in between: a failure
from an app means either "it will not do this" or "it was busy", the two look
identical, and a busy app is not busy from a cold start.

`resave.sh` is the harder test, for the parts of a document no dictionary will
report. Pages has no header, footer, footnote or column property at all, so
"the app read it back" has to be arranged: the app is made to open the edited
document and **save it**, and the file that comes out was written by Pages from
its own model. A header this crate invented badly does not survive that.

### Fuzzing

`tests/fuzz.rs` is a mutation fuzzer over the corpus, and it runs as part of
`cargo test`: every fixture, every entry of every fixture, every object payload
and a synthetic package built in memory are seeds, a deterministic generator
mutates one, and the result goes through the ZIP layer, the Snappy framing, the
object stream, the plists and then every reader the crate has — under
`catch_unwind`, so a panic is a test failure with the seed that caused it and
the offending bytes dumped to `/tmp`.

```
cargo test --test fuzz                                   # ~20 seconds, the default budget
IWORK_FUZZ_SECONDS=600 cargo test --release --test fuzz  # a proper run
IWORK_FUZZ_SEED=12437 IWORK_FUZZ_ITERATIONS=1 \
  IWORK_FUZZ_BACKTRACE=1 RUST_BACKTRACE=1 \
  cargo test --test fuzz                                 # replay one case
```

Run it in **both** profiles. A debug build panics on integer overflow, which is
how the plist length arithmetic was caught; a release build is an order of
magnitude faster and reaches further into the readers. Everything the fuzzer has
found is fixed, and each fix has a named test beside the harness describing the
shape of the input that caused it.

The table reader carries the same rule down to its arithmetic: a tile id that
overflows `tile_id × tile_size`, a merge extent that reaches the top of `usize`,
a category tree whose child list references itself, a `nextListID` at the 32-bit
ceiling, and hostile `set-cell` values (`n:1e-2147483648`, `d:25…-01-01`) each
yield a bounded error or a truncated read rather than a panic or an abort. The
regression tests for these splice the hostile shape into a real fixture's `.iwa`
stream and assert the call **returns** (`tests/tables.rs`, `tests/cells.rs`).

`cargo fuzz` is not used: it needs a nightly toolchain for `-Z sanitizer` and
this machine has only stable, so the committed harness is the whole of the
fuzzing story rather than half of it.

## Limitations

- **Previews go stale, deliberately.** The three `preview*.jpg` thumbnails are a
  picture of the document as it was, and an edit here does not redraw them —
  drawing one means laying the document out, which is a far larger project than
  reading and writing the file. They are also not *removed*, which was the
  choice worth making rather than assuming: nothing in 927 packages refers to
  them, the app draws them again the next time it saves, and taking them out
  would cost the byte-identity a no-op save promises. So the Finder shows the
  old first page until iWork saves the document, `iwork check` says nothing
  about it, and `iwork strip-previews` is there for the caller who would rather
  have no thumbnail than a wrong one — every template Apple ships has none, and
  all three apps open a document without them. `FORMAT.md` §1 has the evidence.
- **A text edit that would delete an anchored object is refused.** Deleting the
  `U+FFFC` an image or a table hangs off — or the `U+000E` that *is* a footnote
  mark — means deleting that object from the drawable list, the z-order and the
  media registry, or orphaning a note nothing can reach. Pages does all of
  that; this crate does none of it, and says so by name instead of quietly
  detaching the object.
- **Two sections cannot be merged, so deleting a section break is refused.**
  The `U+0004` is what makes the section; deleting it leaves two
  `TP.SectionArchive`s where one boundary is needed, and which of the two keeps
  its three section templates, its eighteen header and footer storages, its
  guides and its background is a question Pages will not answer for anyone. `delete
  section 2` comes back -10000, there is no `make new section`, the menu needs
  a window, and setting a section's body text to the empty string leaves the
  break where it was with a zero-length section behind it. Rather than guess,
  `Error::SectionBreak` says which break, which section, and what is unknown.
- **No template ships a footnote, an endnote or a bookmark, and no script can
  make one.** All 901 templates the three apps ship were scanned for a storage
  of kind 2, a `TSWP.FootnoteReferenceAttachmentArchive` and a
  `TSWP.BookmarkFieldArchive`: zero, zero and zero, and no scripting dictionary
  has a command for any of them. `pages-footnotes.pages` and
  `pages-bookmarks.pages` exist because the Insert menu was clickable on an
  unlocked screen, and they are what turned the containment from Inferred into
  Confirmed: a `U+000E` mark in the text, a
  `TSWP.FootnoteReferenceAttachmentArchive`, and the note's own storage of kind
  2. What stays Unverified is what the fixtures do not reach — endnote modes,
  custom marks, restarting numbering — and nothing here writes a footnote.
- **Rewriting a header replaces whatever the header was.** The date in a
  newsletter's header is a smart field and the storage holds the string it last
  rendered to, so setting the text removes the field and freezes the date. The
  same is true of a page number. The edit report names the tables it rewrote,
  which is how to tell.
- **Changing a paragraph's list level is not implemented.** The level is read —
  `iwork paragraphs` prints it — but nothing here can make an app perform that
  edit to check a write against: Pages' rich text carries `font`, `size` and
  `color` and no list property, and the menu item needs a window.
- **Most style fields have no name.** Ten are named; the rest of the property
  bag is addressed by number, because the meaning is not published and
  [`style`](src/style.rs) will not guess. Read the tree with `iwork style`,
  compare two styles that differ in the way you want, and set the field that
  moved. New styles come from copying, so fields you never touch keep whatever
  the template had.
- **Editing a named style may change nothing.** Text usually points at an
  anonymous *variation* style that inherits from the named one and overrides
  some fields. `iwork style` prints what a style inherits from; edit the style
  the runs actually point at.
- **A copy of a variation style stays anonymous.** Named styles and variations
  are different things, and an object that is flagged a variation, carries a
  name and has no internal identifier is neither — Pages crashes on opening the
  document. `create_text_style` therefore applies the requested name only when
  the template has one, and reports which happened.
- **Deleting a style is refused rather than forced.** If a reference this crate
  cannot account for would be left dangling, the delete fails and says which
  objects still hold one.
- **A slide can be copied, added to, skipped and moved — but not created or
  deleted.** `duplicate-slide` copies a slide that exists; there is no
  `new-slide`, because a slide made from nothing is a component made from
  nothing, and ground rule 3 says copy. There is no `delete-slide` either: the
  slide tree entry, the node, the component, its metadata entry, its media
  refcounts and its external-reference declarations all have to go together and
  no probe has watched Keynote do it. A slide's *layout* cannot be changed and
  cannot be copied — Keynote's own dictionary makes `slide layout` read-only.
- **A transition can be written; a build cannot.** `iwork set-transition` gives
  a slide any of the 44 effects, with a duration, a delay and the automatic
  flag, and `none` takes it away — verified through a resave in Keynote. What
  is *not* written is the direction, because Keynote's own writer never emits
  one, and the `custom_*` parameters, which belong to the effect that wrote
  them and are dropped when the effect changes. The eight builds
  `keynote-builds.key` carries are read — effect, delivery, event trigger and
  the `"In"`/`"Out"` direction — and `iwork add-build` writes one: the build,
  its chunk, both of the slide's lists and the node's three counters. Keynote
  resaved a deck with a build this crate wrote on a text box this crate also
  wrote. Two fields go in because the app writes them and **neither has been
  measured** — the attributes' field 17, `60` on all eight of the fixture's
  builds, and the chunk's flag and pair of UUIDs; they are reproduced in the
  app's shape and nothing here claims to know what they mean. The action-build,
  motion-path and by-bullet-group fields beside them stay schema-only, so a
  build is a whole-object dissolve or disappear and nothing subtler.
- **A copied slide's thumbnail is the original's.** The node keeps the source's
  `thumbnails` data reference and is marked `thumbnailsAreDirty`, which is what
  Keynote's own duplicate leaves behind — so the navigator shows the right
  picture only after the app redraws it.
- **A reference that leaves its component has to be declared.** The document
  body and the stylesheet are separate components, so pointing text at a style
  is two edits: the run, and a `ComponentInfo.external_references` entry saying
  where the style lives. Without the second, iWork never loads the style — and
  the failure is *quiet*: one such document opened in Pages with the paragraph
  simply unstyled, as though nothing had been done, and another crashed on open.
  `apply_text_style` and `create_text_style` maintain the declarations;
  `iwork check` reports any that are missing.
- **A formula can be written from its text, and its answer cannot.**
  `set_formula` parses `=SUM(B2:B4)` and writes the node stream the app writes —
  every node copied from one Numbers wrote for the same formula, down to the
  list node a parenthesis needs and the `5: 1` on every colon tract — and
  registers the cell in the calculation engine so the app recalculates it. What
  it cannot do is *evaluate*: the value the cell shows until a precedent moves
  is the caller's to supply, exactly as for `fill_formula`. Refused by name:
  a cell that already holds a formula, a reference to another table, a whole row
  or column, a header name, a function this crate does not know, and a table
  with no cell owner in the calculation engine — which no longer includes the
  tables this crate builds: a new table is given the two `TSCE` owners a formula
  needs, and **Numbers recalculates a formula written into a document made from
  nothing**.
- **The organisation layer is written in three places and read everywhere
  else.** Sort rules go in whole — they are one inline archive on the model, and
  they say what to sort *by*, not what order the rows are in, so nothing here
  moves a row. A filter's switch and its all/any mode go in; **its rules do
  not**, because Numbers compiles a filter condition into a `TSCE` formula and
  this corpus carries exactly one of those to learn from. And a conditional
  highlight's threshold can be changed — in all four places a rule keeps it —
  for the two predicates whose meaning is established, greater-than and
  less-than. Categories and pivots stay read-only for the same reason filters
  do.
- **Switching a filter off does not un-hide its rows.** Which rows are hidden is
  *stored*, not worked out when the document opens: with the filter written off,
  Numbers opened the document, edited a cell and saved, and the ten hidden rows
  were still hidden. The app recomputes them when the filter is next touched in
  its own interface. Clearing the hidden state as well means rewriting the
  UUID-keyed extent that `insert_row` refuses to maintain, for the same reason.
- **A page-layout document's body is refused, not written.** Pages has two
  modes, and the crate reads both: word processing, where text flows from page
  to page, and page layout, where every word is in a text box. A page-layout
  document still *has* a body storage — holding the `U+0004` that starts its
  first section — and the app never draws it, because that is what the Document
  Body switch being off means. Appending to it therefore succeeds at the byte
  level and produces text nobody will ever see: measured, by appending a
  paragraph to `pages-layout.pages`, opening it in Pages and asking for every
  word in the document — forty lines came back and the new paragraph was not
  among them. So `append_paragraph` and `body_mut` refuse on such a document and
  say why; `add_text_box` is how words get onto its pages, and the storage
  itself is still reachable by identifier for a caller who means exactly that.
- **A row or column can be deleted, and the refusal list is longer than the
  insert's.** What goes takes its cells' references with it, which is the part an
  insert never has to do. Refused: the table's only row or column, a header or
  footer line, a merge at or after it, a cell in it holding a formula, and any
  formula anywhere that names the line or a range across it — stricter than the
  insert's check, because a delete takes cells away and a reference to a deleted
  line is a `#REF!` however it was written. The organised tables — categorised,
  filtered, pivoted, conditionally highlighted, hidden — are refused as they are
  for an insert.
- **Cells can be merged, and the app has no merge property to check it with.**
  A merge is a formula in the table's merge owner, and the node array this crate
  writes for a range is byte for byte the one the app wrote for the same merge —
  all four in the fixture, reproduced from nothing but their row, column and
  size. The covered cells are emptied, which is what the app leaves behind. What
  cannot be asked of the app is "is this merged": a merged-away cell is reported
  under the *name and value of the cell the merge began in*, and that is the
  check — `A1 A1` across a row is a 1×2 merge at A1.
- **A data format is written into the slot the value uses, and nowhere else.**
  `set_format` gives a cell a number, percent, scientific, currency or date
  format — the archive being the one the app wrote for the same format, down to
  the `253` that means "as many decimals as it takes". What it will not do is
  put a format in another slot: a currency format on a plain number cell is
  ignored by Numbers, drawn as a plain number, so it is refused rather than
  written into a file where it would sit and never show. The slot follows the
  *value's type*, so making a number into a currency is a value write —
  `set_currency` is that write, and the money it makes is drawn as money.
  Column
  widths and row heights are one float each and the app reports them back
  exactly; the table's frame is deliberately untouched, because the app lays a
  table out from its columns and not from its frame.
- **No layout, no rendering, no formula evaluation.** This reads and rewrites
  the document; it does not understand it. In particular, writing a cell a
  formula depends on leaves that formula's **cached value stale**. Numbers
  recalculates on open, so the app shows the right answer; anything that reads
  the file without evaluating — this crate included — shows the old one.
- **Media is replaced in place, and the frame does not follow.** A replacement
  of a different shape is drawn stretched into the frame the old picture had —
  `replace-media` says so and `set-geometry` fixes it, but the app would instead
  have re-fitted the picture and cropped it with a mask. And a picture whose
  drawable carries a crop, a shaped mask, an Instant Alpha path, adjustments or
  cached renderings of the old pixels is not replaced at all: that state is
  computed from the pixels being thrown away, and the result would open and
  render wrong.
- **A shape that sizes itself to its text has no size in the archive.** Its
  stored height is 0 and its stored position is the centre of a box the app
  computes when it lays the text out — Keynote reports such a text box 58 points
  above where the file puts it. `Geometry::fits_its_text` says when a rectangle
  is an anchor rather than a box; nothing here can turn one into the other
  without doing layout.
- **Groups, movies and drawings are read, never authored.** No script can make
  Keynote create a group or a movie, so nothing here writes one; the archives
  are decoded and carried through. Live video sources, recorded presentations
  and pencil annotations are on the never-author list by design.
- **Cells are written in place, one or many at a time.** `set_cell` changes one
  value; `set_cells` and `set_block` write a batch, and the batch is not just
  sugar. A cell lives in its row's `TileRowInfo` and names strings and formats
  in table-wide lists, so writing a table one cell at a time decodes and
  re-encodes a tile and two lists per value — quadratic in the table, measured
  at 15 000 cells in 206s. A batch decodes each tile, list and bucket once:
  **100 000 cells in 0.26s**. It is also all or nothing, where a loop of single
  writes stops half-applied. A row that holds no cells at all is no longer
  refused — the first value written into one builds the `TileRowInfo` the row
  never had, which is what makes an inserted row fillable. What is still
  refused by name: a formula cell, a rich-text cell, a cell covered by a merge,
  a value type this crate does not write, and adding or removing rows and
  columns, which is `insert_row`/`insert_column`'s business.
- **"iWork opens it" is not tested here, and it is not a formality.** The tests
  prove the bytes are structurally correct and survive an independent decode.
  They cannot prove an app will accept the result, and the difference is real:
  documents that pass every check in this repository — including `iwork check`,
  which finds nothing wrong with them — have crashed Pages on opening. Each time,
  the fix has been to find an invariant the real documents hold to exactly, teach
  `iwork check` to assert it, and maintain it on write; the checker is that much
  sharper each round, and still not a substitute for opening the file. Anything
  written by this crate needs trying in the app before it is trusted —
  `scripts/app-check.sh` is how, and `IWORK_APP_CHECK=1 cargo test` runs it over
  every fixture, on a machine that has the apps.

  **And when such a document is found, fixing it comes first.** That is a rule
  of this repository, not an aspiration — `PLAN.md` ground rule 1a states it in
  full. A document the app rejects, or silently rewrites, stops the work in
  hand: find the invariant the app's own documents hold to (by bisecting the
  document until the smallest difference is in hand), teach `iwork check` to
  assert it, maintain it on write, and leave the case behind as a test named
  after the shape that caused it. Refusing the write by name is an acceptable
  outcome; leaving a writer in place that is known to produce documents the app
  will not open is not.
- **A comment can be written; a reply cannot.** `iwork add-comment` attaches one
  to a range of text: the comment, its author — added to the document's one
  author storage if that name is not there yet — and the two entries in the
  storage's run-anchored `table_highlight`, the anchor at the start and the
  *bare* index at the end that stops the run. That table also has to **start at
  0**, whatever the comment does, or the characters before it have no attribute
  at all. Pages opened a document with a comment authored this way and wrote
  every part of it back, which is the only measure there is: no scripting
  dictionary can read a comment. Refused by name: a storage carrying tracked
  changes, a range past the end of the text, an empty range, and an overlap with
  a comment already there — overlapping comments are what
  `table_overlapping_highlight` is for, and nothing here has one to write from.
- **Comments come only from the menu, and replies not yet at all.** No
  scripting dictionary has a comment command or a comment class, and no
  template the three apps ship carries a comment: all 26 base-corpus fixtures
  and all 901 templates hold an empty `TSK.AnnotationAuthorStorageArchive` and
  nothing below it. `pages-comments.pages`, made by driving the Pages UI on an
  unlocked screen, is the one document that finally has some — a real author
  and two comments anchored through the highlight table at the characters the
  recipe selected, which turned the anchor route from a schema reading into a
  measured fact (and fixed one bug the schema alone could not show). What stays
  decoded-from-the-schema and **Unverified in `FORMAT.md`**, with a tripwire
  test for the day a fixture reaches it, is replies, any resolved state (no
  descriptor anywhere contains `resolv`), cell comments, and the
  overlapping-highlight table. **Tracked changes are the same story:**
  `pages-tracked.pages`, made with Track Changes on from the menu, carries a
  real `table_deletion`, so the change decoder has its one example and an edit
  through it is refused (below).
- **An edit through a storage with tracked changes is refused.** A tracked
  deletion keeps its characters — they are still in the text and Pages draws
  them struck through — so `table_deletion` is not the run table it looks like,
  and nothing available here can make the app perform such an edit to be
  watched. `Error::TrackedChanges` declines and names the storage.
- **A password-protected document is refused, not decrypted.** A locked package
  is recognised by its `.iwpv2` entry, its hint is read out of `.iwph`, and
  `Error::Encrypted` says so; every `Index/*.iwa`, every `Data/*` and the build
  history are ciphertext. Setting or removing a password is something the apps
  do and this crate does not.
- **A copy needs a new identity, and the app agrees but iCloud could not be
  asked.** `save_as_new` gives a copy four fresh UUIDs and keeps
  `stableDocumentUUID`, which is what Pages' own Save As was measured doing —
  and Pages then saves that copy repeatedly while moving only its `versionUUID`,
  whereas it re-identifies a plain byte copy of its own accord. What could not
  be shown is a *collision*: Pages opens an original and a byte-identical copy
  side by side without complaint, and there is no iCloud account here to watch
  the sync layer care.
- **Applying a character style may not change how text looks.** Pointing a run
  at a different character style is accepted and survives a reopen, but has not
  been observed to change the rendering, so something else evidently wins.
  Unresolved.
- **A document is created from a template, or from nothing.**
  `Document::from_template` copies a template bundle into a new identity; two
  things it cannot do are name the template when the bundle is not one of the
  app's own — a user template in `~/Library` has an identifier and it is not
  derivable from the path — and clear view state, of which there is none in any
  bundled template to clear. A `.template` renamed `.pages` also works and
  always has; what `from_template` adds is the identity.
  `Document::new` needs no template and no app, and all three apps open what it
  writes and save it back. What it writes is the measured minimum plus what
  makes the result usable rather than merely legal — a paragraph style with a
  font, a table with writable cells, a slide with a master. It is not a document
  with a *theme*: the apps fill the rest in the first time they save, which was
  watched happening rather than hoped for.
- **The file type is kept, not chosen.** A package (a directory) is read and
  saved as a package, a single file as a single file. `File > Advanced > Change
  File Type` is a menu item, so what a document a user has set to the package
  form does on save could not be watched; what was watched is Pages handed a
  package from a script and told to save, which wrote one file back over it.
  `Package::write_as` performs the conversion when a caller asks for it.
- **Hostile files are fuzzed, not proven safe.** `tests/fuzz.rs` runs thousands
  of mutated documents through every decoder on every `cargo test`, and
  everything it has found is fixed — a Snappy block claiming to decompress to
  four gigabytes, a `MessageInfo` claiming 2^60 bytes of payload, a ZIP entry
  claiming a size no file has, plist lengths that overflow when doubled, an
  archive with no message in it. That is coverage, not a proof: the harness is
  dumb mutation over a corpus of 27 documents, `cargo fuzz` needs a nightly
  toolchain this machine does not have, and a decoder can still meet a shape
  nothing here generated.
- **Encrypted documents are refused rather than decrypted** — see the
  password-protected bullet above; there is no key derivation here and none is
  planned.

## Changes

[`CHANGELOG.md`](CHANGELOG.md) — what changed and, because this is a
reverse-engineered format, **how each claim was established**.

## Prior art

[numbers-parser](https://github.com/masaccio/numbers-parser),
[keynote-parser](https://github.com/psobot/keynote-parser) and
[iWorkFileFormat](https://github.com/obriensp/iWorkFileFormat) mapped much of
this territory first, in Python and Objective-C. This crate is an independent
Rust implementation working from the bytes up, and is deliberately schema-less
where those projects carry extracted `.proto` files.

## Legal

Reverse engineering a file format to achieve interoperability is permitted under
EU Directive 2009/24/EC Art. 6 and Swiss URG Art. 21. This repository contains no
Apple code, no Apple `.proto` files and no iWork documents.

## License

MIT — see [LICENSE](LICENSE).
