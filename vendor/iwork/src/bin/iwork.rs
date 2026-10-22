//! `iwork` — inspect and edit Pages, Numbers and Keynote documents.

use std::collections::BTreeMap;
use std::process::ExitCode;

use iwork::drawable::Outline;
use iwork::pb::{self, Message, Value};
use iwork::{registry, style, Document, Error};

const USAGE: &str = "\
iwork — inspect and edit Apple iWork documents (.pages, .numbers, .key)

  iwork inspect   <file>                   package, components, media, object census
  iwork text      <file>                   every text storage, with its object id
  iwork set-text  <file> <id> <text> <out> replace one text storage
  iwork insert-text <file> <id> <at> <text> <out>
                                           insert text at a character index
  iwork delete-text <file> <id> <from> <to> <out>
                                           delete a range of characters
  iwork storages  <file>                   every text storage and the attribute
                                           tables it carries
  iwork links     <file>                   hyperlinks and smart fields, with the
                                           text each one covers
  iwork objects   <file> [type]            list objects, optionally of one message type
  iwork dump      <file> <id>              one object, field by field
  iwork check     <file>                   look for a broken object graph
  iwork extract   <file> <dir>             write embedded media to a directory
  iwork roundtrip <file> <out>             decode and re-encode every object

metadata, identity and the review layer

  iwork metadata  <file>                   Properties.plist, DocumentIdentifier,
                                           build history, locale, template,
                                           custom-format list, encryption
  iwork add-comment <file> <storage> <start> <end> <author> <text> <out>
                                           attach a comment to a range of text
  iwork annotations <file>                 annotation authors, comments and
                                           their anchors, tracked changes
  iwork duplicate <file> <out>             save a copy with a *new* document
                                           identity, so the two do not collide
  iwork new       <template> <out>         make a document from a template
                                           bundle (.template/.nmbtemplate/.kth)
  iwork create    <kind> [paper] <out>     make an empty document from nothing:
                                           pages/numbers/keynote, a4 or letter
  iwork strip-previews <file> <out>        drop the preview images

`iwork duplicate` gives the copy fresh documentUUID, shareUUID, privateUUID and
versionUUID values and a revision to match — what Pages' own Save As was
measured doing — and keeps stableDocumentUUID, which is what says the copy came
from this document. A plain save keeps every one of them.

A document carries three `preview*.jpg` — the picture the Finder shows — and an
edit makes them a picture of the document as it was. They are left alone
anyway, because a no-op save reproduces every byte and because the app draws
them again the next time it saves; `iwork check` says nothing about them.
`iwork strip-previews` is for the caller who would rather have no thumbnail
than a wrong one: every template bundle Apple ships has none, and all three
apps open a document without them.

`iwork new` does the same and takes stableDocumentUUID with it, because a
document made from a template is a new document rather than a copy of the
template — measured on documents the apps themselves made from templates they
ship. The templates are in
`/Applications/<App>.app/Contents/SharedSupport/Templates`, and a document made
from one there records which one it came from.

No app's scripting dictionary has a comment command, so the fixtures with one
were made through the Insert menu on an unlocked screen. `iwork add-comment`
writes one: the comment, its author (added to the document's one author storage
if that name is not there yet) and the two entries in the storage's run-anchored
`table_highlight` — the anchor at <start>, and the bare index at <end> that ends
the run. It refuses a storage carrying tracked changes, a range past the end of
the text, an empty range, and an overlap with a comment already there.

Character indices — <at>, <from>, <to> — are UTF-16 code units, the unit iWork
counts text in, so an emoji is two, and ranges are half-open. An edit that
would land inside a surrogate pair, or delete the character an image, a
footnote or a section break is anchored to, is refused by name rather than
performed badly.

drawables and media

  iwork drawables <file>                   every placed object: geometry, style,
                                           media and non-destructive edit state
  iwork media     <file>                   every media file, its digest and its users
  iwork set-geometry <file> <id> <x> <y> [<w> <h>] <out>
                                           move or resize one drawable
  iwork replace-media <file> <id> <image> <out>
                                           swap the bytes an image is drawn from
  iwork add-text-box <file> <where> <text> <x> <y> <w> <h> <out>
                                           put a text box on a slide, a sheet
                                           or a page
  iwork add-shape <file> <where> <outline> <text> <x> <y> <w> <h> <out>
                                           the same, drawn along <outline>:
                                           rectangle, ellipse or line
  iwork add-image <file> <where> <image> <x> <y> [<w> <h>] <out>
                                           place a PNG or a JPEG, at its own
                                           pixel size unless given one

An <id> is an object identifier as printed by `iwork drawables`; for
replace-media it may also be a media identifier as printed by `iwork media`.
Positions and sizes are in points, and are the rectangle the app reports — for
a cropped image that is the mask's window, not the picture's own rectangle.

For add-text-box, <where> is a Keynote slide (an object id from `iwork
slides`), a Numbers sheet (its name or id) or a Pages page (`page 2`, counting
from one). The box is drawn with a style the document already has — the style
of a text shape on it, or the theme's text-box preset — so it looks like what
is around it. A document with no such style is refused rather than given an
invented one.

tables

  iwork tables    <file>                   every table: size, headers, geometry
  iwork cells     <file> <table> [--raw]   every cell of one table, with type and format
  iwork csv       <file> <table>           one table as CSV
  iwork organise  <file> [<table>]         sort rules, filters, categories,
                                           pivots, conditional highlighting,
                                           custom cell formats
  iwork formulas  <file> [<table>]         every formula, with its cell, its
                                           text and the value the app cached
  iwork set-cell  <file> <table> <cell> <value> <out>
  iwork set-cell  <file> <table> <row> <col> <value> <out>
                                           write one value into one cell
  iwork set-cells <file> <table> <cell> <csv> <out>
                                           write a CSV file into the table,
                                           its first field landing at <cell>
  iwork set-formula <file> <table> <cell> <formula> <value> <out>
                                           give a cell a formula and the value
                                           it shows until the app recalculates
  iwork set-format <file> <table> <cell> <format> <out>
                                           a cell's data format: automatic,
                                           number[:places], percent[:places],
                                           scientific[:places],
                                           currency:CODE[:places], date:PATTERN
  iwork set-width  <file> <table> <column> <points>|default <out>
  iwork set-height <file> <table> <row> <points>|default <out>
                                           a column's width, a row's height
  iwork insert-row <file> <table> <at> <out>
                                           insert an empty row before index
                                           <at> (<at> == row count appends)
  iwork insert-column <file> <table> <at> <out>
                                           the same for a column
  iwork delete-row <file> <table> <at> <out>
  iwork delete-column <file> <table> <at> <out>
                                           delete a row or a column, with
                                           everything in it
  iwork merge   <file> <table> <cell> <rows> <cols> <out>
                                           merge a rectangle into one cell
  iwork unmerge <file> <table> <cell> <out>
                                           take that merge apart
  iwork add-table <file> <where> <name> <rows> <cols> [<x> <y>] <out>
                                           add a table to a sheet, a slide or a
                                           Pages page, at <x>,<y> if given
  iwork add-sheet <file> <name> <table> <rows> <cols> <out>
                                           add a sheet with one table on it
  iwork fill-formula <file> <table> <from> <to> [value] <out>
                                           give a cell the formula another
                                           holds, and the value it shows

charts

  iwork charts    <file>                   every chart: type, placement, the
                                           data it carries and the table
                                           references it follows
  iwork set-chart-data <file> <chart> <csv> <out>
                                           rewrite the numbers a chart draws
  iwork add-chart <file> <where> <from> <csv> <x> <y> <w> <h> <out>
                                           copy chart <from> onto a slide, a
                                           sheet or a page, drawing <csv>

A <csv> is a small table: the first row names the categories, the first cell of
each row after it names the series. Blank cells are blank; anything that is not
a number is refused rather than plotted as zero.

A chart holds its data twice. The grid printed as a small table is the chart's
own private copy — the thing it draws, and all a Pages or Keynote chart has. A
Numbers chart also has a mediator holding formulas back into tables, printed as
`fed by …`; the grid is then a cache of what those last evaluated to — so a
chart that says `fed by` is refused by both set-chart-data and add-chart, since
numbers written into that cache disagree with the table the moment Numbers
recalculates. add-chart copies a chart the document already has: a dozen
objects of theme properties stand behind one, and none of them can be invented
honestly.

A <table> is an object id, as printed by `iwork tables`, or a table name. A
<cell> is an A1 reference; <row> and <col> are the 0-based indices the API
uses, so `B3` and `2 1` are the same cell.

A cell <value> is text unless it says otherwise: `n:` a number, `b:true` or
`b:false`, `d:2024-03-01T10:30:00Z` a date, `dur:5400` a duration in seconds,
`text:` text that would otherwise look like a prefix, and `empty:` to clear the
cell. A number is stored as a decimal rather than a float, so `n:0.1` is
exactly a tenth.

  iwork set-cell Budget.numbers Sales B3 n:43 out.numbers
  iwork set-cells Budget.numbers Sales A2 rows.csv out.numbers
  iwork set-formula Budget.numbers Sales B9 '=SUM(B2:B8)' n:1234 out.numbers

Pages document structure

  iwork sections  <file>                   sections, their text ranges, page
                                           numbering and header/footer storages
  iwork structure <file>                   document mode, paper and margins,
                                           page templates, linked text-box
                                           threads, contents lists, footnotes,
                                           bookmarks, columns

Both are Pages-only; a Numbers or Keynote document has no `TP.DocumentArchive`
and they say so. A header or footer is an ordinary text storage, so editing one
is `iwork set-text <file> <storage> <text> <out>` with the id `iwork sections`
prints.

Keynote decks

  iwork slides    <file>                   the deck: every slide, its layout,
                                           its placeholders and their text, its
                                           presenter notes, its transition and
                                           how many builds it carries
  iwork layouts   <file>                   the theme's slide layouts, in the
                                           order the app lists them
  iwork set-notes <file> <slide> <text> <out>
                                           replace a slide's presenter notes
  iwork skip-slide   <file> <slide> <out>  leave a slide out of the show
  iwork unskip-slide <file> <slide> <out>  put it back
  iwork move-slide <file> <slide> <to> <out>
                                           move a slide to position <to>, from 0
  iwork duplicate-slide <file> <slide> <out>
                                           copy a slide, straight after it
  iwork add-slide <file> [layout] <out>    add an empty slide at the end, drawn
                                           from one of the deck's layouts
  iwork effects                            every transition effect, by name and
                                           by the identifier the archive holds
  iwork set-transition <file> <slide> <effect> [<duration> [<delay>]] <out>
                                           give a slide a transition; `none`
                                           takes its transition away
  iwork add-build <file> <slide> <drawable> in|out [<effect>] <out>
                                           animate a drawable on or off

All Keynote-only; a Pages or Numbers document has no `KN.ShowArchive` and they
say so. A <slide> is the object id `iwork slides` prints, either the slide's or
its node's. Title and body text are ordinary storages — `iwork slides` prints
their ids and `iwork set-text` writes them, with the same remapping as any
other text. A skipped slide has no number: the app answers -1 for it and
numbers the rest around it.

text styles

  iwork styles       <file>                            every text style, with its object id
  iwork style        <file> <id>                       one style: every field, and what uses it
  iwork new-style    <file> <template> <name> <out>    copy a style under a new name
  iwork set-style    <file> <id> <path=value> <out>    set or clear one field of a style
  iwork delete-style <file> <id> [<replacement>] <out> remove a style
  iwork apply-style  <file> <storage> <start> <end> <style> <out>
                                                       point a range of text at a style
  iwork paragraphs   <file> <storage>                  paragraph ranges, for apply-style
  iwork properties                                     every named style property
  iwork set-color    <file> <id> <r> <g> <b> <out>     text colour, everywhere the style keeps it

A <path> is a dotted list of protobuf field numbers, as printed by `iwork
style`, or one of the named properties — `iwork properties` lists them. A
<value> is varint:N, f32:N, f64:N, str:TEXT, hex:BYTES, or empty to remove the
field. Ranges are half-open and counted in UTF-16 code units, the unit iWork
indexes text in.

  iwork set-style Report.pages 3801 font-size=f32:18 out.pages
  iwork set-style Report.pages 3801 11.3=f32:18      out.pages   # the same field
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();

    let result = match refs.as_slice() {
        ["inspect", file] => inspect(file),
        ["text", file] => text(file),
        ["set-text", file, id, new_text, out] => match id.parse() {
            Ok(id) => set_text(file, id, new_text, out),
            Err(_) => Err(Error::Format(format!("'{id}' is not an object identifier"))),
        },
        ["insert-text", file, id, at, new_text, out] => identifier(id)
            .and_then(|id| index(at).map(|at| (id, at)))
            .and_then(|(id, at)| insert_text(file, id, at as u64, new_text, out)),
        ["delete-text", file, id, from, to, out] => identifier(id)
            .and_then(|id| index(from).map(|f| (id, f)))
            .and_then(|(id, f)| index(to).map(|t| (id, f, t)))
            .and_then(|(id, f, t)| delete_text(file, id, f as u64..t as u64, out)),
        ["storages", file] => storages(file),
        ["links", file] => links(file),
        ["objects", file] => objects(file, None),
        ["objects", file, message_type] => match message_type.parse() {
            Ok(t) => objects(file, Some(t)),
            Err(_) => Err(Error::Format(format!(
                "'{message_type}' is not a message type"
            ))),
        },
        ["dump", file, id] => identifier(id).and_then(|id| dump_object(file, id)),
        ["check", file] => check(file),
        ["extract", file, dir] => extract(file, dir),
        ["roundtrip", file, out] => roundtrip(file, out),
        ["styles", file] => styles(file),
        ["style", file, id] => identifier(id).and_then(|id| show_style(file, id)),
        ["new-style", file, template, name, out] => {
            identifier(template).and_then(|t| new_style(file, t, name, out))
        }
        ["set-style", file, id, assignment, out] => {
            identifier(id).and_then(|id| set_style(file, id, assignment, out))
        }
        ["delete-style", file, id, out] => {
            identifier(id).and_then(|id| delete_style(file, id, None, out))
        }
        ["delete-style", file, id, replacement, out] => identifier(id)
            .and_then(|id| Ok((id, identifier(replacement)?)))
            .and_then(|(id, replacement)| delete_style(file, id, Some(replacement), out)),
        ["apply-style", file, storage, start, end, style, out] => {
            apply_style(file, storage, start, end, style, out)
        }
        ["drawables", file] => drawables(file),
        ["charts", file] => charts(file),
        ["set-chart-data", file, chart, csv, out] => {
            identifier(chart).and_then(|chart| set_chart_data(file, chart, csv, out))
        }
        ["add-chart", file, place, from, csv, x, y, w, h, out] => {
            identifier(from).and_then(|from| add_chart(file, place, from, csv, x, y, w, h, out))
        }
        ["media", file] => media(file),
        ["set-geometry", file, id, x, y, out] => set_geometry(file, id, x, y, None, out),
        ["set-geometry", file, id, x, y, w, h, out] => {
            set_geometry(file, id, x, y, Some((w, h)), out)
        }
        ["replace-media", file, target, image, out] => replace_media(file, target, image, out),
        ["tables", file] => tables(file),
        ["cells", file, table] => cells(file, table, false),
        ["cells", file, table, "--raw"] => cells(file, table, true),
        ["csv", file, table] => csv(file, table),
        ["organise", file] => organise(file, None),
        ["organise", file, table] => organise(file, Some(table)),
        ["formulas", file] => formulas(file, None),
        ["formulas", file, table] => formulas(file, Some(table)),
        ["set-cell", file, table, cell, value, out] => reference_position(cell)
            .and_then(|(row, column)| set_cell(file, table, row, column, value, out)),
        ["set-cell", file, table, row, column, value, out] => index(row)
            .and_then(|row| Ok((row, index(column)?)))
            .and_then(|(row, column)| set_cell(file, table, row, column, value, out)),
        ["set-cells", file, table, cell, csv, out] => reference_position(cell)
            .and_then(|(row, column)| set_cells(file, table, (row, column), csv, out)),
        ["set-formula", file, table, cell, formula, value, out] => reference_position(cell)
            .and_then(|(row, column)| set_formula(file, table, row, column, formula, value, out)),
        ["set-format", file, table, cell, format, out] => reference_position(cell)
            .and_then(|(row, column)| set_format(file, table, row, column, format, out)),
        ["set-width", file, table, column, points, out] => {
            index(column).and_then(|column| set_size(file, table, column, points, true, out))
        }
        ["set-height", file, table, row, points, out] => {
            index(row).and_then(|row| set_size(file, table, row, points, false, out))
        }
        ["delete-row", file, table, at, out] => {
            index(at).and_then(|at| delete_line(file, table, at, true, out))
        }
        ["delete-column", file, table, at, out] => {
            index(at).and_then(|at| delete_line(file, table, at, false, out))
        }
        ["merge", file, table, cell, rows, columns, out] => reference_position(cell)
            .and_then(|(row, column)| Ok((row, column, index(rows)?, index(columns)?)))
            .and_then(|(row, column, rows, columns)| {
                merge(file, table, row, column, rows, columns, out)
            }),
        ["unmerge", file, table, cell, out] => reference_position(cell)
            .and_then(|(row, column)| unmerge(file, table, row, column, out)),
        ["insert-column", file, table, at, out] => {
            index(at).and_then(|at| insert_column(file, table, at, out))
        }
        ["insert-row", file, table, at, out] => {
            index(at).and_then(|at| insert_row(file, table, at, out))
        }
        ["metadata", file] => metadata(file),
        ["annotations", file] => annotations(file),
        ["add-comment", file, storage, start, end, author, text, out] => identifier(storage)
            .and_then(|storage| {
                let range = |raw: &str| {
                    raw.parse::<u64>().map_err(|_| {
                        Error::Format(format!("{raw}: a text position is a whole number"))
                    })
                };
                Ok((storage, range(start)?, range(end)?))
            })
            .and_then(|(storage, start, end)| {
                add_comment(file, storage, start, end, author, text, out)
            }),
        ["duplicate", file, out] => duplicate(file, out),
        ["new", template, out] => new_document(template, out),
        ["fill-formula", file, table, from, to, out] => {
            fill_formula(file, table, from, to, None, out)
        }
        ["fill-formula", file, table, from, to, value, out] => {
            fill_formula(file, table, from, to, Some(value), out)
        }
        ["add-sheet", file, name, table, rows, columns, out] => index(rows)
            .and_then(|rows| Ok((rows, index(columns)?)))
            .and_then(|(rows, columns)| add_sheet(file, name, table, rows, columns, out)),
        ["add-table", file, sheet, name, rows, columns, out] => index(rows)
            .and_then(|rows| Ok((rows, index(columns)?)))
            .and_then(|(rows, columns)| add_table(file, sheet, name, rows, columns, None, out)),
        ["add-table", file, place, name, rows, columns, x, y, out] => index(rows)
            .and_then(|rows| Ok((rows, index(columns)?)))
            .and_then(|(rows, columns)| {
                add_table(file, place, name, rows, columns, Some((x, y)), out)
            }),
        ["add-text-box", file, place, text, x, y, w, h, out] => {
            add_text_box(file, place, "rectangle", text, x, y, w, h, out)
        }
        ["add-shape", file, place, outline, text, x, y, w, h, out] => {
            add_text_box(file, place, outline, text, x, y, w, h, out)
        }
        ["add-image", file, place, image, x, y, out] => {
            add_image(file, place, image, x, y, None, out)
        }
        ["add-image", file, place, image, x, y, w, h, out] => {
            add_image(file, place, image, x, y, Some((w, h)), out)
        }
        ["effects"] => {
            for (name, wire) in iwork::keynote::EFFECTS {
                println!("  {name:<28} {wire}");
            }
            Ok(())
        }
        ["set-transition", file, slide, effect, out] => {
            identifier(slide).and_then(|slide| set_transition(file, slide, effect, None, None, out))
        }
        ["set-transition", file, slide, effect, duration, out] => identifier(slide)
            .and_then(|slide| set_transition(file, slide, effect, Some(duration), None, out)),
        ["set-transition", file, slide, effect, duration, delay, out] => identifier(slide)
            .and_then(|slide| {
                set_transition(file, slide, effect, Some(duration), Some(delay), out)
            }),
        ["add-build", file, slide, drawable, kind, out] => identifier(slide)
            .and_then(|slide| Ok((slide, identifier(drawable)?)))
            .and_then(|(slide, drawable)| add_build(file, slide, drawable, kind, None, out)),
        ["add-build", file, slide, drawable, kind, effect, out] => identifier(slide)
            .and_then(|slide| Ok((slide, identifier(drawable)?)))
            .and_then(|(slide, drawable)| {
                add_build(file, slide, drawable, kind, Some(effect), out)
            }),
        ["add-slide", file, out] => add_slide(file, None, out),
        ["add-slide", file, layout, out] => add_slide(file, Some(layout), out),
        ["create", kind, out] => create_document(kind, out, None),
        ["create", kind, paper, out] => create_document(kind, out, Some(paper)),
        ["strip-previews", file, out] => strip_previews(file, out),
        ["sections", file] => sections(file),
        ["structure", file] => structure(file),
        ["slides", file] => slides(file),
        ["layouts", file] => layouts(file),
        ["set-notes", file, slide, new_text, out] => {
            identifier(slide).and_then(|slide| set_notes(file, slide, new_text, out))
        }
        ["skip-slide", file, slide, out] => {
            identifier(slide).and_then(|slide| skip_slide(file, slide, true, out))
        }
        ["unskip-slide", file, slide, out] => {
            identifier(slide).and_then(|slide| skip_slide(file, slide, false, out))
        }
        ["move-slide", file, slide, to, out] => identifier(slide)
            .and_then(|slide| index(to).map(|to| (slide, to)))
            .and_then(|(slide, to)| move_slide(file, slide, to, out)),
        ["duplicate-slide", file, slide, out] => {
            identifier(slide).and_then(|slide| duplicate_slide(file, slide, out))
        }
        ["properties"] => properties(),
        ["set-color", file, id, r, g, b, out] => set_color(file, id, r, g, b, out),
        ["paragraphs", file, storage] => {
            identifier(storage).and_then(|storage| paragraphs(file, storage))
        }
        _ => {
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn inspect(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    println!("{} document — {path}", doc.kind().as_str());
    // Which of the two shapes it is on disk, because a save puts it back in
    // that shape and a reader should know which one they have.
    println!("stored as {}", doc.package().form.as_str());

    // Pages has two kinds of document and they are not the same file at all:
    // one has a body storage the text flows through, the other has named page
    // templates and text only in boxes. Saying which, first, saves a reader the
    // question.
    if let Some(structure) = doc.structure() {
        println!(
            "{}, {} section(s), {:.0} × {:.0} pt{}",
            structure.mode.as_str(),
            structure.sections.len(),
            structure.setup.width,
            structure.setup.height,
            if structure.setup.facing_pages {
                ", facing pages"
            } else {
                ""
            }
        );
    }

    // Who this document is, where it came from and what it says it speaks —
    // the things a reader wants before any object census.
    let metadata = doc.metadata()?;
    if let Some(properties) = &metadata.properties {
        println!(
            "identity  {} (version {})",
            properties.document_uuid.as_deref().unwrap_or("no UUID"),
            properties.version_uuid.as_deref().unwrap_or("—")
        );
        if properties.stable_document_uuid != properties.document_uuid {
            println!(
                "          a copy of {}",
                properties.stable_document_uuid.as_deref().unwrap_or("—")
            );
        }
    }
    let mut origin = Vec::new();
    if let Some(template) = &metadata.template_identifier {
        origin.push(format!("from {template}"));
    }
    if let Some(locale) = &metadata.locale {
        origin.push(format!("locale {locale}"));
    }
    if let Some(first) = metadata.build_versions.first() {
        origin.push(first.clone());
    }
    if let Some(last) = metadata.build_versions.last() {
        if metadata.build_versions.len() > 1 {
            origin.push(format!("last written by {last}"));
        }
    }
    if !origin.is_empty() {
        println!("origin    {}", origin.join(", "));
    }

    // The review layer, always — "no comments" is a fact about the document.
    println!("review    {}", doc.annotations().summary());
    match doc.previews().len() {
        0 => println!("previews  none (the app draws them when it saves)"),
        n => println!("previews  {n}, drawn by the app and stale after an edit"),
    }

    println!("\n== package entries ==");
    for (name, data) in &doc.package().entries {
        println!("  {name:<48} {:>10} bytes", data.len());
    }

    let components = doc.components();
    if !components.is_empty() {
        println!("\n== components ({}) ==", components.len());
        for c in &components {
            println!(
                "  id={:<8} {:<28} {:<34} refs={}",
                c.identifier,
                c.preferred_name,
                c.stream_name(),
                c.external_reference_count
            );
        }
    }

    let data_files = doc.data_files();
    if !data_files.is_empty() {
        println!("\n== media ({}) ==", data_files.len());
        for d in &data_files {
            match d.entry_name() {
                Some(entry) => println!("  id={:<5} {entry}", d.identifier),
                None => println!(
                    "  id={:<5} (theme asset, not stored) {} {}",
                    d.identifier, d.original_name, d.asset_path
                ),
            }
        }
    }

    if let Some(last) = doc.last_object_identifier() {
        println!("\nhighest allocated object identifier: {last}");
    }

    println!("\n== objects by message type ==");
    let mut census: BTreeMap<u32, usize> = BTreeMap::new();
    for (_, object) in doc.objects() {
        for message in &object.messages {
            *census.entry(message.message_type).or_default() += 1;
        }
    }
    for (message_type, count) in &census {
        println!(
            "  {message_type:>6} x{count:<5} {}",
            registry::describe_in(doc.kind(), *message_type)
        );
    }
    println!(
        "\n{} objects across {} streams; names ending in ? are inferred, ?? unverified",
        census.values().sum::<usize>(),
        doc.stream_names().count()
    );
    Ok(())
}

fn metadata(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let metadata = doc.metadata()?;

    if let Some(properties) = &metadata.properties {
        println!("== {} ==", iwork::metadata::PROPERTIES);
        let show = |name: &str, value: &Option<String>| {
            println!("  {name:<34} {}", value.as_deref().unwrap_or("—"));
        };
        show("documentUUID", &properties.document_uuid);
        show("shareUUID", &properties.share_uuid);
        show("stableDocumentUUID", &properties.stable_document_uuid);
        show("privateUUID", &properties.private_uuid);
        show("versionUUID", &properties.version_uuid);
        show("revision", &properties.revision);
        show("fileFormatVersion", &properties.file_format_version);
        for (name, value) in [
            ("isMultiPage", properties.is_multi_page),
            (
                "hasExternalReferenceOrMissingData",
                properties.has_external_reference_or_missing_data,
            ),
            (
                "hasUnmaterializedRemoteData",
                properties.has_unmaterialized_remote_data,
            ),
        ] {
            match value {
                Some(value) => println!("  {name:<34} {value}"),
                None => println!("  {name:<34} —"),
            }
        }
        for key in &properties.other {
            println!("  {key:<34} (not named by this crate, carried through)");
        }
        if properties.document_uuid == properties.stable_document_uuid {
            println!("  — this document has never been copied: the lineage is itself");
        }
    } else {
        println!("no {}", iwork::metadata::PROPERTIES);
    }

    println!(
        "\n{:<36} {}",
        iwork::metadata::DOCUMENT_IDENTIFIER,
        metadata.document_identifier.as_deref().unwrap_or("—")
    );

    if metadata.build_versions.is_empty() {
        println!("{:<36} —", iwork::metadata::BUILD_VERSION_HISTORY);
    } else {
        println!("\n== {} ==", iwork::metadata::BUILD_VERSION_HISTORY);
        for (i, line) in metadata.build_versions.iter().enumerate() {
            println!("  {i}  {line}");
        }
    }

    println!("\n== document archive ==");
    for (name, value) in [
        ("locale (TSK)", &metadata.locale),
        ("creation locale (TSK)", &metadata.creation_locale),
        ("document language (TSA)", &metadata.document_language),
        ("template (TSA)", &metadata.template_identifier),
    ] {
        println!("  {name:<26} {}", value.as_deref().unwrap_or("—"));
    }
    match metadata.custom_format_list {
        Some(list) => println!(
            "  {:<26} object {list}, {} custom format(s)",
            "custom format list",
            doc.custom_formats().len()
        ),
        None => println!("  {:<26} —", "custom format list"),
    }
    if let Some(list) = metadata.tables_custom_format_list {
        println!("  {:<26} object {list}", "tables custom formats");
    }
    match metadata.annotation_author_storage {
        Some(storage) => println!("  {:<26} object {storage}", "annotation authors"),
        None => println!("  {:<26} —", "annotation authors"),
    }
    Ok(())
}

fn annotations(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let annotations = doc.annotations();

    match &annotations.author_storage {
        Some((identifier, stream)) => println!(
            "author storage: object {identifier} in {stream}, {} author(s)",
            annotations.authors.len()
        ),
        None => println!("no annotation author storage — every iWork document has one"),
    }
    for author in &annotations.authors {
        println!(
            "  {:<8} {:<28} {}{}",
            author.identifier,
            author.name.as_deref().unwrap_or("(unnamed)"),
            author
                .color
                .map(|c| c.to_string())
                .unwrap_or_else(|| "no colour".into()),
            if author.is_public_author {
                ", shared"
            } else {
                ""
            }
        );
    }

    if annotations.comments.is_empty() {
        println!("\nno comments");
    } else {
        println!("\n== comments ({}) ==", annotations.comments.len());
        for comment in &annotations.comments {
            println!(
                "  {:<8} {:<44} {}",
                comment.identifier,
                comment.anchor.as_str(),
                comment
                    .created
                    .map(iwork::table::format_date)
                    .unwrap_or_else(|| "no date".into())
            );
            if let Some(text) = &comment.text {
                println!("      {text}");
            }
            if !comment.replies.is_empty() {
                let listed: Vec<String> = comment.replies.iter().map(u64::to_string).collect();
                println!(
                    "      {} repl(y|ies): {}",
                    comment.replies.len(),
                    listed.join(", ")
                );
            }
        }
    }

    if annotations.changes.is_empty() {
        println!("\nno tracked changes");
    } else {
        println!("\n== tracked changes ({}) ==", annotations.changes.len());
        for change in &annotations.changes {
            println!(
                "  {:<8} {:<12} {}",
                change.identifier,
                change.kind.as_str(),
                match change.anchor {
                    Some((storage, field, index)) => format!(
                        "storage {storage} at {index} ({})",
                        iwork::text::table(field).map(|t| t.name).unwrap_or("?")
                    ),
                    None => "not anchored anywhere this crate walks".to_string(),
                }
            );
        }
        println!("  {} change session(s)", annotations.sessions.len());
    }

    if !annotations.tracked_storages.is_empty() {
        let listed: Vec<String> = annotations
            .tracked_storages
            .iter()
            .map(u64::to_string)
            .collect();
        println!(
            "\nstorages carrying tracked changes, which this crate will not edit: {}",
            listed.join(", ")
        );
    }
    if !annotations.unreached.is_empty() {
        println!("\nannotation objects nothing here points at:");
        for (identifier, message_type) in &annotations.unreached {
            println!(
                "  {identifier:<8} type {message_type} {}",
                registry::describe_in(doc.kind(), *message_type)
            );
        }
    }
    Ok(())
}

fn new_document(template: &str, out: &str) -> Result<(), Error> {
    let doc = Document::from_template(template)?;
    doc.save(out)?;
    let metadata = doc.metadata()?;
    println!("wrote {out} — a {} document", doc.kind().as_str());
    if let Some(properties) = &metadata.properties {
        println!(
            "  documentUUID       {}",
            properties.document_uuid.as_deref().unwrap_or("—")
        );
        println!(
            "  stableDocumentUUID {} (its own — a document from a template is not a copy of it)",
            properties.stable_document_uuid.as_deref().unwrap_or("—")
        );
    }
    match &metadata.template_identifier {
        Some(identifier) => println!("  template           {identifier}"),
        None => println!("  template           — (not one of the app's own, so nothing claimed)"),
    }
    Ok(())
}

fn create_document(kind: &str, out: &str, paper: Option<&str>) -> Result<(), Error> {
    let kind = match kind.to_ascii_lowercase().as_str() {
        "pages" => iwork::Kind::Pages,
        "numbers" => iwork::Kind::Numbers,
        "keynote" | "key" => iwork::Kind::Keynote,
        other => {
            return Err(Error::Format(format!(
                "{other}: create takes pages, numbers or keynote"
            )))
        }
    };
    let paper = match paper.map(str::to_ascii_lowercase).as_deref() {
        None => iwork::create::Paper::default(),
        Some("a4" | "iso" | "iso-a4") => iwork::create::Paper::A4,
        Some("letter" | "us" | "na-letter") => iwork::create::Paper::Letter,
        Some(other) => return Err(Error::Format(format!("{other}: paper is a4 or letter"))),
    };
    let doc = Document::new_on(kind, paper)?;
    doc.save(out)?;
    println!(
        "wrote {out} — a {} document made from nothing, {} object(s)",
        doc.kind().as_str(),
        doc.objects().count()
    );
    Ok(())
}

fn fill_formula(
    path: &str,
    table: &str,
    from: &str,
    to: &str,
    value: Option<&str>,
    out: &str,
) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let from = reference_position(from)?;
    let to = reference_position(to)?;
    let value = match value {
        Some(text) => Some(parse_cell_value(text)?),
        None => None,
    };
    doc.fill_formula(table, from, to, value)?;
    let shown = doc
        .table(table)
        .map(|t| t.value(to.0, to.1).to_text())
        .unwrap_or_default();
    doc.save(out)?;
    println!(
        "filled r{}c{} into r{}c{}, showing {shown:?}",
        from.0, from.1, to.0, to.1
    );
    println!("  the app recalculates it when a cell it reads changes");
    report_streams(&doc, out);
    Ok(())
}

fn add_sheet(
    path: &str,
    name: &str,
    table: &str,
    rows: usize,
    columns: usize,
    out: &str,
) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let sheet = doc.add_sheet(name, table, rows, columns)?;
    doc.save(out)?;
    println!("added sheet {name:?} (object {sheet}) with table {table:?}, {rows}×{columns}");
    report_streams(&doc, out);
    Ok(())
}

fn add_table(
    path: &str,
    container: &str,
    name: &str,
    rows: usize,
    columns: usize,
    at: Option<(&str, &str)>,
    out: &str,
) -> Result<(), Error> {
    let position = match at {
        Some((x, y)) => {
            let number = |raw: &str| {
                raw.parse::<f32>()
                    .map_err(|_| Error::Format(format!("{raw}: a position is a number of points")))
            };
            (number(x)?, number(y)?)
        }
        None => (0.0, 0.0),
    };
    let mut doc = Document::open(path)?;
    let table = doc.add_table_at(container, name, rows, columns, position)?;
    doc.save(out)?;
    println!("added table {name:?} (object {table}) to {container}, {rows}×{columns}");
    report_streams(&doc, out);
    Ok(())
}

/// `iwork add-text-box` — a text box on a slide, a sheet or a page.
#[allow(clippy::too_many_arguments)]
fn add_text_box(
    path: &str,
    where_: &str,
    outline: &str,
    text: &str,
    x: &str,
    y: &str,
    w: &str,
    h: &str,
    out: &str,
) -> Result<(), Error> {
    let number = |what: &str, raw: &str| {
        raw.parse::<f32>()
            .map_err(|_| Error::Format(format!("{raw}: {what} is a number of points")))
    };
    let outline = match outline {
        "rectangle" | "box" => Outline::Rectangle,
        "ellipse" | "oval" | "circle" => Outline::Ellipse,
        "line" | "rule" => Outline::Line,
        other => {
            return Err(Error::Format(format!(
                "{other}: an outline is rectangle, ellipse or line"
            )))
        }
    };
    let position = (number("a position", x)?, number("a position", y)?);
    let size = (number("a size", w)?, number("a size", h)?);
    let mut doc = Document::open(path)?;
    let shape = doc.add_shape(where_, outline, text, position, size)?;
    doc.save(out)?;
    let what = match outline {
        Outline::Rectangle if text.is_empty() => "rectangle",
        Outline::Rectangle => "text box",
        Outline::Ellipse => "ellipse",
        Outline::Line => "line",
    };
    println!(
        "added {what} {shape} at {},{} — {} × {}",
        position.0, position.1, size.0, size.1
    );
    report_streams(&doc, out);
    Ok(())
}

/// `iwork add-image` — a picture on a slide, a sheet or a page.
fn add_image(
    path: &str,
    where_: &str,
    image: &str,
    x: &str,
    y: &str,
    size: Option<(&str, &str)>,
    out: &str,
) -> Result<(), Error> {
    let number = |what: &str, raw: &str| {
        raw.parse::<f32>()
            .map_err(|_| Error::Format(format!("{raw}: {what} is a number of points")))
    };
    let position = (number("a position", x)?, number("a position", y)?);
    let size = match size {
        Some((w, h)) => Some((number("a size", w)?, number("a size", h)?)),
        None => None,
    };
    let bytes = std::fs::read(image).map_err(|e| Error::Format(format!("{image}: {e}")))?;
    let name = std::path::Path::new(image)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string());
    let mut doc = Document::open(path)?;
    let drawable = doc.add_image(where_, &bytes, &name, position, size)?;
    doc.save(out)?;
    println!(
        "added image {drawable} at {},{} — {} byte(s) as {name}",
        position.0,
        position.1,
        bytes.len()
    );
    report_streams(&doc, out);
    Ok(())
}

/// `iwork set-transition` — the effect a slide is left by.
fn set_transition(
    path: &str,
    slide: u64,
    effect: &str,
    duration: Option<&str>,
    delay: Option<&str>,
    out: &str,
) -> Result<(), Error> {
    let seconds = |what: &str, raw: Option<&str>| match raw {
        Some(raw) => raw
            .parse::<f64>()
            .map(Some)
            .map_err(|_| Error::Format(format!("{raw}: {what} is a number of seconds"))),
        None => Ok(None),
    };
    let edit = iwork::keynote::TransitionEdit {
        effect: effect.to_string(),
        duration: seconds("a duration", duration)?,
        delay: seconds("a delay", delay)?,
        ..Default::default()
    };
    let mut doc = Document::open(path)?;
    let now = doc.set_transition(slide, &edit)?;
    doc.save(out)?;
    if now.is_none() {
        println!("slide {slide} now leaves with no transition");
    } else {
        println!(
            "slide {slide} now leaves with {} ({}), {}s, {}",
            iwork::keynote::effect_name(&now.effect).unwrap_or("an effect this crate cannot name"),
            now.effect,
            now.duration,
            if now.automatic {
                format!("automatic after {}s", now.delay)
            } else {
                "on click".to_string()
            }
        );
    }
    report_streams(&doc, out);
    Ok(())
}

/// `iwork add-comment` — a comment on a range of text.
fn add_comment(
    path: &str,
    storage: u64,
    start: u64,
    end: u64,
    author: &str,
    text: &str,
    out: &str,
) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let comment = doc.add_comment(
        storage,
        start,
        end,
        &iwork::annotations::CommentEdit {
            author: author.to_string(),
            text: text.to_string(),
        },
    )?;
    doc.save(out)?;
    println!("added comment {comment} on storage {storage}, characters {start}..{end}");
    report_streams(&doc, out);
    Ok(())
}

/// `iwork add-build` — a drawable animated on or off a slide.
fn add_build(
    path: &str,
    slide: u64,
    drawable: u64,
    kind: &str,
    effect: Option<&str>,
    out: &str,
) -> Result<(), Error> {
    use iwork::keynote::{BuildEdit, BuildKind};
    let kind = match kind {
        "in" | "In" => BuildKind::In,
        "out" | "Out" => BuildKind::Out,
        other => {
            return Err(Error::Format(format!(
                "{other}: a build is `in` (onto the slide) or `out` (off it)"
            )))
        }
    };
    let mut doc = Document::open(path)?;
    let build = doc.add_build(
        slide,
        drawable,
        &BuildEdit {
            kind,
            effect: effect.unwrap_or_default().to_string(),
            ..Default::default()
        },
    )?;
    doc.save(out)?;
    println!(
        "added build {build}: drawable {drawable} animates {} with {}",
        match kind {
            BuildKind::In => "onto the slide",
            BuildKind::Out => "off the slide",
        },
        effect.unwrap_or_else(|| kind.default_effect())
    );
    report_streams(&doc, out);
    Ok(())
}

fn add_slide(path: &str, layout: Option<&str>, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let layout = match layout {
        Some(text) => Some(
            text.parse::<u64>()
                .map_err(|_| Error::Format(format!("{text}: a layout is an object id")))?,
        ),
        None => None,
    };
    let slide = doc.add_slide(layout)?;
    let slides = doc.slides().len();
    doc.save(out)?;
    println!(
        "added slide {} at position {} of {}",
        slide.identifier, slide.index, slides
    );
    report_streams(&doc, out);
    Ok(())
}

/// What a write touched, in the shape the other write commands report it.
fn report_streams(doc: &Document, out: &str) {
    let changed = doc.changed_streams();
    println!(
        "wrote {out} ({} of {} streams rewritten{})",
        changed.len(),
        doc.stream_names().count(),
        match changed.is_empty() {
            true => String::new(),
            false => format!(": {}", changed.join(", ")),
        }
    );
}

fn strip_previews(path: &str, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let gone = doc.strip_previews();
    doc.save(out)?;
    println!("wrote {out} ({gone} preview image(s) removed)");
    if gone > 0 {
        println!(
            "  the app draws them again the next time it saves; nothing in the \n  \
             document refers to them"
        );
    }
    Ok(())
}

fn duplicate(path: &str, out: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let identity = doc.save_as_new(out)?;
    println!("wrote {out}");
    println!("  documentUUID       {}", identity.document_uuid);
    println!("  shareUUID          {}", identity.share_uuid);
    println!("  privateUUID        {}", identity.private_uuid);
    println!("  versionUUID        {}", identity.version_uuid);
    println!("  revision           {}", identity.revision);
    println!(
        "  stableDocumentUUID {} (kept — this is what says where the copy came from)",
        identity.stable_document_uuid.as_deref().unwrap_or("—")
    );
    Ok(())
}

fn text(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    for storage in doc.text_storages() {
        println!("--- {} storage {} ---", storage.stream, storage.identifier);
        println!("{}", storage.text);
    }
    Ok(())
}

fn set_text(path: &str, identifier: u64, new_text: &str, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    report_edit(doc.set_text(identifier, new_text)?);
    save(&doc, out)
}

fn insert_text(
    path: &str,
    identifier: u64,
    at: u64,
    new_text: &str,
    out: &str,
) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    report_edit(doc.insert_text(identifier, at, new_text)?);
    save(&doc, out)
}

fn delete_text(
    path: &str,
    identifier: u64,
    range: std::ops::Range<u64>,
    out: &str,
) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    report_edit(doc.delete_text(identifier, range)?);
    save(&doc, out)
}

fn storages(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    for storage in doc.storages() {
        println!(
            "  {:<9} {:<18} {:>6} unit(s), {} paragraph(s)   {}",
            storage.identifier,
            storage.kind_name(),
            storage.length,
            storage.paragraphs,
            storage.stream
        );
        for table in &storage.tables {
            println!(
                "      {:>2} {:<28} {:<10} {} entr(y|ies)",
                table.field,
                table.name,
                format!("{:?}", table.anchoring).to_lowercase(),
                table.entries
            );
        }
        if let Some(field) = storage.unknown_field {
            println!("      !! field {field} is not a table this crate knows — edits refused");
        }
    }
    Ok(())
}

fn links(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let fields = doc.smart_fields();
    if fields.is_empty() {
        println!("no smart fields");
        return Ok(());
    }
    for field in fields {
        println!(
            "  storage {:<9} {:>5}..{:<5} {:<34} {:?}",
            field.storage,
            field.range.start,
            field.range.end,
            registry::lookup(field.message_type)
                .map(|e| e.name)
                .unwrap_or("unknown"),
            field.text
        );
        if let Some(payload) = &field.payload {
            let arrow = if field.message_type == 2032 {
                "->"
            } else {
                " ="
            };
            println!("      {arrow} {payload}   (object {})", field.object);
        }
    }
    Ok(())
}

/// What the edit did to the things anchored into the storage. Printed because
/// it is the whole point: an edit that moves nothing has probably missed
/// something, and one that drops a great deal has probably taken too much.
fn report_edit(edit: iwork::TextEdit) {
    let tables: Vec<String> = edit
        .report
        .tables
        .iter()
        .map(|field| match iwork::text::table(*field) {
            Some(t) => format!("{field} {}", t.name),
            None => field.to_string(),
        })
        .collect();
    println!(
        "storage {}: {} unit(s) removed at {}, {} inserted; \
         {} run(s) moved, {} dropped, {} added",
        edit.storage,
        edit.edit.removed,
        edit.edit.at,
        edit.edit.inserted,
        edit.report.moved,
        edit.report.dropped,
        edit.report.added
    );
    if !tables.is_empty() {
        println!("  tables rewritten: {}", tables.join(", "));
    }
}

/// Write the document and say which streams that actually rewrote.
///
/// Worth printing: it is the difference between "I edited a style" and "I
/// rewrote the whole document", and only one of those is easy to review.
fn save(doc: &Document, out: &str) -> Result<(), Error> {
    let changed = doc.changed_streams();
    doc.save(out)?;
    match changed.len() {
        0 => println!("wrote {out} (no stream changed)"),
        n => println!(
            "wrote {out} ({n} of {} streams rewritten: {})",
            doc.stream_names().count(),
            changed.join(", ")
        ),
    }
    Ok(())
}

fn objects(path: &str, filter: Option<u32>) -> Result<(), Error> {
    let doc = Document::open(path)?;
    for (stream, object) in doc.objects() {
        let message_type = object.message_type();
        if filter.is_some_and(|t| t != message_type) {
            continue;
        }
        println!(
            "  id={:<8} type={:<6} {:<34} {:>7} bytes  {stream}",
            object.identifier,
            message_type,
            registry::describe_in(doc.kind(), message_type),
            object.payload().len()
        );
    }
    Ok(())
}

/// One line of preview text, with the characters that are not text named.
fn oneline(text: &str, width: usize) -> String {
    let mut out = String::new();
    for character in text.chars() {
        if out.chars().count() >= width {
            out.push('…');
            break;
        }
        match character {
            '\n' | '\r' | '\u{000C}' => out.push('⏎'),
            '\u{FFFC}' => out.push('▨'),
            '\u{0004}' => out.push('§'),
            '\u{0005}' => out.push('¶'),
            '\t' => out.push('→'),
            c => out.push(c),
        }
    }
    out
}

fn sections(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let Some(structure) = doc.structure() else {
        println!("not a Pages document — sections are a TP archive");
        return Ok(());
    };
    let body = structure
        .body_storage
        .map(|id| doc.storage_text(id).unwrap_or_default())
        .unwrap_or_default();

    println!(
        "{} document, {} section(s)",
        structure.mode.as_str(),
        structure.sections.len()
    );
    for section in &structure.sections {
        let name = if section.name.is_empty() {
            String::new()
        } else {
            format!(" \"{}\"", section.name)
        };
        println!(
            "\nsection {} id={}{} — [{}, {}), {} unit(s)",
            section.index + 1,
            section.identifier,
            name,
            section.start,
            section.end,
            section.length()
        );
        let text = slice_units(&body, section.start, section.end);
        if !text.is_empty() {
            println!("  text  {}", oneline(&text, 64));
        }
        println!("  pages {}", section.numbering());
        let mut switches = Vec::new();
        if section.inherits_header_footer {
            switches.push("inherits the previous header and footer");
        }
        if section.first_page_different {
            switches.push("first page different");
        }
        if section.even_odd_different {
            switches.push("even and odd pages different");
        }
        if section.hides_header_footer_on_first_page {
            switches.push("first page hides the header and footer");
        }
        if section.has_background {
            switches.push("has a background fill");
        }
        if section.hyperlink_uuid {
            switches.push("can be linked to");
        }
        if !switches.is_empty() {
            println!("  {}", switches.join(", "));
        }
        for (slot, page) in iwork::pages::TemplatePage::ALL.iter().enumerate() {
            let Some(template) = section.templates[slot] else {
                continue;
            };
            let zones: Vec<String> = structure
                .header_footers
                .iter()
                .filter(|hf| hf.section_template == template)
                .map(|hf| {
                    let text = if hf.text.is_empty() {
                        "empty".to_string()
                    } else {
                        format!("\"{}\"", oneline(&hf.text, 28))
                    };
                    // A lone ▨ is not much of a report; say what stands there.
                    let numbers: Vec<String> = hf
                        .numbers
                        .iter()
                        .map(|n| format!("{} as {}", n.kind_name(), n.format_name))
                        .collect();
                    format!(
                        "    {:<6} {:<6} id={:<8} {text}{}{}",
                        hf.kind(),
                        hf.zone.as_str(),
                        hf.storage,
                        if numbers.is_empty() { "" } else { " — " },
                        numbers.join(", ")
                    )
                })
                .collect();
            println!("  {} page template id={template}", page.as_str());
            for zone in zones {
                println!("{zone}");
            }
        }
    }
    Ok(())
}

/// UTF-16 code units `start..end` of a string, for printing a section.
fn slice_units(text: &str, start: u64, end: u64) -> String {
    let units: Vec<u16> = text.encode_utf16().collect();
    let from = (start as usize).min(units.len());
    let to = (end as usize).min(units.len());
    String::from_utf16_lossy(&units[from..to])
}

// -- Keynote -----------------------------------------------------------------

fn slides(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let Some(show) = doc.show() else {
        println!("not a Keynote document — a show is a KN archive");
        return Ok(());
    };

    let theme = if show.theme_name.is_empty() {
        "unnamed theme".to_string()
    } else {
        format!("theme {}", show.theme_name)
    };
    println!(
        "{} slide(s), {} layout(s), {} × {} points, {}",
        show.slides.len(),
        show.layouts.len(),
        show.width,
        show.height,
        theme
    );
    let shown = show.numbers_shown_on();
    let mut playback = vec![match shown {
        0 => "slide numbers hidden".to_string(),
        n if n == show.slides.len() => "slide numbers shown".to_string(),
        n => format!("slide numbers shown on {n} of {}", show.slides.len()),
    }];
    playback.push(format!("presentation {}", show.mode_name()));
    if show.loop_presentation {
        playback.push("loops".into());
    }
    if show.plays_on_open {
        playback.push("plays on open".into());
    }
    if show.idle_timer_active {
        // Seconds on the wire; the app's `maximum idle duration` is minutes.
        playback.push(format!(
            "restarts after {}s idle ({} min)",
            show.idle_timer_delay,
            show.idle_minutes()
        ));
    }
    if show.mode == 1 {
        playback.push(format!(
            "self-playing: {}s between slides, {}s between builds",
            show.autoplay_transition_delay, show.autoplay_build_delay
        ));
    }
    println!("  {}", playback.join(", "));
    if let Some(track) = &show.soundtrack {
        println!(
            "  soundtrack id={} {} track(s), volume {}, {}",
            track.identifier,
            track.tracks(),
            track.volume,
            track.mode_name()
        );
        for (index, media) in track.media.iter().enumerate() {
            println!("    track {} media {}", index + 1, media);
        }
    }
    if let Some(recording) = &show.recording {
        println!(
            "  recorded presentation id={} {}s, {} event track(s){} \
             — read only, never authored",
            recording.identifier,
            recording.duration,
            recording.event_tracks.len(),
            match recording.movie_track {
                Some(id) => format!(", movie track {id}"),
                None => String::new(),
            }
        );
    }
    for source in &show.live_video_sources {
        println!(
            "  live video source id={} \"{}\"{}{} — read only, never authored",
            source.identifier,
            source.name,
            if source.is_default { ", default" } else { "" },
            if source.listed { "" } else { ", unlisted" }
        );
    }

    for slide in &show.slides {
        let number = match slide.number {
            Some(n) => format!("slide {n}"),
            None => "skipped".to_string(),
        };
        let layout = if slide.layout_name.is_empty() {
            String::new()
        } else {
            format!(" on \"{}\"", slide.layout_name)
        };
        println!(
            "\n[{}] {} id={} node={}{} — {}",
            slide.index, number, slide.identifier, slide.node, layout, slide.stream
        );
        for text in &slide.texts {
            let shown = match text.role {
                iwork::keynote::Role::Title if !slide.title_showing() => " (not shown)",
                iwork::keynote::Role::Body if !slide.body_showing() => " (not shown)",
                _ => "",
            };
            println!(
                "  {:<18} storage {}{} {}",
                text.role.as_str(),
                text.storage,
                shown,
                oneline(&text.text, 56)
            );
        }
        if !slide.transition.is_none() {
            let transition = &slide.transition;
            println!(
                "  transition         {} {}s, {}",
                transition.effect,
                transition.duration,
                if transition.automatic {
                    format!("automatic after {}s", transition.delay)
                } else {
                    "on click".to_string()
                }
            );
            // The parameters that belong to the effect rather than the timing.
            // Only what is on the wire is printed: Keynote writes the ones the
            // chosen effect has and no others.
            let mut detail = Vec::new();
            if transition.direction != 0 {
                detail.push(format!("direction {}", transition.direction_name()));
            }
            if let Some(color) = transition.color {
                detail.push(format!("through {color}"));
            }
            if transition.rtl {
                detail.push("right-to-left writing".into());
            }
            detail.extend(transition.parameters.describe());
            if !detail.is_empty() {
                println!("                     {}", detail.join(", "));
            }
            if !transition.unknown_parameters.is_empty() {
                println!(
                    "                     unknown parameter field(s) {:?} — \
                     the 15.3.1 schema does not name them",
                    transition.unknown_parameters
                );
            }
        }
        if !slide.builds.is_empty() || !slide.build_chunks.is_empty() {
            println!(
                "  builds             {} build(s), {} chunk(s)",
                slide.builds.len(),
                slide.build_chunks.len()
            );
            for build in &slide.builds {
                println!(
                    "    build {} {}{} on drawable {}{}",
                    build.identifier,
                    match build.animation.animation_type.as_str() {
                        "In" => "in ",
                        "Out" => "out ",
                        _ => "",
                    },
                    if build.animation.effect.is_empty() {
                        "(no effect)"
                    } else {
                        &build.animation.effect
                    },
                    match build.drawable {
                        Some(id) => id.to_string(),
                        None => "?".to_string(),
                    },
                    if build.delivery.is_empty() {
                        String::new()
                    } else {
                        format!(", delivery {}", build.delivery)
                    }
                );
            }
        }
        println!(
            "  drawables          {} owned, {} in z-order",
            slide.drawables.len(),
            slide.z_order.len()
        );
    }
    Ok(())
}

fn layouts(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let Some(show) = doc.show() else {
        println!("not a Keynote document — a slide layout is a KN archive");
        return Ok(());
    };
    let used: std::collections::BTreeMap<u64, usize> =
        show.slides.iter().fold(Default::default(), |mut acc, s| {
            if let Some(layout) = s.layout {
                *acc.entry(layout).or_default() += 1;
            }
            acc
        });
    println!("{} slide layout(s)", show.layouts.len());
    for layout in &show.layouts {
        let count = used.get(&layout.identifier).copied().unwrap_or(0);
        println!(
            "\n[{}] \"{}\" id={} node={} — {} slide(s) use it",
            layout.index + 1,
            layout.name,
            layout.identifier,
            layout.node,
            count
        );
        for placeholder in &layout.placeholders {
            println!(
                "  {:<14} id={} {}{}",
                placeholder.kind.as_str(),
                placeholder.identifier,
                if placeholder.shown {
                    ""
                } else {
                    "(not shown) "
                },
                oneline(&placeholder.text, 48)
            );
        }
        println!(
            "  {:<14} {} drawable(s), {} outline level style(s)",
            "carries",
            layout.drawables,
            layout.body_paragraph_styles.len()
        );
    }
    Ok(())
}

fn set_notes(path: &str, slide: u64, new_text: &str, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let edit = doc.set_presenter_notes(slide, new_text)?;
    report_edit(edit);
    save(&doc, out)
}

fn skip_slide(path: &str, slide: u64, skipped: bool, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let changed = doc.set_slide_skipped(slide, skipped)?;
    println!(
        "slide {slide} is {}{}",
        if skipped { "skipped" } else { "in the show" },
        if changed { "" } else { " (it already was)" }
    );
    save(&doc, out)
}

fn move_slide(path: &str, slide: u64, to: usize, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let landed = doc.move_slide(slide, to)?;
    println!("slide {slide} is now at position {landed}");
    save(&doc, out)
}

fn duplicate_slide(path: &str, slide: u64, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let copy = doc.duplicate_slide(slide)?;
    println!(
        "copied slide {} to {} (node {}) at position {}",
        copy.source, copy.identifier, copy.node, copy.index
    );
    println!("  {} object(s) into {}", copy.objects, copy.stream);
    if copy.media > 0 {
        println!("  {} media file(s) shared with the original", copy.media);
    }
    if copy.declarations > 0 {
        println!("  {} external reference(s) declared", copy.declarations);
    }
    save(&doc, out)
}

fn structure(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let Some(s) = doc.structure() else {
        println!("not a Pages document — the structure is a TP archive");
        return Ok(());
    };

    println!("mode      {}", s.mode.as_str());
    println!(
        "paper     {:.0} × {:.0} pt {} ({}), scale {}",
        s.setup.width,
        s.setup.height,
        if s.setup.portrait() {
            "portrait"
        } else {
            "landscape"
        },
        if s.setup.paper_id.is_empty() {
            "no paper id"
        } else {
            &s.setup.paper_id
        },
        s.setup.scale
    );
    println!(
        "margins   left {:.0} right {:.0} top {:.0} bottom {:.0}, header {:.0} footer {:.0}",
        s.setup.left_margin,
        s.setup.right_margin,
        s.setup.top_margin,
        s.setup.bottom_margin,
        s.setup.header_margin,
        s.setup.footer_margin
    );
    let mut switches = Vec::new();
    if s.setup.facing_pages {
        switches.push("facing pages".to_string());
    }
    if s.setup.single_header_footer {
        switches.push("one header and footer for the whole document".to_string());
    }
    if !s.setup.headers_shown {
        switches.push("headers hidden".to_string());
    }
    if !s.setup.footers_shown {
        switches.push("footers hidden".to_string());
    }
    if s.setup.rtl {
        switches.push("right to left".to_string());
    }
    if s.setup.body_vertical {
        switches.push("body laid out vertically".to_string());
    }
    if !s.setup.language.is_empty() {
        switches.push(format!("language {}", s.setup.language));
    }
    if !s.setup.template.is_empty() {
        switches.push(format!("from template {}", s.setup.template));
    }
    if !switches.is_empty() {
        println!("          {}", switches.join(", "));
    }

    println!("sections  {} — `iwork sections` for each", s.sections.len());
    let with_text = s.header_footers.iter().filter(|hf| !hf.text.is_empty());
    println!(
        "headers   {} storage(s), {} with text",
        s.header_footers.len(),
        with_text.count()
    );

    if s.page_templates.is_empty() {
        println!("templates none — a word-processing document has no page templates");
    } else {
        println!("templates {}", s.page_templates.len());
        for template in &s.page_templates {
            let mut notes = Vec::new();
            if template.matches_previous_page {
                notes.push("matches the previous page");
            }
            if template.hides_headers_footers {
                notes.push("hides headers and footers");
            }
            if template.has_background {
                notes.push("has a background fill");
            }
            println!(
                "    id={:<8} \"{}\" — {} drawable(s), {} placeholder(s){}{}",
                template.identifier,
                template.name,
                template.drawables,
                template.placeholders,
                if notes.is_empty() { "" } else { "; " },
                notes.join(", ")
            );
        }
    }

    if s.threads.is_empty() {
        println!("threads   none");
    } else {
        println!("threads   {}", s.threads.len());
        for thread in &s.threads {
            let boxes: Vec<String> = thread.boxes.iter().map(u64::to_string).collect();
            println!(
                "    id={:<8} number {} — storage {} through {} box(es): {}",
                thread.identifier,
                thread.number,
                thread
                    .storage
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "none".into()),
                thread.boxes.len(),
                boxes.join(" → ")
            );
        }
    }

    if s.contents.is_empty() {
        println!("contents  none");
    } else {
        println!("contents  {} settings archive(s)", s.contents.len());
        for toc in &s.contents {
            let shown = toc.rules.iter().filter(|r| r.show).count();
            println!(
                "    id={:<8} \"{}\" scope {} — {} of {} paragraph style(s) included{}",
                toc.identifier,
                toc.name,
                toc.scope,
                shown,
                toc.rules.len(),
                match toc.placed_in {
                    Some(owner) => format!(", placed in {owner}"),
                    None => ", the document's own".to_string(),
                }
            );
            for (heading, page) in &toc.entries {
                println!("      p{page:<4} {heading}");
            }
        }
    }

    let tracking = &s.change_tracking;
    if tracking.is_default() {
        println!("tracking  off, and the six display switches are at their defaults");
    } else {
        println!(
            "tracking  {}, {} session(s){}, markup {}, deletions {}, bubbles {}, \
             change bars {}, format changes {}, annotations {}",
            if tracking.enabled { "on" } else { "off" },
            tracking.sessions,
            if tracking.has_recent_session {
                ", one most recent"
            } else {
                ""
            },
            tracking.show_markup,
            tracking.show_deletions,
            tracking.bubbles_visibility,
            tracking.change_bars,
            tracking.format_changes,
            tracking.annotations_visible
        );
    }

    let notes = &s.footnote_settings;
    println!(
        "footnotes {} as {}, numbered {}, gap {} pt — {} in the text",
        notes.kind_name(),
        notes.format_name(),
        notes.numbering_name(),
        notes.gap,
        if s.footnotes.is_empty() {
            "none".to_string()
        } else {
            s.footnotes.len().to_string()
        }
    );
    for note in &s.footnotes {
        println!(
            "    storage {} at {} → {} {}",
            note.storage,
            note.index,
            note.body
                .map(|b| b.to_string())
                .unwrap_or_else(|| "no body".into()),
            oneline(&note.text, 48)
        );
    }

    println!(
        "bookmarks {}",
        if s.bookmarks.is_empty() {
            "none".to_string()
        } else {
            s.bookmarks.len().to_string()
        }
    );
    for (storage, index, object) in &s.bookmarks {
        println!(
            "    storage {storage} at {index} → {}",
            object
                .map(|o| o.to_string())
                .unwrap_or_else(|| "nothing".into())
        );
    }

    if let Some(body) = s.body_storage {
        let layouts = doc.column_layouts(body);
        if layouts.is_empty() {
            println!("columns   none recorded on the body storage");
        } else {
            println!("columns   {} layout(s) on the body storage", layouts.len());
            for layout in &layouts {
                // Widths and gaps are fractions of the text width, not
                // points, so they are printed as percentages.
                let shape = match (&layout.equal, &layout.unequal) {
                    (Some((count, gap)), _) => {
                        format!("{count} equal column(s), gap {:.1}%", gap * 100.0)
                    }
                    (_, Some(_)) => {
                        let parts: Vec<String> = layout
                            .fractions()
                            .iter()
                            .map(|f| format!("{:.1}%", f * 100.0))
                            .collect();
                        format!(
                            "{} unequal column(s): {}",
                            layout.count(),
                            parts.join(" | ")
                        )
                    }
                    _ if layout.none => "no columns".to_string(),
                    _ => "one column".to_string(),
                };
                println!(
                    "    [{}, {}) style {} — {shape}",
                    layout.start,
                    layout.end,
                    layout
                        .style
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "none".into())
                );
            }
        }
    }
    Ok(())
}

/// The named style properties, with how far each name can be trusted.
fn tables(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let tables = doc.tables();
    if tables.is_empty() {
        println!("no tables");
        return Ok(());
    }
    for table in &tables {
        let where_ = match &table.sheet {
            Some(sheet) => format!("sheet {sheet}"),
            None => table.stream.clone(),
        };
        println!(
            "table {:<8} {:<24} {}×{} in {where_}",
            table.identifier, table.name, table.rows, table.columns
        );
        println!(
            "  headers: {} row(s){}, {} column(s){}, {} footer row(s)",
            table.header_rows,
            if table.header_rows_frozen {
                " frozen"
            } else {
                ""
            },
            table.header_columns,
            if table.header_columns_frozen {
                " frozen"
            } else {
                ""
            },
            table.footer_rows
        );
        let hidden_rows = hidden_list(&table.row_extents);
        let hidden_columns = hidden_list(&table.column_extents);
        // The counts come from the hidden-state extents, not from the model's
        // count fields (14/15/40/41/42): Numbers no longer maintains those,
        // and `numbers-hidden.numbers` — three rows and a column hidden by
        // hand — has every one of them at zero.
        println!(
            "  hidden: {} row(s) ({} by the user, {} by a filter), {} column(s) ({} by the user)",
            table.row_states.user_hidden.len() + table.row_states.filtered.len(),
            table.row_states.user_hidden.len(),
            table.row_states.filtered.len(),
            table.column_states.user_hidden.len() + table.column_states.filtered.len(),
            table.column_states.user_hidden.len()
        );
        if !hidden_rows.is_empty() {
            println!("  hidden rows: {}", hidden_rows.join(", "));
        }
        if !hidden_columns.is_empty() {
            println!("  hidden columns: {}", hidden_columns.join(", "));
        }
        println!(
            "  default size: {} × {} pt; {} stored cell(s)",
            table.default_column_width,
            table.default_row_height,
            table.cells().len()
        );
        let sized_rows = table
            .row_extents
            .iter()
            .filter(|e| e.size.is_some())
            .count();
        let sized_columns = table
            .column_extents
            .iter()
            .filter(|e| e.size.is_some())
            .count();
        if sized_rows + sized_columns > 0 {
            println!("  explicitly sized: {sized_rows} row(s), {sized_columns} column(s)");
        }
        for merge in &table.merges {
            println!(
                "  merged: {} spanning {} row(s) × {} column(s)",
                reference_name(merge.row, merge.column),
                merge.rows,
                merge.columns
            );
        }
        organisation_summary(table, "  ");
        for problem in &table.problems {
            println!("  problem: {problem}");
        }
    }
    Ok(())
}

fn drawables(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let all = doc.drawables();
    if all.is_empty() {
        println!("no drawables");
        return Ok(());
    }
    let by_id: BTreeMap<u64, &iwork::Drawable> = all.iter().map(|d| (d.identifier, d)).collect();
    let mut place = String::new();

    for drawable in &all {
        let where_ = drawable.placement.as_str();
        if where_ != place {
            println!("== {where_} ==");
            place = where_;
        }
        let mask = drawable.mask().and_then(|id| by_id.get(&id)).copied();
        let frame = drawable.frame(mask);
        let mut what = drawable.kind.as_str().to_string();
        if drawable
            .path_source
            .as_ref()
            .is_some_and(|p| p.looks_like_a_line())
        {
            what = "line".to_string();
        }
        println!(
            "  {:<8} z{:<3} {:<12} {:>8.1},{:<8.1} {:>8.1} × {:<8.1}{}",
            drawable.identifier,
            drawable.z,
            what,
            frame.x,
            frame.y,
            frame.width,
            frame.height,
            match drawable.geometry.angle {
                angle if angle != 0.0 => format!("  {angle:.1}°"),
                _ => String::new(),
            }
        );
        let mut notes: Vec<String> = Vec::new();
        if drawable.locked {
            notes.push("locked".into());
        }
        if drawable.aspect_ratio_locked {
            notes.push("aspect locked".into());
        }
        if drawable.geometry.flags != 3 {
            notes.push(format!("geometry flags {}", drawable.geometry.flags));
        }
        if let Some(parent) = drawable.parent {
            notes.push(format!("parent {parent}"));
        }
        if let Some(text) = drawable.text {
            notes.push(format!("text storage {text}"));
        }
        if let Some(comment) = drawable.comment {
            notes.push(format!("comment {comment}"));
        }
        if !drawable.pencil_annotations.is_empty() {
            notes.push(format!(
                "{} pencil annotation(s)",
                drawable.pencil_annotations.len()
            ));
        }
        if !notes.is_empty() {
            println!("      {}", notes.join(", "));
        }
        if let Some(link) = &drawable.hyperlink {
            println!("      link {link}");
        }
        if let Some(description) = &drawable.description {
            println!("      description {description:?}");
        }
        if !drawable.children.is_empty() {
            let list: Vec<String> = drawable.children.iter().map(u64::to_string).collect();
            println!("      children (back to front): {}", list.join(", "));
        }
        if let Some(source) = &drawable.path_source {
            println!(
                "      path: {:?}{}{}",
                source.kind,
                match source.natural_size {
                    Some((w, h)) => format!(" natural {w:.1} × {h:.1}"),
                    None => String::new(),
                },
                match source.elements {
                    0 => String::new(),
                    n => format!(", {n} element(s)"),
                }
            );
        }
        if let Some(media) = &drawable.media {
            let name = media
                .data
                .and_then(|id| doc.data_files().into_iter().find(|d| d.identifier == id))
                .map(|d| {
                    if d.stored_name.is_empty() {
                        format!("{} (theme asset)", d.original_name)
                    } else {
                        format!("Data/{}", d.stored_name)
                    }
                })
                .unwrap_or_else(|| "none".to_string());
            println!(
                "      media: {name}{}{}{}",
                match media.natural_size {
                    Some((w, h)) => format!(", natural {w:.0} × {h:.0}"),
                    None => String::new(),
                },
                if media.is_placeholder() {
                    ", placeholder"
                } else {
                    ""
                },
                if media.was_replaced() {
                    ", replaced"
                } else {
                    ""
                }
            );
            let mut playback: Vec<String> = Vec::new();
            if let Some((start, end)) = media.trim {
                playback.push(format!("trimmed {start:.2}s–{end:.2}s"));
            }
            if let Some(time) = media.poster_time {
                playback.push(format!("poster at {time:.2}s"));
            }
            if let Some(volume) = media.volume {
                playback.push(format!("volume {:.0}%", volume * 100.0));
            }
            if let Some(option) = media.loop_option {
                playback.push(format!("loop {option}"));
            }
            if media.audio_only {
                playback.push("audio only".into());
            }
            if media.live_video {
                playback.push("live video source".into());
            }
            if let Some(url) = &media.remote_url {
                playback.push(format!("streams from {url}"));
            }
            if !playback.is_empty() {
                println!("      playback: {}", playback.join(", "));
            }
        }
        if !drawable.extensions.is_empty() {
            println!("      carries: {}", drawable.extensions.join(", "));
        }
        if let Some(state) = &drawable.edit_state {
            if let Some(mask) = state.mask {
                println!(
                    "      mask {mask}{}",
                    if state.crops {
                        " (crops)"
                    } else {
                        " (identity)"
                    }
                );
            }
            for objection in state.objections() {
                println!("      edit state: {objection}");
            }
        }
        if let Some(style) = drawable.style.and_then(|id| doc.object_style(id)) {
            let mut parts: Vec<String> = Vec::new();
            if let Some(name) = &style.name {
                parts.push(format!("{name:?}"));
            }
            if let Some(fill) = &style.fill {
                parts.push(match fill {
                    iwork::drawable::Fill::Color(colour) => format!("fill {colour}"),
                    other => format!("fill {}", other.as_str()),
                });
            }
            if let Some(stroke) = &style.stroke {
                parts.push(format!(
                    "stroke {:?} {:.1}pt{}",
                    stroke.pattern,
                    stroke.width,
                    match stroke.color {
                        Some(colour) => format!(" {colour}"),
                        None => String::new(),
                    }
                ));
            }
            if let Some(opacity) = style.opacity {
                parts.push(format!("opacity {:.0}%", opacity * 100.0));
            }
            if let Some(shadow) = &style.shadow {
                parts.push(format!(
                    "shadow {}{:.0}° {:.0}pt r{}",
                    if shadow.enabled { "" } else { "off " },
                    shadow.angle,
                    shadow.offset,
                    shadow.radius
                ));
            }
            if let Some(reflection) = style.reflection {
                parts.push(format!("reflection {:.0}%", reflection * 100.0));
            }
            println!(
                "      style {}{}: {}",
                style.identifier,
                match style.override_count {
                    Some(n) if n > 0 => format!(" (+{n} override(s))"),
                    _ => String::new(),
                },
                if parts.is_empty() {
                    "nothing this crate reads".to_string()
                } else {
                    parts.join(", ")
                }
            );
        }
    }
    println!("\n{} drawable(s)", all.len());
    Ok(())
}

/// Every media file the package registers, and what points at it.
fn media(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let files = doc.data_files();
    if files.is_empty() {
        println!("no media");
        return Ok(());
    }
    let drawables = doc.drawables();
    for file in &files {
        let users: Vec<String> = drawables
            .iter()
            .filter(|d| d.media.as_ref().and_then(|m| m.data) == Some(file.identifier))
            .map(|d| format!("{} ({})", d.identifier, d.kind.as_str()))
            .collect();
        let posters: Vec<String> = drawables
            .iter()
            .filter(|d| d.media.as_ref().and_then(|m| m.poster) == Some(file.identifier))
            .map(|d| format!("{} poster", d.identifier))
            .collect();
        let stored = match file.entry_name() {
            Some(entry) => match doc.package().get(&entry) {
                Some(bytes) => format!("{entry} ({} bytes)", bytes.len()),
                None => format!("{entry} — MISSING from the package"),
            },
            None => format!(
                "{} in {} (theme asset)",
                file.original_name, file.asset_path
            ),
        };
        println!("  {:<6} {stored}", file.identifier);
        println!(
            "         digest {} {}",
            file.digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            match file
                .entry_name()
                .and_then(|e| doc.package().get(&e).map(iwork::media::sha1))
            {
                Some(actual) if actual.as_slice() == file.digest => "(matches the bytes)",
                Some(_) => "(DOES NOT match the bytes)",
                None => "",
            }
        );
        let used = [users, posters].concat();
        if !used.is_empty() {
            println!("         used by {}", used.join(", "));
        }
    }
    println!("\n{} media file(s)", files.len());
    Ok(())
}

fn set_geometry(
    path: &str,
    id: &str,
    x: &str,
    y: &str,
    size: Option<(&str, &str)>,
    out: &str,
) -> Result<(), Error> {
    let identifier = identifier(id)?;
    let number = |text: &str| -> Result<f32, Error> {
        text.parse::<f32>()
            .map_err(|_| Error::Format(format!("'{text}' is not a number of points")))
    };
    let position = Some((number(x)?, number(y)?));
    let size = match size {
        Some((w, h)) => Some((number(w)?, number(h)?)),
        None => None,
    };

    let mut doc = Document::open(path)?;
    let before = doc
        .drawable(identifier)
        .ok_or(Error::NoSuchObject(identifier))?;
    if before.locked {
        println!("note: drawable {identifier} is locked, so the app will not let a user move it");
    }
    let change = doc.set_geometry(identifier, position, size)?;
    println!(
        "{} {}: {:.1},{:.1} {:.1} × {:.1} -> {:.1},{:.1} {:.1} × {:.1}",
        before.kind.as_str(),
        change.drawable,
        change.before.x,
        change.before.y,
        change.before.width,
        change.before.height,
        change.after.x,
        change.after.y,
        change.after.width,
        change.after.height
    );
    if let Some(mask) = change.mask {
        println!("  mask {mask} scaled with it");
    }
    save(&doc, out)
}

fn replace_media(path: &str, target: &str, image: &str, out: &str) -> Result<(), Error> {
    let target = identifier(target)?;
    let bytes = std::fs::read(image)?;
    let name = std::path::Path::new(image)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("replacement");

    let mut doc = Document::open(path)?;
    let replacement = doc.replace_media(target, &bytes, name, None)?;
    println!(
        "media {}: {} -> {} ({} bytes, {:.0} × {:.0})",
        replacement.data,
        replacement.was,
        replacement.now,
        replacement.bytes,
        replacement.new_pixel_size.0,
        replacement.new_pixel_size.1
    );
    println!(
        "  digest {}",
        replacement
            .digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    if !replacement.drawables.is_empty() {
        let list: Vec<String> = replacement.drawables.iter().map(u64::to_string).collect();
        println!("  drawable(s) brought into step: {}", list.join(", "));
    }
    if let Some(note) = replacement.aspect_note() {
        println!("  warning: {note}");
    }
    if !replacement.pixel_size_recorded {
        println!(
            "  warning: the registry entry carries no image attributes, so the new pixel \
             size could not be recorded there"
        );
    }
    save(&doc, out)
}

/// Hidden rows or columns, each with the reason the model gives.
fn hidden_list(extents: &[iwork::table::Extent]) -> Vec<String> {
    extents
        .iter()
        .enumerate()
        .filter(|(_, e)| e.hidden())
        .map(|(i, e)| format!("{i} ({})", e.hiding().as_str()))
        .collect()
}

/// The one-line-per-feature view `iwork tables` shows; `iwork organise` prints
/// the same things in full.
fn organisation_summary(table: &iwork::Table, indent: &str) {
    if !table.sort_rules.is_empty() {
        let rules: Vec<String> = table
            .sort_rules
            .iter()
            .map(|rule| {
                format!(
                    "{} {}",
                    column_name(rule.column),
                    if rule.descending { "desc" } else { "asc" }
                )
            })
            .collect();
        println!("{indent}sorted by: {}", rules.join(", "));
    }
    if let Some(filter) = &table.filter {
        println!(
            "{indent}filter: {} rule(s), match {}, {}",
            filter.rules.len(),
            if filter.match_any { "any" } else { "all" },
            if filter.enabled { "on" } else { "off" }
        );
    }
    for category in &table.categories {
        let columns: Vec<String> = category
            .columns
            .iter()
            .map(|c| column_name(c.column.unwrap_or(usize::MAX)))
            .collect();
        println!(
            "{indent}category: by {}, {} group(s), {} summary assignment(s), {}",
            columns.join(" then "),
            category.groups().len(),
            category.summaries.len(),
            if category.enabled { "on" } else { "off" }
        );
    }
    if let Some(pivot) = &table.pivot {
        println!(
            "{indent}pivot of {:?}: {} row field(s), {} column field(s), {} value(s){}",
            pivot.source_name,
            pivot.rows.len(),
            pivot.columns.len(),
            pivot.values.len(),
            if pivot.empty { ", empty" } else { "" }
        );
    }
    let conditional: usize = table
        .conditional_styles
        .iter()
        .map(|set| set.rules.len())
        .sum();
    if conditional > 0 {
        println!(
            "{indent}conditional highlighting: {conditional} rule(s) in {} set(s)",
            table.conditional_styles.len()
        );
    }
}

/// A column index as the letter the app shows, or `?` when a UUID did not
/// resolve to one.
fn column_name(column: usize) -> String {
    if column == usize::MAX {
        return "?".to_string();
    }
    let mut name = String::new();
    let mut n = column + 1;
    while n > 0 {
        let digit = (n - 1) % 26;
        name.insert(0, (b'A' + digit as u8) as char);
        n = (n - 1) / 26;
    }
    name
}

/// Everything a table is organised by, in full.
fn organise(path: &str, wanted: Option<&str>) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let formats = doc.custom_formats();
    if !formats.is_empty() {
        println!("== custom cell formats ({}) ==", formats.len());
        for format in &formats {
            println!(
                "  {:<24} format {} {:?}{}",
                format.name,
                format.format_type,
                format.format_string,
                match format.conditions {
                    0 => String::new(),
                    n => format!(", {n} conditional sub-rule(s)"),
                }
            );
        }
        println!();
    }

    // Every table, always: a rule's condition is a formula, and a formula
    // resolves names against the whole document.
    let tables = doc.tables();
    let index = Some(iwork::table::names(&tables));
    let chosen: Vec<usize> = tables
        .iter()
        .enumerate()
        .filter(|(_, table)| match wanted {
            Some(name) => table.name == name || table.identifier.to_string() == name,
            None => true,
        })
        .map(|(at, _)| at)
        .collect();
    if let (Some(name), true) = (wanted, chosen.is_empty()) {
        return Err(Error::Format(format!("no table called '{name}' in {path}")));
    }
    if chosen.is_empty() {
        println!("no tables");
        return Ok(());
    }

    for position in chosen {
        let table = &tables[position];
        println!("== table {} {} ==", table.identifier, table.name);
        let hidden_rows = hidden_list(&table.row_extents);
        let hidden_columns = hidden_list(&table.column_extents);
        if !hidden_rows.is_empty() {
            println!("  hidden rows: {}", hidden_rows.join(", "));
        }
        if !hidden_columns.is_empty() {
            println!("  hidden columns: {}", hidden_columns.join(", "));
        }
        if !table.row_states.user_hidden.is_empty() || !table.row_states.filtered.is_empty() {
            println!(
                "  row hidden-state extent: {} user-hidden, {} filtered",
                table.row_states.user_hidden.len(),
                table.row_states.filtered.len()
            );
        }
        if !table.column_states.user_hidden.is_empty() || !table.column_states.filtered.is_empty() {
            println!(
                "  column hidden-state extent: user-hidden {:?}, filtered {:?}",
                table.column_states.user_hidden, table.column_states.filtered
            );
        }
        for rule in &table.sort_rules {
            println!(
                "  sort: column {} {}",
                column_name(rule.column),
                if rule.descending {
                    "descending"
                } else {
                    "ascending"
                }
            );
        }
        if let Some(filter) = &table.filter {
            println!(
                "  filter set {} — match {}, {}",
                filter.identifier,
                if filter.match_any { "any" } else { "all" },
                if filter.enabled {
                    "filters on"
                } else {
                    "filters off"
                }
            );
            for (at, rule) in filter.rules.iter().enumerate() {
                println!(
                    "    rule {at}: column {}, {}, {}",
                    rule.column.map(column_name).unwrap_or("?".into()),
                    if rule.enabled { "enabled" } else { "disabled" },
                    describe_in(&rule.predicate, index.as_ref(), position, rule.column)
                );
            }
        }
        for category in &table.categories {
            println!(
                "  category {} — {}",
                category.identifier,
                if category.enabled { "on" } else { "off" }
            );
            for column in &category.columns {
                println!(
                    "    grouped by column {} (grouping type {}{})",
                    column.column.map(column_name).unwrap_or("?".into()),
                    column.grouping_type,
                    if column.has_functor { ", bucketed" } else { "" }
                );
            }
            for summary in &category.summaries {
                println!(
                    "    summary: column {} at level {} = {}",
                    summary.column.map(column_name).unwrap_or("?".into()),
                    summary.level,
                    summary.function
                );
            }
            print_groups(category, 0);
        }
        if let Some(pivot) = &table.pivot {
            println!(
                "  pivot {} of table {:?}{}",
                pivot.identifier,
                pivot.source_name,
                if pivot.empty { " — empty" } else { "" }
            );
            for (label, fields) in [("rows", &pivot.rows), ("columns", &pivot.columns)] {
                for field in fields {
                    println!(
                        "    {label}: column {} (grouping type {}{})",
                        field.column.map(column_name).unwrap_or("?".into()),
                        field.grouping_type,
                        if field.has_functor { ", bucketed" } else { "" }
                    );
                }
            }
            for value in &pivot.values {
                println!(
                    "    value: column {} = {}{}",
                    value.column.map(column_name).unwrap_or("?".into()),
                    value.function,
                    match value.show_as {
                        0 => String::new(),
                        n => format!(" shown as {n}"),
                    }
                );
            }
            println!(
                "    grand totals: rows {}, columns {}",
                if pivot.hide_grand_total_rows {
                    "hidden"
                } else {
                    "shown"
                },
                if pivot.hide_grand_total_columns {
                    "hidden"
                } else {
                    "shown"
                }
            );
        }
        for set in &table.conditional_styles {
            println!(
                "  conditional highlighting set {} (key {})",
                set.identifier,
                set.key.map(|k| k.to_string()).unwrap_or("—".into())
            );
            for rule in &set.rules {
                println!(
                    "    {} -> cell style {:?}, text style {:?}",
                    describe_in(&rule.predicate, index.as_ref(), position, None),
                    rule.cell_style,
                    rule.text_style
                );
            }
        }
    }
    Ok(())
}

fn print_groups(category: &iwork::table::Category, _depth: usize) {
    let Some(root) = &category.root else { return };
    fn walk(group: &iwork::table::Group, depth: usize) {
        for child in &group.children {
            println!(
                "    {}group {:?}: {} row(s){}",
                "  ".repeat(depth),
                child
                    .value
                    .as_ref()
                    .map(|v| v.to_text())
                    .unwrap_or_default(),
                child.rows.len(),
                if child.collapsed { ", collapsed" } else { "" }
            );
            walk(child, depth + 1);
        }
    }
    walk(root, 0);
}

/// A predicate in one line: its code, whatever it compares against, and — when
/// the condition is a formula — the formula itself.
fn describe_in(
    predicate: &iwork::table::Predicate,
    index: Option<&iwork::formula::Names>,
    table: usize,
    column: Option<usize>,
) -> String {
    let mut text = format!("predicate {}", predicate.kind);
    if predicate.qualifiers != (0, 0) {
        text.push_str(&format!(" {:?}", predicate.qualifiers));
    }
    for value in &predicate.values {
        text.push_str(&format!(" {:?}", value.to_text()));
    }
    match index.and_then(|index| {
        iwork::table::predicate_text(predicate, index, table, column.unwrap_or(0))
    }) {
        Some(formula) => text.push_str(&format!(" {formula}")),
        None if predicate.has_formula => text.push_str(" (against a formula)"),
        None => {}
    }
    if predicate.pre_pivot {
        text.push_str(" [pre-pivot form]");
    }
    text
}

/// Every formula in the document: where it is, what it says, what it cached.
fn formulas(path: &str, wanted: Option<&str>) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let tables = doc.tables();
    let all = iwork::table::formulas(&tables);
    let listed: Vec<&iwork::table::FormulaCell> = all
        .iter()
        .filter(|f| match wanted {
            Some(name) => f.table_name == name || f.table.to_string() == name,
            None => true,
        })
        .collect();
    if listed.is_empty() {
        println!("no formulas");
        return Ok(());
    }
    let mut last = None;
    for formula in listed {
        if last != Some(formula.table) {
            match &formula.sheet {
                Some(sheet) => println!(
                    "== table {} {} (sheet {sheet}) ==",
                    formula.table, formula.table_name
                ),
                None => println!("== table {} {} ==", formula.table, formula.table_name),
            }
            last = Some(formula.table);
        }
        println!(
            "{:<6} {:<48} = {}",
            formula.reference,
            formula.text,
            formula.value.to_text()
        );
    }
    Ok(())
}

/// Every chart, its type, its placement, its private data as a small table,
/// and — in Numbers — the table references that feed it.
///
/// The two are printed together and labelled, because they are the phase's
/// whole point: the grid is what the chart *draws*, the references are what it
/// *follows*, and a Numbers chart whose table has changed since the last
/// recalculation has a grid that disagrees with them.
fn charts(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let charts = doc.charts();
    if charts.is_empty() {
        println!("no charts");
        return Ok(());
    }
    for chart in &charts {
        let mut what = chart.type_label();
        if chart.is_interactive() {
            what.push_str(" (interactive)");
        }
        println!(
            "== {} — {} — {} — {:.1}×{:.1} at {:.1},{:.1} ==",
            chart.identifier,
            what,
            chart.placement.as_str(),
            chart.frame.width,
            chart.frame.height,
            chart.frame.x,
            chart.frame.y
        );
        let mut notes = vec![
            format!("{} ({})", chart.type_name(), chart.chart_type),
            format!(
                "series {}",
                iwork::chart::series_direction_name(chart.series_direction)
            ),
        ];
        // `scatter_format` is written on every chart and only means anything on
        // the four that pair X against Y.
        if matches!(chart.chart_type, 9 | 22 | 23 | 24) && chart.scatter_format != 0 {
            notes.push(iwork::chart::scatter_format_name(chart.scatter_format).to_string());
        }
        if let Some(index) = chart.multidataset_index {
            notes.push(format!("data set {index}"));
        }
        if chart.contains_default_data {
            notes.push("placeholder data".to_string());
        }
        if chart.is_3d() {
            notes.push("3D".to_string());
        }
        println!("   {}", notes.join(", "));

        let series = chart.series();
        let categories = chart.categories();
        let width = series
            .iter()
            .filter_map(|s| s.name.as_ref().map(|n| n.chars().count()))
            .chain(std::iter::once(6))
            .max()
            .unwrap_or(6);
        let columns = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
        let mut header = format!("   {:<width$}", "", width = width);
        for column in 0..columns {
            header.push_str(&format!(
                " {:>12}",
                categories.get(column).map(String::as_str).unwrap_or("")
            ));
        }
        println!("{header}");
        for one in &series {
            let mut line = format!(
                "   {:<width$}",
                one.name.as_deref().unwrap_or(""),
                width = width
            );
            for value in &one.values {
                line.push_str(&format!(" {:>12}", value.to_text()));
            }
            println!("{line}");
        }

        match &chart.references {
            Some(references) => {
                let printed = |list: &[iwork::chart::ChartReference]| {
                    list.iter()
                        .map(|r| r.to_text())
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                println!(
                    "   fed by {}",
                    join_or(&[printed(&references.data)], "nothing")
                );
                if !references.row_labels.is_empty() {
                    println!("   row labels {}", printed(&references.row_labels));
                }
                if !references.column_labels.is_empty() {
                    println!("   column labels {}", printed(&references.column_labels));
                }
                println!(
                    "   mediator {}, {} reference(s) through function 175{}",
                    references.mediator,
                    references.wrapped_in_175,
                    if references.unwrapped > 0 {
                        format!(", {} not wrapped", references.unwrapped)
                    } else {
                        String::new()
                    }
                );
            }
            None => println!("   private data only — no mediator, nothing to follow"),
        }

        let overrides = chart.series_private_styles.entries.len();
        println!(
            "   styles: {} theme series, {} of {} with a private override, {} of {} with a non-style",
            chart.series_theme_styles.len(),
            overrides,
            chart.series_private_styles.count,
            chart.series_non_styles.entries.len(),
            chart.series_non_styles.count
        );
        if !chart.extensions.is_empty() {
            println!(
                "   extensions: {}",
                chart
                    .extensions
                    .iter()
                    .map(|(number, name)| format!("{name} ({number})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    Ok(())
}

fn join_or(parts: &[String], empty: &str) -> String {
    if parts.is_empty() {
        empty.to_string()
    } else {
        parts.join(", ")
    }
}

fn find_table(path: &str, wanted: &str) -> Result<iwork::Table, Error> {
    let doc = Document::open(path)?;
    doc.table(wanted)
        .ok_or_else(|| Error::Format(format!("no table called '{wanted}' in {path}")))
}

fn cells(path: &str, wanted: &str, raw: bool) -> Result<(), Error> {
    // The name index is document-wide: a formula in this table may name a
    // column of another one, and which table owns a shared name depends on all
    // of them.
    let doc = Document::open(path)?;
    let tables = doc.tables();
    let index = iwork::table::names(&tables);
    let position = tables
        .iter()
        .position(|t| t.name == wanted || t.identifier.to_string() == wanted)
        .ok_or_else(|| Error::Format(format!("no table called '{wanted}' in {path}")))?;
    let table = &tables[position];
    println!(
        "table {} — {} — {}×{}",
        table.identifier, table.name, table.rows, table.columns
    );
    for cell in table.cells() {
        if cell.value.is_empty() && !raw {
            continue;
        }
        let mut notes = Vec::new();
        notes.push(cell.format.to_string());
        if let Some(control) = cell.control {
            notes.push(format!("control: {}", control.as_str()));
        }
        if let Some(formula) = table.formula(cell.row, cell.column) {
            let at = iwork::formula::Site::new(
                &index,
                Some(position),
                (cell.column as i64, cell.row as i64),
            );
            notes.push(formula.text(at));
        } else if cell.has_formula {
            notes.push("formula".to_string());
        }
        if let Some(merge) = table.merge_at(cell.row, cell.column) {
            notes.push(format!("merged {}×{}", merge.rows, merge.columns));
        }
        let suffix = if notes.is_empty() {
            String::new()
        } else {
            format!("  [{}]", notes.join(", "))
        };
        println!(
            "{:<6} {:<10} {}{suffix}",
            reference_name(cell.row, cell.column),
            cell.value.kind(),
            cell.value.to_text()
        );
        if raw {
            let r = &cell.record;
            println!(
                "       type {:<3} reserved {:02x}{:02x}{:02x}{:02x} extras {:04x} flags {:08x} \
                 keys{}{}{}{}{}{}{}{}{}{}{}{}",
                r.cell_type,
                r.reserved[0],
                r.reserved[1],
                r.reserved[2],
                r.reserved[3],
                r.extras,
                r.flags,
                key(" string", r.string_id),
                key(" rich", r.rich_id),
                key(" cell-style", r.cell_style_id),
                key(" text-style", r.text_style_id),
                // The two an edit must carry rather than re-synthesise: a cell
                // that loses them loses its highlighting and nothing says so.
                key(" conditional-style", r.conditional_style_id),
                key(" conditional-rule", r.conditional_rule_id),
                key(" formula", r.formula_id),
                key(" control", r.control_id),
                key(" comment", r.comment_id),
                key(" format-kind", r.format_kind),
                key(" number-format", r.number_format_id),
                key(
                    " other-format",
                    r.format_id().filter(|_| r.number_format_id.is_none())
                ),
            );
        }
    }
    Ok(())
}

fn set_cell(
    path: &str,
    table: &str,
    row: usize,
    column: usize,
    value: &str,
    out: &str,
) -> Result<(), Error> {
    let value = parse_cell_value(value)?;
    let mut doc = Document::open(path)?;
    let previous = doc.set_cell(table, row, column, value.clone())?;
    println!(
        "{} of table {table}: {} -> {}",
        reference_name(row, column),
        describe_value(&previous),
        describe_value(&value)
    );
    save(&doc, out)
}

/// Write a CSV file into a table at a corner — the batch write, from a shell.
///
/// The inverse of `iwork csv`, and the reason [`Document::set_block`] exists:
/// a table of data arrives as rows, not as one cell at a time. Values take the
/// same `n:`/`b:`/`d:` prefixes `set-cell` does, so a column of numbers is
/// `n:1,n:2,n:3`; an empty field leaves the cell alone rather than clearing it,
/// because a CSV row shorter than the table is not a row of blanks.
fn set_cells(
    path: &str,
    table: &str,
    at: (usize, usize),
    csv: &str,
    out: &str,
) -> Result<(), Error> {
    let text = std::fs::read_to_string(csv)?;
    let mut rows: Vec<Vec<iwork::table::CellValue>> = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            rows.push(Vec::new());
            continue;
        }
        rows.push(
            line.split(',')
                .map(|field| parse_cell_value(field.trim()))
                .collect::<Result<Vec<_>, Error>>()?,
        );
    }
    let mut doc = Document::open(path)?;
    let written = doc.set_block(table, at, &rows)?;
    println!(
        "table {table}: {written} cell(s) written from {csv} at {}; rewrote {}",
        reference_name(at.0, at.1),
        doc.changed_streams().join(", ")
    );
    save(&doc, out)
}

/// Give a cell a formula, written as text, with the value it shows.
fn set_formula(
    path: &str,
    table: &str,
    row: usize,
    column: usize,
    formula: &str,
    value: &str,
    out: &str,
) -> Result<(), Error> {
    let value = parse_cell_value(value)?;
    let mut doc = Document::open(path)?;
    doc.set_formula(table, row, column, formula, value.clone())?;
    println!(
        "{} of table {table}: {formula} showing {}",
        reference_name(row, column),
        describe_value(&value)
    );
    save(&doc, out)
}

/// Give a cell a data format: `percent:1`, `currency:CHF:2`, `date:dd.MM.y`.
fn set_format(
    path: &str,
    table: &str,
    row: usize,
    column: usize,
    format: &str,
    out: &str,
) -> Result<(), Error> {
    let format = parse_format(format)?;
    let mut doc = Document::open(path)?;
    doc.set_format(table, [(row, column)], &format)?;
    println!(
        "{} of table {table}: {}",
        reference_name(row, column),
        format.name()
    );
    save(&doc, out)
}

/// `automatic`, `number[:places]`, `percent[:places]`, `scientific[:places]`,
/// `currency:CODE[:places]`, `date:PATTERN`.
fn parse_format(text: &str) -> Result<iwork::table::Format, Error> {
    use iwork::table::Format;
    let mut parts = text.split(':');
    let kind = parts.next().unwrap_or("");
    let rest: Vec<&str> = parts.collect();
    let places = |at: usize| -> Result<Option<u8>, Error> {
        match rest.get(at) {
            None => Ok(None),
            Some(text) => text
                .parse()
                .map(Some)
                .map_err(|_| Error::Format(format!("'{text}' is not a number of decimal places"))),
        }
    };
    Ok(match kind {
        "automatic" => Format::Automatic,
        "number" => Format::Number {
            decimals: places(0)?,
        },
        "percent" => Format::Percent {
            decimals: places(0)?,
        },
        "scientific" => Format::Scientific {
            decimals: places(0)?,
        },
        "currency" => Format::Currency {
            code: rest
                .first()
                .ok_or_else(|| Error::Format("currency needs a code, as currency:CHF".into()))?
                .to_string(),
            decimals: places(1)?,
        },
        "date" => Format::DateTime {
            // A date pattern has colons in it — `HH:mm` — so the rest of the
            // argument is the pattern, not the next field.
            pattern: rest.join(":"),
        },
        other => {
            return Err(Error::Format(format!(
                "'{other}' is not a format — automatic, number, percent, scientific, currency \
                 or date"
            )))
        }
    })
}

/// A column's width or a row's height, in points; `default` gives it back.
fn set_size(
    path: &str,
    table: &str,
    index: usize,
    points: &str,
    column: bool,
    out: &str,
) -> Result<(), Error> {
    let size = match points {
        "default" => None,
        text => Some(
            text.parse::<f32>()
                .map_err(|_| Error::Format(format!("'{text}' is not a size in points")))?,
        ),
    };
    let mut doc = Document::open(path)?;
    if column {
        doc.set_column_width(table, index, size)?;
    } else {
        doc.set_row_height(table, index, size)?;
    }
    println!(
        "table {table}: {} {index} is now {}",
        if column { "column" } else { "row" },
        match size {
            Some(points) => format!("{points} pt"),
            None => "the table default".to_string(),
        }
    );
    save(&doc, out)
}

/// Delete a row or a column, with everything in it.
fn delete_line(path: &str, table: &str, at: usize, row: bool, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let (rows, columns) = find_table_size(&doc, table)?;
    if row {
        doc.delete_row(table, at)?;
    } else {
        doc.delete_column(table, at)?;
    }
    println!(
        "table {table}: deleted {} {at} ({}×{} -> {}×{}); rewrote {}",
        if row { "row" } else { "column" },
        rows,
        columns,
        if row { rows - 1 } else { rows },
        if row { columns } else { columns - 1 },
        doc.changed_streams().join(", ")
    );
    save(&doc, out)
}

/// Merge a rectangle of cells into one.
fn merge(
    path: &str,
    table: &str,
    row: usize,
    column: usize,
    rows: usize,
    columns: usize,
    out: &str,
) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    doc.merge_cells(table, row, column, rows, columns)?;
    println!(
        "table {table}: {} spans {rows} row(s) × {columns} column(s)",
        reference_name(row, column)
    );
    save(&doc, out)
}

fn unmerge(path: &str, table: &str, row: usize, column: usize, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    doc.unmerge_cells(table, row, column)?;
    println!(
        "table {table}: the merge at {} is gone; its cells stay empty",
        reference_name(row, column)
    );
    save(&doc, out)
}

fn insert_row(path: &str, table: &str, at: usize, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let before = find_table_rows(&doc, table)?;
    doc.insert_row(table, at)?;
    println!(
        "table {table}: inserted an empty row at index {at} ({} -> {} rows); rewrote {}",
        before,
        before + 1,
        doc.changed_streams().join(", ")
    );
    save(&doc, out)
}

/// Row count of a table named for a write, so the CLI can report the delta.
fn insert_column(path: &str, table: &str, at: usize, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let before = find_table_size(&doc, table)?.1;
    doc.insert_column(table, at)?;
    println!(
        "table {table}: inserted an empty column at index {at} ({} -> {} columns); rewrote {}",
        before,
        before + 1,
        doc.changed_streams().join(", ")
    );
    save(&doc, out)
}

/// The table's rows and columns, for a message that says what changed.
fn find_table_size(doc: &Document, wanted: &str) -> Result<(usize, usize), Error> {
    let by_id: Option<u64> = wanted.parse().ok();
    doc.tables()
        .into_iter()
        .find(|t| Some(t.identifier) == by_id || t.name == wanted)
        .map(|t| (t.rows, t.columns))
        .ok_or_else(|| Error::Format(format!("no table called '{wanted}'")))
}

fn find_table_rows(doc: &Document, wanted: &str) -> Result<usize, Error> {
    let by_id: Option<u64> = wanted.parse().ok();
    doc.tables()
        .into_iter()
        .find(|t| Some(t.identifier) == by_id || t.name == wanted)
        .map(|t| t.rows)
        .ok_or_else(|| Error::Format(format!("no table called '{wanted}'")))
}

/// A cell value as the command line spells it — text unless it says otherwise.
fn parse_cell_value(text: &str) -> Result<iwork::table::CellValue, Error> {
    use iwork::table::CellValue;
    let bad = |what: &str, value: &str| Error::Format(format!("'{value}' is not {what}"));
    let Some((kind, rest)) = text.split_once(':') else {
        return Ok(CellValue::Text(text.to_string()));
    };
    Ok(match kind {
        "text" => CellValue::Text(rest.to_string()),
        "n" => CellValue::Number(
            iwork::table::Decimal::parse(rest).ok_or_else(|| bad("a number", rest))?,
        ),
        "b" => match rest {
            "true" | "yes" | "1" => CellValue::Bool(true),
            "false" | "no" | "0" => CellValue::Bool(false),
            other => return Err(bad("true or false", other)),
        },
        "d" => CellValue::Date(
            iwork::table::parse_date(rest)
                .ok_or_else(|| bad("a date, as 2024-03-01 or 2024-03-01T10:30:00Z", rest))?,
        ),
        "dur" => CellValue::Duration(rest.parse().map_err(|_| bad("a number of seconds", rest))?),
        "empty" => CellValue::Empty,
        // A colon is ordinary in a cell — "Total: 40" is text, not a prefix
        // this crate failed to understand.
        _ => CellValue::Text(text.to_string()),
    })
}

fn describe_value(value: &iwork::table::CellValue) -> String {
    match value {
        iwork::table::CellValue::Empty => "(empty)".to_string(),
        other => format!("{} {:?}", other.kind(), other.to_text()),
    }
}

fn csv(path: &str, wanted: &str) -> Result<(), Error> {
    let table = find_table(path, wanted)?;
    for row in table.to_rows() {
        let fields: Vec<String> = row.iter().map(|field| csv_field(field)).collect();
        println!("{}", fields.join(","));
    }
    Ok(())
}

/// A small CSV as a chart's data: the first row names the categories, the
/// first cell of every row after it names the series.
///
/// Deliberately strict. A cell that is neither blank nor a number is refused by
/// name rather than plotted as zero, because a chart drawing a zero nobody
/// wrote is exactly the kind of plausible wrong answer this crate is built to
/// avoid.
fn chart_data(path: &str) -> Result<iwork::chart::ChartData, Error> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Format(format!("{path}: {e}")))?;
    let mut lines = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(split_csv);
    let heading = lines
        .next()
        .ok_or_else(|| Error::Format(format!("{path} is empty")))?;
    let mut data = iwork::chart::ChartData {
        // The first cell of the heading sits above the series names and names
        // nothing, so it is dropped.
        column_names: heading.into_iter().skip(1).collect(),
        ..Default::default()
    };
    for (number, line) in lines.enumerate() {
        let mut cells = line.into_iter();
        data.row_names.push(cells.next().unwrap_or_default());
        let mut row = Vec::new();
        for cell in cells {
            let cell = cell.trim().to_string();
            if cell.is_empty() {
                row.push(iwork::chart::GridValue::Empty);
                continue;
            }
            let value = cell.parse::<f64>().map_err(|_| {
                Error::Format(format!(
                    "{path} row {}: {cell:?} is not a number, and a chart plots numbers",
                    number + 2
                ))
            })?;
            row.push(iwork::chart::GridValue::Number(value));
        }
        data.rows.push(row);
    }
    Ok(data)
}

/// One CSV line: commas separate, double quotes group, `""` is a quote.
fn split_csv(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut field)),
            _ => field.push(c),
        }
    }
    out.push(field);
    out
}

/// `iwork set-chart-data` — new numbers in a chart that is already there.
fn set_chart_data(path: &str, chart: u64, csv: &str, out: &str) -> Result<(), Error> {
    let data = chart_data(csv)?;
    let mut doc = Document::open(path)?;
    doc.set_chart_data(chart, &data)?;
    doc.save(out)?;
    println!(
        "chart {chart} now draws {} series over {} categories",
        data.rows.len(),
        data.column_names.len()
    );
    report_streams(&doc, out);
    Ok(())
}

/// `iwork add-chart` — a copy of a chart the document has, somewhere else.
#[allow(clippy::too_many_arguments)]
fn add_chart(
    path: &str,
    where_: &str,
    from: u64,
    csv: &str,
    x: &str,
    y: &str,
    w: &str,
    h: &str,
    out: &str,
) -> Result<(), Error> {
    let number = |what: &str, raw: &str| {
        raw.parse::<f32>()
            .map_err(|_| Error::Format(format!("{raw}: {what} is a number of points")))
    };
    let position = (number("a position", x)?, number("a position", y)?);
    let size = (number("a size", w)?, number("a size", h)?);
    let data = chart_data(csv)?;
    let mut doc = Document::open(path)?;
    let chart = doc.add_chart(where_, from, &data, position, size)?;
    doc.save(out)?;
    println!(
        "added chart {chart}, copied from {from}, at {},{} — {} × {}",
        position.0, position.1, size.0, size.1
    );
    report_streams(&doc, out);
    Ok(())
}

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

fn key(label: &str, value: Option<u32>) -> String {
    value.map(|v| format!("{label}={v}")).unwrap_or_default()
}

/// `A1`-style name for a cell, as the app writes it.
fn reference_name(row: usize, column: usize) -> String {
    let mut letters = String::new();
    let mut n = column + 1;
    while n > 0 {
        let digit = (n - 1) % 26;
        letters.insert(0, (b'A' + digit as u8) as char);
        n = (n - 1) / 26;
    }
    format!("{letters}{}", row + 1)
}

fn properties() -> Result<(), Error> {
    println!("  {:<24} {:<12} evidence", "name", "path");
    for (name, path, confidence) in style::property::BY_NAME {
        let dotted = path
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(".");
        println!(
            "  {name:<24} {dotted:<12} {}",
            match confidence {
                registry::Confidence::Confirmed => "measured in an imported document",
                registry::Confidence::Inferred => "observed changing alongside a measured one",
                registry::Confidence::Unverified => "name only, not observed here",
            }
        );
    }
    Ok(())
}

fn check(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;

    // Not a problem — a state. An object can carry older encodings of itself as
    // patches, and this crate reads the first message and rewrites none of
    // them; saying so is the difference between "nothing here" and "nothing
    // here that this tool would touch".
    let patched = doc.patched_objects();
    if !patched.is_empty() {
        let listed: Vec<String> = patched
            .iter()
            .map(|(id, patches)| format!("{id} ({patches})"))
            .collect();
        println!(
            "  note: {} object(s) carry version patches, read-only here: {}",
            patched.len(),
            listed.join(", ")
        );
    }

    let problems = doc.problems();
    if problems.is_empty() {
        println!(
            "{path}: no problems found in {} objects",
            doc.objects().count()
        );
        return Ok(());
    }
    for problem in &problems {
        println!("  {problem}");
    }
    Err(Error::Format(format!("{} problem(s)", problems.len())))
}

fn extract(path: &str, dir: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    std::fs::create_dir_all(dir)?;
    let mut written = 0;
    for name in doc.package().data_names() {
        let bytes = doc
            .package()
            .get(&name)
            .expect("name came from the package");
        // The name comes out of a ZIP a stranger wrote, and this is where it
        // becomes a path on this machine: `Data/../../../evil` would otherwise
        // be written wherever it says. Checked, and refused by name.
        let file = iwork::package::entry_path(name.trim_start_matches("Data/"))?;
        let target = std::path::Path::new(dir).join(file);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Never follow a symlink already at the target: `fs::write` would write
        // through it and overwrite a file outside `dir`. See `write_regular_file`.
        iwork::package::write_regular_file(&target, bytes)?;
        println!("  {} ({} bytes)", target.display(), bytes.len());
        written += 1;
    }
    if written == 0 {
        println!("no embedded media");
    }
    Ok(())
}

// -- text styles -------------------------------------------------------------

fn styles(path: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let styles = doc.text_styles();
    if styles.is_empty() {
        println!("no text styles in {path}");
        return Ok(());
    }
    for style in &styles {
        println!(
            "  id={:<8} {:<10} {:<34} {:>4} run(s)  {}",
            style.identifier,
            style.kind.as_str(),
            style.label().unwrap_or("(unnamed variation)"),
            doc.text_style_usage(style.identifier).len(),
            style.stream,
        );
    }
    println!("\n{} text styles", styles.len());
    Ok(())
}

fn show_style(path: &str, id: u64) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let style = doc.text_style(id).ok_or(Error::NoSuchStyle(id))?;
    let message_type = style.kind.message_type();
    println!(
        "{} style {id} — type {message_type} {}, in {}",
        style.kind.as_str(),
        registry::describe_in(doc.kind(), message_type),
        style.stream
    );
    match &style.name {
        Some(name) => println!("name: {name:?}"),
        None => println!("name: none — this is a variation style"),
    }
    if let Some(identifier) = &style.style_identifier {
        println!("style identifier: {identifier:?}");
    }
    if let Some(parent) = style.parent {
        let inherited = doc
            .text_style(parent)
            .and_then(|s| s.label().map(str::to_string));
        println!(
            "inherits from: {parent} {}",
            inherited.unwrap_or_else(|| "(unnamed)".into())
        );
    }

    println!("\n== fields ==");
    dump(&style.archive, "");

    let uses = doc.text_style_usage(id);
    println!("\n== used by {} run(s) ==", uses.len());
    for used in &uses {
        println!(
            "  storage {:<8} field {:<3} chars {}..{}  {}",
            used.storage, used.table, used.range.start, used.range.end, used.stream
        );
    }
    Ok(())
}

/// Print any object's fields, whatever it is.
///
/// The registry names a fraction of the message types and this crate models
/// fewer still, so the way to find out what an unknown archive holds is to look
/// at it. Reference-shaped fields are marked, which is usually enough to walk
/// the graph by hand.
fn dump_object(path: &str, id: u64) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let (stream, object) = doc.object(id).ok_or(Error::NoSuchObject(id))?;
    println!(
        "object {id} — type {} {}, in {stream}",
        object.message_type(),
        registry::describe_in(doc.kind(), object.message_type())
    );
    for (index, message) in object.messages.iter().enumerate() {
        if object.messages.len() > 1 {
            println!("\n-- message {index}, type {} --", message.message_type);
        }
        println!(
            "version {:?}, {} bytes\n",
            message.version,
            message.payload.len()
        );
        match Message::decode(&message.payload) {
            Ok(archive) => dump(&archive, ""),
            Err(e) => println!("  could not decode: {e}"),
        }
    }
    Ok(())
}

/// Print an archive as addressable field paths, so `set-style` has something to
/// aim at. Nothing here claims to know what a field means.
fn dump(message: &Message, prefix: &str) {
    for field in &message.fields {
        let path = match prefix.is_empty() {
            true => field.number.to_string(),
            false => format!("{prefix}.{}", field.number),
        };
        match &field.value {
            Value::Varint(v) => println!("  {path:<14} varint  {v}"),
            Value::Fixed32(b) => println!("  {path:<14} f32     {}", f32::from_le_bytes(*b)),
            Value::Fixed64(b) => println!("  {path:<14} f64     {}", f64::from_le_bytes(*b)),
            Value::Bytes(raw) => match pb::decode_nested(raw) {
                Some(nested) => {
                    match iwork::style::reference_target(&nested) {
                        Some(target) => println!("  {path:<14} ref     -> object {target}"),
                        None => println!("  {path:<14} message {} bytes", raw.len()),
                    }
                    dump(&nested, &path);
                }
                None => match std::str::from_utf8(raw) {
                    Ok(text) if !text.is_empty() && text.chars().all(|c| !c.is_control()) => {
                        println!("  {path:<14} str     {text:?}")
                    }
                    _ => println!("  {path:<14} bytes   {} bytes", raw.len()),
                },
            },
        }
    }
}

fn new_style(path: &str, template: u64, name: &str, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let created = doc.create_text_style(template, name)?;
    println!(
        "created style {} from {} in {} ({} stylesheet entries cloned)",
        created.identifier, created.template, created.stream, created.registrations_cloned
    );
    let result = save(&doc, out);
    if created.name.is_none() {
        println!(
            "note: style {} is a variation, so the copy is anonymous and keeps no name.\n      \
             Naming it would make Pages refuse the document — see create_text_style.",
            created.template
        );
    }
    result
}

/// A style keeps its text colour in up to four places and they must agree, so
/// setting one of them by path is usually the wrong thing to do.
fn set_color(path: &str, id: &str, r: &str, g: &str, b: &str, out: &str) -> Result<(), Error> {
    let id = identifier(id)?;
    let channel = |text: &str| -> Result<f32, Error> {
        text.parse::<f32>()
            .map_err(|_| Error::Format(format!("'{text}' is not a channel value in 0.0..=1.0")))
    };
    let mut doc = Document::open(path)?;
    let set = doc.set_text_style_color(id, channel(r)?, channel(g)?, channel(b)?, 1.0)?;
    if set == 0 {
        return Err(Error::Format(format!(
            "style {id} keeps no colour of its own, and one invented here would \
             make Pages refuse the document — copy a style that has one"
        )));
    }
    println!("set {set} colour field(s) on style {id}");
    save(&doc, out)
}

fn set_style(path: &str, id: u64, assignment: &str, out: &str) -> Result<(), Error> {
    let (field, value) = assignment
        .split_once('=')
        .ok_or_else(|| Error::Format(format!("'{assignment}' is not <path>=<value>")))?;

    let mut doc = Document::open(path)?;
    if field == "name" {
        doc.rename_text_style(id, value)?;
        println!("renamed style {id} to {value:?}");
    } else {
        let field_path = parse_path(field)?;
        let value = parse_value(value)?;
        let cleared = value.is_none();
        doc.set_text_style_property(id, &field_path, value)?;
        match cleared {
            true => println!("cleared field {field} of style {id}"),
            false => println!("set field {field} of style {id}"),
        }
    }
    save(&doc, out)
}

fn delete_style(path: &str, id: u64, replacement: Option<u64>, out: &str) -> Result<(), Error> {
    let mut doc = Document::open(path)?;
    let deleted = doc.delete_text_style(id, replacement)?;
    println!(
        "deleted style {id}: {} run(s) repointed, {} dropped, {} stylesheet entries removed",
        deleted.runs_repointed, deleted.runs_dropped, deleted.registrations_removed
    );
    save(&doc, out)
}

fn apply_style(
    path: &str,
    storage: &str,
    start: &str,
    end: &str,
    style: &str,
    out: &str,
) -> Result<(), Error> {
    let storage = identifier(storage)?;
    let start = identifier(start)?;
    let end = identifier(end)?;
    let style = identifier(style)?;

    let mut doc = Document::open(path)?;
    doc.apply_text_style(storage, start..end, style)?;
    println!("storage {storage} chars {start}..{end} now use style {style}");
    save(&doc, out)
}

fn paragraphs(path: &str, storage: u64) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let text: Vec<u16> = doc.storage_text(storage)?.encode_utf16().collect();
    let list = doc.list_paragraphs(storage)?;
    for (index, paragraph) in list.iter().enumerate() {
        let slice = &text[paragraph.range.start as usize..paragraph.range.end as usize];
        let style = doc
            .style_of_run(storage, paragraph.range.start, style::StyleKind::Paragraph)?
            .map(|resolved| {
                let name = resolved
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("style {}", resolved.style));
                match (resolved.is_variation, resolved.overrides.len()) {
                    (true, 0) => format!("{name} (variation)"),
                    (true, n) => format!("{name} + {n} override(s)"),
                    (false, _) => name,
                }
            })
            .unwrap_or_else(|| "-".to_string());
        let bullet = match doc.text_style(paragraph.style.unwrap_or(0)) {
            Some(list_style) => list_style
                .name
                .unwrap_or_else(|| format!("list {}", list_style.identifier)),
            None => "-".to_string(),
        };
        println!(
            "  {index:<4} {:>7}..{:<7} L{}  {:<28} {:<14} {}",
            paragraph.range.start,
            paragraph.range.end,
            paragraph.level,
            style,
            bullet,
            snippet(slice)
        );
    }
    println!("\n{} paragraph(s), {} characters", list.len(), text.len());
    Ok(())
}

/// One line of a paragraph's text, short enough to scan in a listing.
fn snippet(units: &[u16]) -> String {
    let text = String::from_utf16_lossy(units);
    let flat: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let trimmed = flat.trim();
    match trimmed.char_indices().nth(64) {
        Some((cut, _)) => format!("{}…", &trimmed[..cut]),
        None => trimmed.to_string(),
    }
}

// -- argument parsing --------------------------------------------------------

fn identifier(text: &str) -> Result<u64, Error> {
    text.parse()
        .map_err(|_| Error::Format(format!("'{text}' is not an object identifier")))
}

fn index(text: &str) -> Result<usize, Error> {
    text.parse()
        .map_err(|_| Error::Format(format!("'{text}' is not a row or column index")))
}

/// A cell named as the app names it, as indices — the library's own reader, so
/// the CLI and a caller of the crate accept exactly the same thing.
fn reference_position(text: &str) -> Result<(usize, usize), Error> {
    iwork::table::CellRef::from(text).resolve()
}

/// A dotted path of field numbers, or one of the names in
/// [`iwork::style::property`].
fn parse_path(text: &str) -> Result<Vec<u32>, Error> {
    if text.is_empty() {
        return Err(Error::Format("empty field path".into()));
    }
    if let Some(known) = style::property::path(text) {
        return Ok(known.to_vec());
    }
    text.split('.')
        .map(|part| {
            part.parse().map_err(|_| {
                let names: Vec<&str> = style::property::BY_NAME
                    .iter()
                    .map(|(n, _, _)| *n)
                    .collect();
                Error::Format(format!(
                    "'{part}' is not a field number; known names are {}",
                    names.join(", ")
                ))
            })
        })
        .collect()
}

/// `varint:1`, `f32:18`, `f64:1.5`, `str:Body`, `hex:0a0b`, or empty to remove
/// the field.
fn parse_value(text: &str) -> Result<Option<Value>, Error> {
    if text.is_empty() {
        return Ok(None);
    }
    let (kind, rest) = text
        .split_once(':')
        .ok_or_else(|| Error::Format(format!("'{text}' is not <type>:<value>")))?;
    let bad = |what: &str| Error::Format(format!("'{rest}' is not {what}"));
    let value = match kind {
        "varint" => Value::Varint(rest.parse().map_err(|_| bad("an unsigned integer"))?),
        "f32" => Value::Fixed32(
            rest.parse::<f32>()
                .map_err(|_| bad("a number"))?
                .to_le_bytes(),
        ),
        "f64" => Value::Fixed64(
            rest.parse::<f64>()
                .map_err(|_| bad("a number"))?
                .to_le_bytes(),
        ),
        "str" => Value::Bytes(rest.as_bytes().to_vec()),
        "hex" => Value::Bytes(parse_hex(rest)?),
        other => {
            return Err(Error::Format(format!(
                "unknown value type '{other}' (varint, f32, f64, str, hex)"
            )))
        }
    };
    Ok(Some(value))
}

fn parse_hex(text: &str) -> Result<Vec<u8>, Error> {
    let digits: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if digits.len() % 2 != 0 {
        return Err(Error::Format("hex needs an even number of digits".into()));
    }
    digits
        .chunks(2)
        .map(|pair| {
            u8::from_str_radix(&pair.iter().collect::<String>(), 16).map_err(|_| {
                Error::Format(format!("'{}' is not hex", pair.iter().collect::<String>()))
            })
        })
        .collect()
}

fn roundtrip(path: &str, out: &str) -> Result<(), Error> {
    let doc = Document::open(path)?;
    let count = doc.objects().count();
    save(&doc, out)?;

    // Re-open and compare object streams, so the check covers the file that was
    // actually written rather than the in-memory graph.
    let reopened = Document::open(out)?;
    let before: Vec<_> = doc.objects().map(|(_, o)| o.identifier).collect();
    let after: Vec<_> = reopened.objects().map(|(_, o)| o.identifier).collect();
    if before != after {
        return Err(Error::Format(
            "object identifiers changed on re-encode".into(),
        ));
    }
    println!("decoded and checked {count} objects");
    Ok(())
}
