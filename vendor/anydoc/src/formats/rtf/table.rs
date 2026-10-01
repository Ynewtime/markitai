//! RTF table assembly: cell properties captured at `\cellx` time, per-depth
//! row accumulation (nested tables live at `\itap` depth); the canonical
//! grid construction is the shared edge-based assembly (`shared::grid`).

use crate::error::ConvertError;
use crate::model::{Block, Inline, inlines_are_empty};
use crate::shared::blockstyle::{BlockStyle, StyledRun};
use crate::shared::grid::{GridRow, build_edge_table};
use crate::shared::list::{ListEntry, flush_list};

pub use crate::shared::grid::CellProp;

#[derive(Default)]
struct RowBuild {
    cells: Vec<(Vec<Block>, CellProp)>,
    header: bool,
}

#[derive(Default)]
struct TableBuild {
    rows: Vec<RowBuild>,
    /// Props declared by the current `\trowd ... \cellx` run.
    row_props: Vec<CellProp>,
    prop_cursor: usize,
    pending_prop: CellProp,
    row: RowBuild,
    row_header: bool,
}

/// Table builders indexed by nesting depth - 1, plus the per-depth pending
/// cell content.
pub struct TableState {
    tables: Vec<TableBuild>,
    cell_blocks: Vec<Vec<Block>>,
    cell_runs: Vec<StyledRun>,
    /// markitai: list paragraphs awaiting their list, per depth.
    cell_lists: Vec<Vec<ListEntry>>,
}

impl TableState {
    pub fn new() -> Self {
        TableState {
            tables: Vec::new(),
            cell_blocks: vec![Vec::new()],
            cell_runs: vec![StyledRun::default()],
            cell_lists: vec![Vec::new()],
        }
    }

    /// Deepest depth with a builder allocated.
    pub fn depth(&self) -> usize {
        self.tables.len()
    }

    fn table_at(&mut self, depth: usize) -> &mut TableBuild {
        while self.tables.len() < depth {
            self.tables.push(TableBuild::default());
        }
        self.ensure_cell_depth(depth);
        &mut self.tables[depth - 1]
    }

    fn ensure_cell_depth(&mut self, depth: usize) {
        while self.cell_blocks.len() < depth {
            self.cell_blocks.push(Vec::new());
        }
        while self.cell_runs.len() < depth {
            self.cell_runs.push(StyledRun::default());
        }
        while self.cell_lists.len() < depth {
            self.cell_lists.push(Vec::new());
        }
    }

    fn cell_block_at(&mut self, depth: usize) -> &mut Vec<Block> {
        self.ensure_cell_depth(depth);
        &mut self.cell_blocks[depth - 1]
    }

    fn flush_cell_run(&mut self, depth: usize) {
        self.ensure_cell_depth(depth);
        let blocks = &mut self.cell_blocks[depth - 1];
        self.cell_runs[depth - 1].flush(blocks);
        flush_list(blocks, &mut self.cell_lists[depth - 1]);
    }

    /// `\trowd`: reset the row's declared properties.
    pub fn begin_row(&mut self, depth: usize) {
        let t = self.table_at(depth);
        t.row_props.clear();
        t.prop_cursor = 0;
        t.pending_prop = CellProp::default();
        t.row_header = false;
    }

    /// `\trhdr`: the row repeats as a header.
    pub fn mark_header_row(&mut self, depth: usize) {
        self.table_at(depth).row_header = true;
    }

    /// The property slot the next `\cellx` will seal.
    pub fn pending_prop(&mut self, depth: usize) -> &mut CellProp {
        &mut self.table_at(depth).pending_prop
    }

    /// `\cellxN`: seal the pending properties with the right boundary.
    pub fn declare_cell(&mut self, depth: usize, right: i64) {
        let t = self.table_at(depth);
        let mut prop = std::mem::take(&mut t.pending_prop);
        prop.right = right;
        t.row_props.push(prop);
    }

    /// A paragraph ended inside a cell at `depth`.
    pub fn push_cell_paragraph(
        &mut self,
        depth: usize,
        style: Option<BlockStyle>,
        inlines: Vec<Inline>,
    ) -> Result<(), ConvertError> {
        // markitai: a table nested in this cell has ended before the
        // paragraph; folding it in only when the cell closed put the
        // cell's later text ahead of it.
        self.flush_into_cell(depth + 1, depth)?;
        self.ensure_cell_depth(depth);
        let blocks = &mut self.cell_blocks[depth - 1];
        flush_list(blocks, &mut self.cell_lists[depth - 1]);
        let run = &mut self.cell_runs[depth - 1];
        if let Some(style) = style {
            run.push(style, inlines, blocks);
        } else {
            run.flush(blocks);
            if !inlines_are_empty(&inlines) {
                blocks.push(Block::Paragraph(inlines));
            }
        }
        Ok(())
    }

    /// markitai: a blank paragraph set in a monospaced font ended inside a
    /// cell at `depth`: a blank line of the code the cell lays out (see
    /// [`crate::shared::code::listing_tables`]), kept as an empty paragraph,
    /// which a table cell renders as nothing. Dropped like any other blank
    /// paragraph, it closed up the listing (`#include <iostream>` and
    /// `int main()` with no line between them).
    pub fn push_cell_blank_line(&mut self, depth: usize) -> Result<(), ConvertError> {
        self.flush_into_cell(depth + 1, depth)?;
        self.ensure_cell_depth(depth);
        let blocks = &mut self.cell_blocks[depth - 1];
        flush_list(blocks, &mut self.cell_lists[depth - 1]);
        self.cell_runs[depth - 1].flush(blocks);
        blocks.push(Block::Paragraph(Vec::new()));
        Ok(())
    }

    /// markitai: a list paragraph ended inside a cell at `depth`. Cells hold
    /// lists as the body does (a task list in a table cell); before, the
    /// items were plain paragraphs with no marker.
    pub fn push_cell_list_entry(
        &mut self,
        depth: usize,
        entry: ListEntry,
    ) -> Result<(), ConvertError> {
        self.flush_into_cell(depth + 1, depth)?;
        self.ensure_cell_depth(depth);
        self.cell_runs[depth - 1].flush(&mut self.cell_blocks[depth - 1]);
        self.cell_lists[depth - 1].push(entry);
        Ok(())
    }

    /// Whether unfinished cell content is pending at `depth`.
    pub fn has_pending_cell(&mut self, depth: usize) -> bool {
        self.flush_cell_run(depth);
        !self.cell_block_at(depth).is_empty()
    }

    /// Whether a row at `depth` is partially built (pending cell content or
    /// already-closed cells awaiting their `\row`).
    pub fn has_partial_row(&mut self, depth: usize) -> bool {
        self.has_pending_cell(depth)
            || self.tables.get(depth - 1).is_some_and(|t| !t.row.cells.is_empty())
    }

    /// `\cell` / `\nestcell`: close the cell, folding in any completed
    /// deeper table.
    pub fn end_cell(
        &mut self,
        depth: usize,
        style: Option<BlockStyle>,
        inlines: Vec<Inline>,
    ) -> Result<(), ConvertError> {
        self.push_cell_paragraph(depth, style, inlines)?;
        self.flush_cell_run(depth);
        // A deeper completed table belongs inside this cell.
        self.flush_into_cell(depth + 1, depth)?;
        let blocks = std::mem::take(self.cell_block_at(depth));
        let t = self.table_at(depth);
        let prop = t.row_props.get(t.prop_cursor).copied().unwrap_or_default();
        t.prop_cursor += 1;
        t.row.cells.push((blocks, prop));
        Ok(())
    }

    /// `\row` / `\nestrow`: close the row (its last cell must already be
    /// closed by the caller).
    pub fn end_row(&mut self, depth: usize) {
        let t = self.table_at(depth);
        let mut row = std::mem::take(&mut t.row);
        row.header = t.row_header;
        t.prop_cursor = 0;
        if !row.cells.is_empty() {
            t.rows.push(row);
        }
    }

    /// Take the finished table at `depth` as a block, if it has any rows.
    pub fn take_table(&mut self, depth: usize) -> Result<Option<Block>, ConvertError> {
        if self.tables.len() < depth || self.tables[depth - 1].rows.is_empty() {
            return Ok(None);
        }
        let t = std::mem::take(&mut self.tables[depth - 1]);
        let rows =
            t.rows.into_iter().map(|r| GridRow { cells: r.cells, header: r.header }).collect();
        build_edge_table(rows)
    }

    /// Build a finished table at `depth` into cell blocks one level up.
    fn flush_into_cell(&mut self, depth: usize, into: usize) -> Result<(), ConvertError> {
        if let Some(block) = self.take_table(depth)? {
            self.flush_cell_run(into);
            self.cell_block_at(into).push(block);
        }
        Ok(())
    }

    /// Collapse any dangling nested tables outward into their parent cells.
    pub fn collapse_nested(&mut self) -> Result<(), ConvertError> {
        for depth in (2..=self.tables.len()).rev() {
            self.flush_into_cell(depth, depth - 1)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styled_paragraphs_are_grouped_within_one_cell() {
        let mut state = TableState::new();
        state.begin_row(1);
        state.declare_cell(1, 1000);
        state.push_cell_paragraph(1, Some(BlockStyle::Code), vec![Inline::plain("first")]).unwrap();
        state.end_cell(1, Some(BlockStyle::Code), vec![Inline::plain("second")]).unwrap();
        state.end_row(1);

        let Block::Table(table) = state.take_table(1).unwrap().unwrap() else {
            panic!("expected a table")
        };
        let crate::model::CellSlot::Origin(cell) = &table.grid[0][0] else {
            panic!("expected an origin cell")
        };
        let [Block::CodeBlock { text, .. }] = &cell.blocks[..] else {
            panic!("unexpected cell blocks: {:?}", cell.blocks)
        };
        assert_eq!(text, "first\nsecond");
    }
}
