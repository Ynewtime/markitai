//! One list row (DESIGN.md 6.3): kind glyph, title, state subtitle and the
//! trailing glyph, named `<title>, <subtitle>` for screen readers.

use adw::prelude::*;
use uuid::Uuid;

use super::detail::kind_icon_name;
use crate::model::source::Kind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trailing {
    None,
    Queued,
    Converting,
    Warning,
    Failed,
}

pub struct Row {
    pub row: gtk::ListBoxRow,
    kind: gtk::Image,
    title: gtk::Label,
    subtitle: gtk::Label,
    trailing: gtk::Stack,
}

impl Row {
    pub fn new(id: Uuid) -> Row {
        let kind = gtk::Image::new();
        let title = gtk::Label::builder()
            .xalign(0.0)
            .single_line_mode(true)
            .build();
        let subtitle = gtk::Label::builder()
            .xalign(0.0)
            .single_line_mode(true)
            .ellipsize(gtk::pango::EllipsizeMode::End)
            .css_classes(["caption", "dimmed"])
            .build();
        let text = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(2)
            .hexpand(true)
            .valign(gtk::Align::Center)
            .build();
        text.append(&title);
        text.append(&subtitle);
        let trailing = gtk::Stack::new();
        trailing.add_named(&gtk::Box::new(gtk::Orientation::Horizontal, 0), Some("none"));
        trailing.add_named(&gtk::Image::from_icon_name("content-loading-symbolic"), Some("queued"));
        let spinner = adw::Spinner::new();
        spinner.set_size_request(16, 16);
        trailing.add_named(&spinner, Some("converting"));
        let warning = gtk::Image::from_icon_name("dialog-warning-symbolic");
        warning.add_css_class("warning");
        trailing.add_named(&warning, Some("warning"));
        let failed = gtk::Image::from_icon_name("dialog-error-symbolic");
        failed.add_css_class("error");
        trailing.add_named(&failed, Some("failed"));
        let line = gtk::Box::builder()
            .spacing(12)
            .margin_start(4)
            .margin_end(4)
            .build();
        line.append(&kind);
        line.append(&text);
        line.append(&trailing);
        let row = gtk::ListBoxRow::builder()
            .child(&line)
            .name(id.to_string())
            .css_classes(["item-row"])
            .build();
        Row {
            row,
            kind,
            title,
            subtitle,
            trailing,
        }
    }

    pub fn set(&self, kind: Kind, title: &str, subtitle: &str, trailing: Trailing) {
        self.kind.set_icon_name(Some(kind_icon_name(kind)));
        self.title.set_label(title);
        self.title.set_ellipsize(match kind {
            Kind::File => gtk::pango::EllipsizeMode::Middle,
            Kind::Url => gtk::pango::EllipsizeMode::End,
        });
        self.subtitle.set_label(subtitle);
        self.trailing.set_visible_child_name(match trailing {
            Trailing::None => "none",
            Trailing::Queued => "queued",
            Trailing::Converting => "converting",
            Trailing::Warning => "warning",
            Trailing::Failed => "failed",
        });
        self.row
            .update_property(&[gtk::accessible::Property::Label(&format!("{title}, {subtitle}"))]);
    }
}

/// The id a row was made for.
pub fn row_id(row: &gtk::ListBoxRow) -> Option<Uuid> {
    Uuid::parse_str(row.widget_name().as_str()).ok()
}

/// Fade a new row in over 150 ms; libadwaita skips it when animations are
/// off.
pub fn fade_in(row: &gtk::ListBoxRow) {
    row.set_opacity(0.0);
    let target = adw::PropertyAnimationTarget::new(row, "opacity");
    let animation = adw::TimedAnimation::new(row, 0.0, 1.0, 150, target);
    animation.play();
}

/// Fade a row out over 150 ms, then take it out of `list`.
pub fn fade_out(list: &gtk::ListBox, row: &gtk::ListBoxRow) {
    let target = adw::PropertyAnimationTarget::new(row, "opacity");
    let animation = adw::TimedAnimation::new(row, row.opacity(), 0.0, 150, target);
    row.set_sensitive(false);
    let list = list.clone();
    let gone = row.clone();
    animation.connect_done(move |_| list.remove(&gone));
    animation.play();
}
