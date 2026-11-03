//! The detail pane (DESIGN.md 6.4), used by Convert and by History: header,
//! warnings, the Preview or Markdown body, the error block and the footer.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use adw::prelude::*;
use gtk::{gdk, gio, glib};
use uuid::Uuid;
use webkit6::prelude::*;

use crate::model::errors::{Second, error_copy, second_key};
use crate::model::format;
use crate::model::source::Kind;
use crate::strings::{settings_name, tr, trf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Preview,
    Markdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glyph {
    Queued,
    Converting,
    Done,
    Failed,
}

/// What the body shows.
#[derive(Clone, Debug)]
pub enum Body {
    Queued,
    Converting {
        started: Instant,
    },
    Failed {
        code: String,
        message: String,
        sentence: String,
    },
    /// `markdown` is None while History reads the file.
    Done {
        markdown: Option<Rc<String>>,
        base_dir: PathBuf,
    },
    Missing {
        path: String,
    },
}

#[derive(Clone, Debug)]
pub struct Shown {
    pub id: Uuid,
    pub title: String,
    pub kind: Kind,
    pub source: String,
    pub status: String,
    pub warnings: Vec<String>,
    pub footer: Option<String>,
    pub body: Body,
}

impl Shown {
    fn glyph(&self) -> Glyph {
        match self.body {
            Body::Queued => Glyph::Queued,
            Body::Converting { .. } => Glyph::Converting,
            Body::Failed { .. } => Glyph::Failed,
            Body::Done { .. } | Body::Missing { .. } => Glyph::Done,
        }
    }
}

/// Shared by both detail panes: the chosen tab, each item's warnings
/// disclosure, the preview's web session and the zoom.
pub struct Shared {
    pub view: Cell<View>,
    pub expanded: RefCell<HashSet<Uuid>>,
    pub session: webkit6::NetworkSession,
    pub zoom: Cell<f64>,
}

pub struct Detail {
    pub page: adw::NavigationPage,
    shared: Rc<Shared>,
    switch: adw::ToggleGroup,
    title_stack: gtk::Stack,
    actions: gtk::Box,
    action_buttons: gtk::Box,
    more: gtk::MenuButton,
    outer: gtk::Stack,
    title: gtk::Label,
    kind_icon: gtk::Image,
    source: gtk::Label,
    status_glyph: gtk::Stack,
    status: gtk::Label,
    warnings_box: gtk::ListBox,
    warnings: adw::ExpanderRow,
    warning_rows: RefCell<Vec<gtk::Widget>>,
    body: gtk::Stack,
    pub web: webkit6::WebView,
    text: gtk::TextView,
    note: gtk::Label,
    error_sentence: gtk::Label,
    error_second: gtk::Button,
    error_message: gtk::Label,
    missing: adw::StatusPage,
    footer: gtk::Label,
    footer_bar: gtk::Box,
    shown: RefCell<Option<Shown>>,
    rendered: RefCell<Option<(Uuid, bool, usize)>>,
    generation: Rc<Cell<u64>>,
    updating_expander: Cell<bool>,
}

fn icon_button(icon: &str, key: &str, action: &str) -> gtk::Button {
    let button = gtk::Button::builder()
        .icon_name(icon)
        .tooltip_text(tr(key))
        .action_name(action)
        .build();
    button.add_css_class("flat");
    button.update_property(&[gtk::accessible::Property::Label(&tr(key))]);
    button
}

pub fn glyph_widget(glyph: Glyph) -> gtk::Widget {
    match glyph {
        Glyph::Converting => {
            let spinner = adw::Spinner::new();
            spinner.set_size_request(16, 16);
            spinner.upcast()
        }
        Glyph::Queued => gtk::Image::from_icon_name("content-loading-symbolic").upcast(),
        Glyph::Done => {
            let image = gtk::Image::from_icon_name("object-select-symbolic");
            image.add_css_class("success");
            image.upcast()
        }
        Glyph::Failed => {
            let image = gtk::Image::from_icon_name("dialog-error-symbolic");
            image.add_css_class("error");
            image.upcast()
        }
    }
}

pub fn kind_icon_name(kind: Kind) -> &'static str {
    match kind {
        Kind::File => "text-x-generic-symbolic",
        Kind::Url => "web-browser-symbolic",
    }
}

impl Detail {
    pub fn new(shared: Rc<Shared>, remove_action: &str) -> Rc<Detail> {
        let switch = adw::ToggleGroup::new();
        switch.add(adw::Toggle::builder().name("preview").label(tr("detail.preview")).build());
        switch.add(adw::Toggle::builder().name("markdown").label(tr("detail.markdown")).build());
        switch
            .upcast_ref::<gtk::Widget>()
            .update_property(&[gtk::accessible::Property::Label(&tr("detail.view"))]);
        let title_stack = gtk::Stack::new();
        title_stack.add_named(&gtk::Box::new(gtk::Orientation::Horizontal, 0), Some("none"));
        title_stack.add_named(&switch, Some("switch"));
        title_stack.set_hhomogeneous(false);

        // The three actions as buttons, or in one menu below 420sp (the
        // window's breakpoint swaps them). `actions` shows for Done only.
        const ACTIONS: [(&str, &str, &str); 3] = [
            ("edit-copy-symbolic", "detail.copy", "win.copy"),
            ("document-save-as-symbolic", "detail.save_as", "win.save-as"),
            ("folder-open-symbolic", "detail.show.gnome", "win.show"),
        ];
        let action_buttons = gtk::Box::new(gtk::Orientation::Horizontal, 4);
        let menu = gio::Menu::new();
        for (icon, key, action) in ACTIONS {
            action_buttons.append(&icon_button(icon, key, action));
            menu.append(Some(&tr(key)), Some(action));
        }
        let more = gtk::MenuButton::builder()
            .icon_name("view-more-symbolic")
            .tooltip_text(tr("detail.more"))
            .menu_model(&menu)
            .visible(false)
            .build();
        more.update_property(&[gtk::accessible::Property::Label(&tr("detail.more"))]);
        let actions = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        actions.append(&action_buttons);
        actions.append(&more);

        let header = adw::HeaderBar::new();
        header.set_title_widget(Some(&title_stack));
        header.pack_end(&actions);

        let title = gtk::Label::builder()
            .xalign(0.0)
            .wrap(true)
            .wrap_mode(gtk::pango::WrapMode::WordChar)
            .selectable(true)
            .css_classes(["title-3"])
            .build();
        let kind_icon = gtk::Image::new();
        let source = gtk::Label::builder()
            .xalign(0.0)
            .selectable(true)
            .ellipsize(gtk::pango::EllipsizeMode::Middle)
            .hexpand(true)
            .css_classes(["dimmed"])
            .build();
        let source_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        kind_icon.add_css_class("dimmed");
        source_row.append(&kind_icon);
        source_row.append(&source);
        let status_glyph = gtk::Stack::new();
        for (name, glyph) in [
            ("queued", Glyph::Queued),
            ("converting", Glyph::Converting),
            ("done", Glyph::Done),
            ("failed", Glyph::Failed),
        ] {
            status_glyph.add_named(&glyph_widget(glyph), Some(name));
        }
        let status = gtk::Label::builder().xalign(0.0).hexpand(true).build();
        let status_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        status_row.append(&status_glyph);
        status_row.append(&status);
        let head = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(4)
            .margin_start(24)
            .margin_end(24)
            .margin_top(8)
            .margin_bottom(14)
            .build();
        head.append(&title);
        head.append(&source_row);
        head.append(&status_row);

        let warnings = adw::ExpanderRow::new();
        let warning_icon = gtk::Image::from_icon_name("dialog-warning-symbolic");
        warning_icon.add_css_class("warning");
        warnings.add_prefix(&warning_icon);
        let warnings_box = gtk::ListBox::builder()
            .selection_mode(gtk::SelectionMode::None)
            .css_classes(["boxed-list"])
            .margin_start(24)
            .margin_end(24)
            .margin_bottom(14)
            .build();
        warnings_box.append(&warnings);
        warnings_box.update_property(&[gtk::accessible::Property::Label(&tr("a11y.warnings"))]);

        let web = webkit6::WebView::builder()
            .network_session(&shared.session)
            .vexpand(true)
            .hexpand(true)
            .build();
        configure_web(&web);
        let text = gtk::TextView::builder()
            .editable(false)
            .monospace(true)
            .wrap_mode(gtk::WrapMode::WordChar)
            .left_margin(24)
            .right_margin(24)
            .top_margin(24)
            .bottom_margin(24)
            .css_classes(["markdown-text", "monospace"])
            .build();
        text.update_property(&[gtk::accessible::Property::Label(&tr("a11y.markdown_text"))]);
        // A text view's own bindings take Delete and Ctrl+V even when it is
        // not editable, so the read-only Markdown hands them on first.
        text.add_controller(super::actions::list_keys(gtk::PropagationPhase::Capture));
        let text_scroller = gtk::ScrolledWindow::builder().child(&text).vexpand(true).build();

        let note = gtk::Label::builder()
            .wrap(true)
            .justify(gtk::Justification::Center)
            .valign(gtk::Align::Center)
            .vexpand(true)
            .margin_start(24)
            .margin_end(24)
            .build();

        let error_sentence = gtk::Label::builder()
            .xalign(0.0)
            .wrap(true)
            .css_classes(["title-4"])
            .build();
        let retry = gtk::Button::builder()
            .label(tr("detail.retry"))
            .action_name("win.retry")
            .css_classes(["suggested-action"])
            .build();
        let error_second = gtk::Button::new();
        let buttons = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        buttons.append(&retry);
        buttons.append(&error_second);
        let details_label = gtk::Label::builder()
            .label(tr("detail.details"))
            .xalign(0.0)
            .css_classes(["heading"])
            .margin_top(12)
            .build();
        let error_message = gtk::Label::builder()
            .xalign(0.0)
            .yalign(0.0)
            .wrap(true)
            .wrap_mode(gtk::pango::WrapMode::WordChar)
            .selectable(true)
            .css_classes(["monospace", "raw-message"])
            .build();
        let card = gtk::Box::builder().css_classes(["card"]).build();
        card.append(&error_message);
        error_message.set_hexpand(true);
        let error_block = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(12)
            .margin_start(24)
            .margin_end(24)
            .margin_top(24)
            .margin_bottom(24)
            .build();
        error_block.append(&error_sentence);
        error_block.append(&buttons);
        error_block.append(&details_label);
        error_block.append(&card);
        let error_scroller = gtk::ScrolledWindow::builder()
            .child(&error_block)
            .vexpand(true)
            .hscrollbar_policy(gtk::PolicyType::Never)
            .build();

        let missing = adw::StatusPage::builder()
            .icon_name("dialog-warning-symbolic")
            .title(tr("detail.missing.title"))
            .vexpand(true)
            .build();
        let missing_remove = gtk::Button::builder()
            .label(tr("history.remove"))
            .action_name(remove_action)
            .halign(gtk::Align::Center)
            .css_classes(["pill"])
            .build();
        missing.set_child(Some(&missing_remove));

        let body = gtk::Stack::builder().hhomogeneous(false).build();
        body.add_named(&web, Some("preview"));
        body.add_named(&text_scroller, Some("markdown"));
        body.add_named(&note, Some("note"));
        body.add_named(&error_scroller, Some("failed"));
        body.add_named(&missing, Some("missing"));
        body.add_named(&adw::Spinner::new(), Some("loading"));
        body.set_vexpand(true);

        let content = gtk::Box::new(gtk::Orientation::Vertical, 0);
        content.append(&head);
        content.append(&warnings_box);
        content.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
        content.append(&body);

        let empty = gtk::Label::builder()
            .label(tr("detail.empty"))
            .wrap(true)
            .justify(gtk::Justification::Center)
            .css_classes(["dimmed"])
            .margin_start(24)
            .margin_end(24)
            .build();
        let outer = gtk::Stack::builder().hhomogeneous(false).build();
        outer.add_named(&empty, Some("empty"));
        outer.add_named(&content, Some("item"));

        let footer = gtk::Label::builder()
            .xalign(0.0)
            .ellipsize(gtk::pango::EllipsizeMode::End)
            .css_classes(["caption", "dimmed"])
            .build();
        let footer_bar = gtk::Box::builder()
            .margin_start(24)
            .margin_end(24)
            .margin_top(10)
            .margin_bottom(10)
            .build();
        footer_bar.append(&footer);

        let toolbar = adw::ToolbarView::new();
        toolbar.add_top_bar(&header);
        toolbar.set_content(Some(&outer));
        toolbar.add_bottom_bar(&footer_bar);
        toolbar.set_bottom_bar_style(adw::ToolbarStyle::Raised);
        let page = adw::NavigationPage::new(&toolbar, &tr("app.name"));

        let detail = Rc::new(Detail {
            page,
            shared,
            switch,
            title_stack,
            actions,
            action_buttons,
            more,
            outer,
            title,
            kind_icon,
            source,
            status_glyph,
            status,
            warnings_box,
            warnings,
            warning_rows: RefCell::new(Vec::new()),
            body,
            web,
            text,
            note,
            error_sentence,
            error_second,
            error_message,
            missing,
            footer,
            footer_bar,
            shown: RefCell::new(None),
            rendered: RefCell::new(None),
            generation: Rc::new(Cell::new(0)),
            updating_expander: Cell::new(false),
        });

        let weak = Rc::downgrade(&detail);
        detail.switch.connect_active_name_notify(move |switch| {
            if let Some(detail) = weak.upgrade() {
                let view = match switch.active_name().as_deref() {
                    Some("markdown") => View::Markdown,
                    _ => View::Preview,
                };
                detail.shared.view.set(view);
                detail.refresh_body();
            }
        });
        let weak = Rc::downgrade(&detail);
        detail.warnings.connect_expanded_notify(move |row| {
            if let Some(detail) = weak.upgrade() {
                if detail.updating_expander.get() {
                    return;
                }
                if let Some(id) = detail.shown.borrow().as_ref().map(|s| s.id) {
                    let mut expanded = detail.shared.expanded.borrow_mut();
                    if row.is_expanded() {
                        expanded.insert(id);
                    } else {
                        expanded.remove(&id);
                    }
                }
            }
        });
        detail.show(None);
        detail
    }

    /// Show `shown`, or the no-selection text.
    pub fn show(&self, shown: Option<Shown>) {
        let Some(shown) = shown else {
            self.outer.set_visible_child_name("empty");
            self.title_stack.set_visible_child_name("none");
            self.actions.set_visible(false);
            self.footer_bar.set_visible(false);
            self.page.set_title(&tr("app.name"));
            *self.shown.borrow_mut() = None;
            return;
        };
        self.outer.set_visible_child_name("item");
        self.page.set_title(&shown.title);
        self.title.set_label(&shown.title);
        self.kind_icon.set_icon_name(Some(kind_icon_name(shown.kind)));
        self.source.set_label(&shown.source);
        self.source.set_ellipsize(match shown.kind {
            Kind::File => gtk::pango::EllipsizeMode::Middle,
            Kind::Url => gtk::pango::EllipsizeMode::End,
        });
        self.status.set_label(&shown.status);
        self.status_glyph.set_visible_child_name(match shown.glyph() {
            Glyph::Queued => "queued",
            Glyph::Converting => "converting",
            Glyph::Done => "done",
            Glyph::Failed => "failed",
        });

        let same_warnings = self
            .shown
            .borrow()
            .as_ref()
            .is_some_and(|old| old.id == shown.id && old.warnings == shown.warnings);
        if !same_warnings {
            for row in self.warning_rows.borrow_mut().drain(..) {
                self.warnings.remove(&row);
            }
            for warning in &shown.warnings {
                let label = gtk::Label::builder()
                    .label(warning)
                    .xalign(0.0)
                    .wrap(true)
                    .wrap_mode(gtk::pango::WrapMode::WordChar)
                    .selectable(true)
                    .margin_start(12)
                    .margin_end(12)
                    .margin_top(12)
                    .margin_bottom(12)
                    .build();
                let row = gtk::ListBoxRow::builder().child(&label).activatable(false).build();
                self.warnings.add_row(&row);
                self.warning_rows.borrow_mut().push(row.upcast());
            }
        }
        let count = shown.warnings.len();
        self.warnings_box.set_visible(count > 0 && matches!(shown.body, Body::Done { .. } | Body::Missing { .. }));
        self.warnings.set_title(&crate::strings::plural("state.warnings", count));
        self.updating_expander.set(true);
        self.warnings
            .set_expanded(self.shared.expanded.borrow().contains(&shown.id));
        self.updating_expander.set(false);

        let done = matches!(shown.body, Body::Done { .. } | Body::Missing { .. });
        self.title_stack
            .set_visible_child_name(if done { "switch" } else { "none" });
        self.actions.set_visible(done);
        self.footer_bar.set_visible(shown.footer.is_some());
        self.footer.set_label(shown.footer.as_deref().unwrap_or_default());

        match &shown.body {
            Body::Failed { code, message, sentence } => {
                let (_, second) = error_copy(code);
                self.error_sentence.set_label(sentence);
                self.error_message.set_label(message);
                match second {
                    Some(second) => {
                        let label = trf(second_key(second), &[("settings", &settings_name())]);
                        self.error_second.set_label(&label);
                        self.error_second.set_action_name(Some(match second {
                            Second::Remove => "win.remove",
                            Second::OpenSettings => "app.preferences",
                            Second::OpenConfig => "win.open-config",
                        }));
                        self.error_second.set_visible(true);
                    }
                    None => self.error_second.set_visible(false),
                }
            }
            Body::Missing { path } => {
                self.missing
                    .set_description(Some(&trf("detail.missing.body", &[("path", path)])));
            }
            _ => {}
        }
        *self.shown.borrow_mut() = Some(shown);
        self.refresh_body();
    }

    /// The body for the current item and tab.
    fn refresh_body(&self) {
        let shown = self.shown.borrow();
        let Some(shown) = shown.as_ref() else {
            return;
        };
        let view = self.shared.view.get();
        let active = match view {
            View::Preview => "preview",
            View::Markdown => "markdown",
        };
        if self.switch.active_name().as_deref() != Some(active) {
            self.switch.set_active_name(Some(active));
        }
        match &shown.body {
            Body::Queued => {
                self.note.set_label(&tr("detail.queued"));
                self.body.set_visible_child_name("note");
            }
            Body::Converting { .. } => {
                self.note.set_label(&tr("detail.converting.note"));
                self.body.set_visible_child_name("note");
            }
            Body::Failed { .. } => self.body.set_visible_child_name("failed"),
            Body::Missing { .. } => self.body.set_visible_child_name("missing"),
            Body::Done { markdown: None, .. } => self.body.set_visible_child_name("loading"),
            Body::Done {
                markdown: Some(markdown),
                base_dir,
            } => {
                let identity = (shown.id, is_dark(), Rc::as_ptr(markdown) as usize);
                match view {
                    View::Markdown => {
                        let buffer = self.text.buffer();
                        let current = buffer.text(&buffer.start_iter(), &buffer.end_iter(), false);
                        if current.as_str() != markdown.as_str() {
                            buffer.set_text(markdown);
                        }
                        self.body.set_visible_child_name("markdown");
                    }
                    View::Preview => {
                        if *self.rendered.borrow() != Some(identity) {
                            *self.rendered.borrow_mut() = Some(identity);
                            self.render(markdown.clone(), base_dir.clone());
                        }
                        self.body.set_visible_child_name("preview");
                    }
                }
            }
        }
    }

    /// Render the preview off the main thread; a newer request wins.
    fn render(&self, markdown: Rc<String>, base_dir: PathBuf) {
        let generation = self.generation.get() + 1;
        self.generation.set(generation);
        let theme = if is_dark() { "dark" } else { "light" };
        let request =
            serde_json::json!({"markdown": markdown.as_str(), "theme": theme, "base_dir": base_dir})
                .to_string();
        let base_uri = glib::filename_to_uri(&base_dir, None)
            .map(|uri| format!("{uri}/"))
            .ok();
        let web = self.web.clone();
        let current = self.generation.clone();
        glib::spawn_future_local(async move {
            let Ok(response) = gio::spawn_blocking(move || markitai_app::preview_html(&request)).await
            else {
                return;
            };
            if current.get() != generation {
                return;
            }
            let value: serde_json::Value = serde_json::from_str(&response).unwrap_or_default();
            let html = value["result"]["html"].as_str().unwrap_or_default();
            web.load_html(html, base_uri.as_deref());
        });
    }

    /// Render again with the current appearance.
    pub fn rerender(&self) {
        *self.rendered.borrow_mut() = None;
        self.refresh_body();
    }

    pub fn set_view(&self, view: View) {
        self.shared.view.set(view);
        self.refresh_body();
    }

    pub fn apply_zoom(&self) {
        self.web.set_zoom_level(self.shared.zoom.get());
    }

    /// Below `breakpoint`, the three header actions fold into one menu.
    pub fn fold_actions_at(&self, breakpoint: &adw::Breakpoint) {
        breakpoint.add_setter(&self.action_buttons, "visible", Some(&false.to_value()));
        breakpoint.add_setter(&self.more, "visible", Some(&true.to_value()));
    }

    /// Update the running conversion's elapsed time.
    pub fn tick(&self, status: impl Fn(Instant) -> String) {
        if let Some(shown) = self.shown.borrow().as_ref()
            && let Body::Converting { started } = shown.body
        {
            self.status.set_label(&status(started));
        }
    }
}

pub fn is_dark() -> bool {
    adw::StyleManager::default().is_dark()
}

pub fn elapsed_status(started: Instant) -> String {
    trf(
        "detail.converting",
        &[("elapsed", &format::elapsed(started.elapsed().as_secs()))],
    )
}

/// Scripts off, no developer tools, a transparent background, only Copy in
/// the context menu, and links handed to the desktop (DESIGN.md section 8).
fn configure_web(web: &webkit6::WebView) {
    if let Some(settings) = WebViewExt::settings(web) {
        settings.set_enable_javascript(false);
        settings.set_enable_developer_extras(false);
    }
    web.set_background_color(&gdk::RGBA::new(0.0, 0.0, 0.0, 0.0));
    web.update_property(&[gtk::accessible::Property::Label(&tr("a11y.preview"))]);
    web.connect_context_menu(|_, menu, _| {
        for item in menu.items() {
            if item.stock_action() != webkit6::ContextMenuAction::Copy {
                menu.remove(&item);
            }
        }
        menu.n_items() == 0
    });
    web.connect_decide_policy(|web, decision, kind| {
        if !matches!(
            kind,
            webkit6::PolicyDecisionType::NavigationAction
                | webkit6::PolicyDecisionType::NewWindowAction
        ) {
            return false;
        }
        let Some(navigation) = decision.downcast_ref::<webkit6::NavigationPolicyDecision>() else {
            return false;
        };
        let Some(action) = navigation.navigation_action() else {
            decision.ignore();
            return true;
        };
        let uri = action
            .request()
            .and_then(|r| r.uri())
            .map(|u| u.to_string())
            .unwrap_or_default();
        let is_link = action.navigation_type() == webkit6::NavigationType::LinkClicked
            || kind == webkit6::PolicyDecisionType::NewWindowAction;
        if !is_link {
            if action.navigation_type() == webkit6::NavigationType::Other {
                decision.use_();
            } else {
                decision.ignore();
            }
            return true;
        }
        let page = web.uri().map(|u| u.to_string()).unwrap_or_default();
        let without_fragment = |u: &str| u.split('#').next().unwrap_or_default().to_owned();
        if kind == webkit6::PolicyDecisionType::NavigationAction
            && uri.contains('#')
            && without_fragment(&uri) == without_fragment(&page)
        {
            decision.use_();
            return true;
        }
        decision.ignore();
        let lower = uri.to_ascii_lowercase();
        if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("mailto:") {
            let window = web.root().and_downcast::<gtk::Window>();
            gtk::UriLauncher::new(&uri).launch(window.as_ref(), gio::Cancellable::NONE, |_| {});
        }
        true
    });
}
