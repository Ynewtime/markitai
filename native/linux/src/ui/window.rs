//! The main window (DESIGN.md 6.1): navigation sidebar, Convert and History,
//! each a list column beside a detail pane, collapsing at 960 and 640 sp.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;
use gtk::{gdk, gio, glib};
use uuid::Uuid;

use super::actions;
use super::detail::{Detail, Shared, View};
use super::rows::Row;
use crate::model::history::History;
use crate::model::queue::Queue;
use crate::model::runs::Runs;
use crate::model::settings::Settings;
use crate::model::{self, compact_utc};
use crate::strings::{tr, trf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Convert,
    History,
}

/// The URL field with its Add button and the error under it.
pub struct UrlRow {
    pub entry: gtk::Entry,
    pub add: gtk::Button,
    pub error: gtk::Box,
    pub row: gtk::Box,
}

pub struct Win {
    pub app: adw::Application,
    pub window: adw::ApplicationWindow,
    pub toasts: adw::ToastOverlay,
    pub outer: adw::NavigationSplitView,
    pub nav_list: gtk::ListBox,
    pub content_page: adw::NavigationPage,
    pub pages: gtk::Stack,
    pub drop_area: gtk::Widget,

    pub empty_status: adw::StatusPage,
    pub empty_choose: gtk::Button,
    pub empty_url: UrlRow,
    pub drop_bar_label: gtk::Label,
    pub convert_split: adw::NavigationSplitView,
    pub convert_list: gtk::ListBox,
    pub convert_menu: gtk::PopoverMenu,
    pub list_url: UrlRow,
    pub convert_detail: Rc<Detail>,
    pub rows: RefCell<HashMap<Uuid, Row>>,

    pub history_split: adw::NavigationSplitView,
    pub history_list: gtk::ListBox,
    pub history_menu: gtk::PopoverMenu,
    pub history_search: gtk::SearchEntry,
    pub history_placeholder: gtk::Label,
    pub history_detail: Rc<Detail>,
    pub history_rows: RefCell<HashMap<Uuid, Row>>,
    pub history_markdown: RefCell<Option<(Uuid, Option<Rc<String>>)>>,
    pub history_loading: Cell<Option<Uuid>>,

    pub shared: Rc<Shared>,
    pub zoom_css: gtk::CssProvider,
    pub queue: RefCell<Queue>,
    pub history: RefCell<History>,
    pub history_selected: Cell<Option<Uuid>>,
    pub settings: RefCell<Settings>,
    pub page: Cell<Page>,
    pub syncing: Cell<bool>,
    pub ticking: Cell<bool>,
    /// The logout inhibitor held while items convert, and the count its
    /// reason names.
    pub inhibit: Cell<Option<(u32, usize)>>,
    /// The quit dialog while it shows, so its count can follow the queue.
    pub quit_dialog: RefCell<Option<adw::AlertDialog>>,
    /// Conversions on worker threads, shared with Quit.
    pub runs: Runs,
    pub default_output: PathBuf,
}

/// A header bar showing `title`, named by it (DESIGN.md 12).
fn header(title: &str) -> adw::HeaderBar {
    let bar = adw::HeaderBar::new();
    bar.set_title_widget(Some(&adw::WindowTitle::new(title, "")));
    bar.upcast_ref::<gtk::Widget>()
        .update_property(&[gtk::accessible::Property::Label(title)]);
    bar
}

fn icon_button(icon: &str, key: &str, action: &str) -> gtk::Button {
    let button = gtk::Button::builder()
        .icon_name(icon)
        .tooltip_text(tr(key))
        .action_name(action)
        .build();
    button.update_property(&[gtk::accessible::Property::Label(&tr(key))]);
    button
}

fn url_row() -> UrlRow {
    let entry = gtk::Entry::builder()
        .placeholder_text(tr("convert.url.placeholder"))
        .primary_icon_name("insert-link-symbolic")
        .input_purpose(gtk::InputPurpose::Url)
        .hexpand(true)
        .build();
    entry.update_property(&[gtk::accessible::Property::Label(&tr("convert.url.label"))]);
    let add = gtk::Button::with_label(&tr("convert.url.add"));
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    row.append(&entry);
    row.append(&add);
    let error_icon = gtk::Image::from_icon_name("dialog-error-symbolic");
    error_icon.add_css_class("error");
    let error_text = gtk::Label::builder()
        .label(tr("convert.url.invalid"))
        .xalign(0.0)
        .wrap(true)
        .build();
    let error = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    error.append(&error_icon);
    error.append(&error_text);
    error.set_visible(false);
    UrlRow {
        entry,
        add,
        error,
        row,
    }
}

fn list_box(name_key: &str) -> gtk::ListBox {
    let list = gtk::ListBox::builder()
        .selection_mode(gtk::SelectionMode::Single)
        .css_classes(["navigation-sidebar"])
        .build();
    list.update_property(&[gtk::accessible::Property::Label(&tr(name_key))]);
    list
}

fn nav_row(icon: &str, key: &str) -> gtk::ListBoxRow {
    let line = gtk::Box::new(gtk::Orientation::Horizontal, 12);
    line.append(&gtk::Image::from_icon_name(icon));
    line.append(&gtk::Label::new(Some(&tr(key))));
    let row = gtk::ListBoxRow::builder().child(&line).build();
    row.update_property(&[gtk::accessible::Property::Label(&tr(key))]);
    row
}

fn inner_split(sidebar: &adw::NavigationPage, content: &adw::NavigationPage) -> adw::NavigationSplitView {
    let split = adw::NavigationSplitView::builder()
        .sidebar_width_fraction(0.32)
        .min_sidebar_width(260.0)
        .max_sidebar_width(420.0)
        .build();
    split.set_sidebar(Some(sidebar));
    split.set_content(Some(content));
    split
}

impl Win {
    pub fn new(app: &adw::Application) -> Rc<Win> {
        let data_dir = model::data_dir();
        let settings = Settings::load(&model::settings_path());
        let (history, reset) = History::load(&data_dir, &compact_utc(std::time::SystemTime::now()));
        let default_output = glib::user_special_dir(glib::UserDirectory::Documents)
            .unwrap_or_else(|| model::home().join("Documents"))
            .join("Markitai");

        let window = adw::ApplicationWindow::builder()
            .application(app)
            .title(tr("app.name"))
            .default_width(settings.window_width.max(360))
            .default_height(settings.window_height.max(480))
            .width_request(360)
            .height_request(480)
            .build();
        if settings.window_maximized {
            window.maximize();
        }

        let shared = Rc::new(Shared {
            view: Cell::new(View::Preview),
            expanded: RefCell::default(),
            session: webkit6::NetworkSession::new_ephemeral(),
            zoom: Cell::new(settings.zoom),
        });

        // Navigation sidebar.
        let nav_list = gtk::ListBox::builder()
            .css_classes(["navigation-sidebar"])
            .build();
        nav_list.update_property(&[gtk::accessible::Property::Label(&tr("app.name"))]);
        nav_list.append(&nav_row("x-office-document-symbolic", "nav.convert"));
        nav_list.append(&nav_row("document-open-recent-symbolic", "nav.history"));
        let menu = gio::Menu::new();
        menu.append(Some(&tr("nav.preferences")), Some("app.preferences"));
        menu.append(Some(&tr("menu.about")), Some("app.about"));
        let menu_button = gtk::MenuButton::builder()
            .icon_name("open-menu-symbolic")
            .menu_model(&menu)
            .primary(true)
            .tooltip_text(tr("menu.main"))
            .build();
        menu_button.update_property(&[gtk::accessible::Property::Label(&tr("menu.main"))]);
        let sidebar_header = header(&tr("app.name"));
        sidebar_header.pack_end(&menu_button);
        let sidebar_view = adw::ToolbarView::new();
        sidebar_view.add_top_bar(&sidebar_header);
        sidebar_view.set_content(Some(
            &gtk::ScrolledWindow::builder()
                .child(&nav_list)
                .hscrollbar_policy(gtk::PolicyType::Never)
                .build(),
        ));
        let sidebar_page = adw::NavigationPage::new(&sidebar_view, &tr("app.name"));

        // Convert, empty.
        let empty_choose = gtk::Button::builder()
            .label(tr("convert.choose"))
            .action_name("win.choose-files")
            .halign(gtk::Align::Center)
            .css_classes(["pill", "suggested-action"])
            .build();
        let or_row = gtk::Box::new(gtk::Orientation::Horizontal, 12);
        let line = || {
            let separator = gtk::Separator::new(gtk::Orientation::Horizontal);
            separator.set_hexpand(true);
            separator.set_valign(gtk::Align::Center);
            separator
        };
        or_row.append(&line());
        or_row.append(
            &gtk::Label::builder()
                .label(tr("convert.or"))
                .css_classes(["caption", "dimmed"])
                .build(),
        );
        or_row.append(&line());
        let empty_url = url_row();
        let empty_child = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(14)
            .build();
        empty_child.append(&empty_choose);
        empty_child.append(&or_row);
        empty_child.append(&empty_url.row);
        empty_child.append(&empty_url.error);
        let empty_status = adw::StatusPage::builder()
            .icon_name("document-open-symbolic")
            .title(tr("convert.empty.title"))
            .child(&empty_child)
            .build();
        let drop_zone = gtk::Box::builder().css_classes(["drop-zone"]).build();
        drop_zone.append(&empty_status);
        empty_status.set_hexpand(true);
        drop_zone.update_property(&[gtk::accessible::Property::Label(&tr("a11y.drop_zone"))]);
        // 560 wide when there is room, narrower down to the 360 window.
        let drop_clamp = adw::Clamp::builder()
            .child(&drop_zone)
            .maximum_size(560)
            .tightening_threshold(560)
            .margin_start(12)
            .margin_end(12)
            .valign(gtk::Align::Center)
            .build();
        let empty_scroller = gtk::ScrolledWindow::builder()
            .child(&drop_clamp)
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .build();
        let empty_view = adw::ToolbarView::new();
        empty_view.add_top_bar(&header(&tr("nav.convert")));
        empty_view.set_content(Some(&empty_scroller));

        // Convert, with items: the list column.
        let drop_bar_label = gtk::Label::builder()
            .label(tr("convert.drop.bar"))
            .xalign(0.0)
            .hexpand(true)
            .css_classes(["dimmed"])
            .build();
        let drop_bar = gtk::Box::builder()
            .spacing(10)
            .css_classes(["drop-bar"])
            .build();
        drop_bar.append(&gtk::Image::from_icon_name("document-open-symbolic"));
        drop_bar.append(&drop_bar_label);
        drop_bar.append(
            &gtk::Button::builder()
                .label(tr("convert.choose"))
                .action_name("win.choose-files")
                .valign(gtk::Align::Center)
                .build(),
        );
        drop_bar.update_property(&[gtk::accessible::Property::Label(&tr("a11y.drop_zone"))]);
        let list_url = url_row();
        let list_top = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(8)
            .css_classes(["list-pane-top"])
            .build();
        list_top.append(&drop_bar);
        list_top.append(&list_url.row);
        list_top.append(&list_url.error);
        let convert_list = list_box("a11y.conversions");
        let convert_column = gtk::Box::new(gtk::Orientation::Vertical, 0);
        convert_column.append(&list_top);
        convert_column.append(
            &gtk::ScrolledWindow::builder()
                .child(&convert_list)
                .hscrollbar_policy(gtk::PolicyType::Never)
                .vexpand(true)
                .build(),
        );
        let convert_header = header(&tr("nav.convert"));
        convert_header.pack_end(&icon_button(
            "edit-clear-all-symbolic",
            "convert.clear_finished",
            "win.clear-finished",
        ));
        let convert_view = adw::ToolbarView::new();
        convert_view.add_top_bar(&convert_header);
        convert_view.set_content(Some(&convert_column));
        let convert_sidebar = adw::NavigationPage::new(&convert_view, &tr("nav.convert"));
        let convert_detail = Detail::new(shared.clone(), "win.remove");
        let convert_split = inner_split(&convert_sidebar, &convert_detail.page);

        // History.
        let history_search = gtk::SearchEntry::builder()
            .placeholder_text(tr("history.search"))
            .build();
        history_search.update_property(&[gtk::accessible::Property::Label(&tr("history.search"))]);
        let history_list = list_box("nav.history");
        let history_placeholder = gtk::Label::builder()
            .wrap(true)
            .justify(gtk::Justification::Center)
            .margin_top(24)
            .margin_start(12)
            .margin_end(12)
            .css_classes(["dimmed"])
            .build();
        history_list.set_placeholder(Some(&history_placeholder));
        let history_top = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .css_classes(["list-pane-top"])
            .build();
        history_top.append(&history_search);
        let history_column = gtk::Box::new(gtk::Orientation::Vertical, 0);
        history_column.append(&history_top);
        history_column.append(
            &gtk::ScrolledWindow::builder()
                .child(&history_list)
                .hscrollbar_policy(gtk::PolicyType::Never)
                .vexpand(true)
                .build(),
        );
        let history_header = header(&tr("nav.history"));
        history_header.pack_end(&icon_button(
            "user-trash-symbolic",
            "history.clear",
            "win.clear-history",
        ));
        let history_view = adw::ToolbarView::new();
        history_view.add_top_bar(&history_header);
        history_view.set_content(Some(&history_column));
        let history_sidebar = adw::NavigationPage::new(&history_view, &tr("nav.history"));
        let history_detail = Detail::new(shared.clone(), "win.remove");
        let history_split = inner_split(&history_sidebar, &history_detail.page);

        let history_empty = adw::StatusPage::builder()
            .icon_name("document-open-recent-symbolic")
            .title(tr("history.empty.title"))
            .description(tr("history.empty.body"))
            .vexpand(true)
            .build();
        let history_empty_view = adw::ToolbarView::new();
        let history_empty_header = header(&tr("nav.history"));
        history_empty_header.pack_end(&icon_button(
            "user-trash-symbolic",
            "history.clear",
            "win.clear-history",
        ));
        history_empty_view.add_top_bar(&history_empty_header);
        history_empty_view.set_content(Some(&history_empty));

        // Only the page on show sets the window's minimum size.
        let pages = gtk::Stack::builder().hhomogeneous(false).vhomogeneous(false).build();
        pages.add_named(&empty_view, Some("convert-empty"));
        pages.add_named(&convert_split, Some("convert"));
        pages.add_named(&history_empty_view, Some("history-empty"));
        pages.add_named(&history_split, Some("history"));
        let content_page = adw::NavigationPage::new(&pages, &tr("nav.convert"));

        let outer = adw::NavigationSplitView::builder()
            .min_sidebar_width(180.0)
            .max_sidebar_width(200.0)
            .build();
        outer.set_sidebar(Some(&sidebar_page));
        outer.set_content(Some(&content_page));

        // Collapsed, the window opens on the page, not on the navigation.
        outer.set_show_content(true);

        let toasts = adw::ToastOverlay::new();
        toasts.set_child(Some(&outer));
        window.set_content(Some(&toasts));

        // DESIGN.md 14. The last matching breakpoint applies alone, so each
        // narrower one repeats the wider ones' setters.
        let breakpoint = |condition: &str, outer_too: bool, fold: bool| {
            let breakpoint = adw::Breakpoint::new(
                adw::BreakpointCondition::parse(condition).expect("breakpoint"),
            );
            if outer_too {
                breakpoint.add_setter(&outer, "collapsed", Some(&true.to_value()));
            }
            breakpoint.add_setter(&convert_split, "collapsed", Some(&true.to_value()));
            breakpoint.add_setter(&history_split, "collapsed", Some(&true.to_value()));
            if fold {
                convert_detail.fold_actions_at(&breakpoint);
                history_detail.fold_actions_at(&breakpoint);
            }
            window.add_breakpoint(breakpoint);
        };
        breakpoint("max-width: 960sp", false, false);
        breakpoint("max-width: 640sp", true, false);
        breakpoint("max-width: 420sp", true, true);

        let zoom_css = gtk::CssProvider::new();
        if let Some(display) = gdk::Display::default() {
            gtk::style_context_add_provider_for_display(
                &display,
                &zoom_css,
                gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
            );
        }
        let convert_menu = gtk::PopoverMenu::from_model(None::<&gio::MenuModel>);
        convert_menu.set_parent(&convert_list);
        convert_menu.set_has_arrow(false);
        let history_menu = gtk::PopoverMenu::from_model(None::<&gio::MenuModel>);
        history_menu.set_parent(&history_list);
        history_menu.set_has_arrow(false);

        let win = Rc::new(Win {
            app: app.clone(),
            window,
            toasts,
            outer: outer.clone(),
            nav_list,
            content_page,
            pages,
            drop_area: outer.upcast(),
            empty_status,
            empty_choose,
            empty_url,
            drop_bar_label,
            convert_split,
            convert_list,
            convert_menu,
            list_url,
            convert_detail,
            rows: RefCell::default(),
            history_split,
            history_list,
            history_menu,
            history_search,
            history_placeholder,
            history_detail,
            history_rows: RefCell::default(),
            history_markdown: RefCell::default(),
            history_loading: Cell::new(None),
            shared,
            zoom_css,
            queue: RefCell::default(),
            history: RefCell::new(history),
            history_selected: Cell::new(None),
            settings: RefCell::new(settings),
            page: Cell::new(Page::Convert),
            syncing: Cell::new(false),
            ticking: Cell::new(false),
            inhibit: Cell::new(None),
            quit_dialog: RefCell::new(None),
            runs: Runs::default(),
            default_output,
        });

        win.connect_navigation();
        win.connect_convert();
        win.connect_history();
        win.connect_drop();
        actions::install_window(&win);
        win.apply_zoom();
        win.refresh_output_folder();
        win.refresh_convert();
        win.refresh_history();
        win.select_page(Page::Convert);
        // The stack shows convert-empty from the start, so refresh_pages
        // never switches to it here.
        win.empty_choose.grab_focus();

        let weak = Rc::downgrade(&win);
        adw::StyleManager::default().connect_dark_notify(move |_| {
            if let Some(win) = weak.upgrade() {
                win.convert_detail.rerender();
                win.history_detail.rerender();
            }
        });
        let weak = Rc::downgrade(&win);
        win.window.connect_close_request(move |_| match weak.upgrade() {
            Some(win) if win.queue.borrow().converting() > 0 => {
                win.ask_to_quit();
                glib::Propagation::Stop
            }
            Some(win) => {
                win.save_window_state();
                glib::Propagation::Proceed
            }
            None => glib::Propagation::Proceed,
        });

        if let Some(name) = reset {
            win.notice(&trf("history.reset", &[("name", &name)]));
        }
        win
    }

    /// Tab walks the rows and the selection follows it; only activation (a
    /// click, Return or Space) changes the page, so Tab goes on to the page's
    /// own controls (DESIGN.md 12). Until then the selection stays on the
    /// page that shows.
    fn connect_navigation(self: &Rc<Self>) {
        let weak = Rc::downgrade(self);
        self.nav_list.connect_row_selected(move |_, _| {
            if let Some(win) = weak.upgrade()
                && !win.syncing.get()
            {
                win.select_nav_row();
            }
        });
        let weak = Rc::downgrade(self);
        self.nav_list.connect_row_activated(move |_, row| {
            if let Some(win) = weak.upgrade() {
                win.select_page(if row.index() == 1 {
                    Page::History
                } else {
                    Page::Convert
                });
                win.outer.set_show_content(true);
            }
        });
    }

    fn select_nav_row(&self) {
        let index = match self.page.get() {
            Page::Convert => 0,
            Page::History => 1,
        };
        self.syncing.set(true);
        self.nav_list.select_row(self.nav_list.row_at_index(index).as_ref());
        self.syncing.set(false);
    }

    /// Show Convert or History.
    pub fn select_page(&self, page: Page) {
        self.page.set(page);
        self.select_nav_row();
        self.content_page.set_title(&tr(match page {
            Page::Convert => "nav.convert",
            Page::History => "nav.history",
        }));
        self.refresh_pages();
        self.refresh_actions();
    }

    pub fn refresh_pages(&self) {
        let name = match self.page.get() {
            Page::Convert if self.queue.borrow().is_empty() => "convert-empty",
            Page::Convert => "convert",
            Page::History if self.history.borrow().is_empty() => "history-empty",
            Page::History => "history",
        };
        if self.pages.visible_child_name().as_deref() != Some(name) {
            self.pages.set_visible_child_name(name);
            if name == "convert-empty" {
                self.empty_choose.grab_focus();
            }
        }
    }

    /// Accept files and web links dropped anywhere on Convert or History.
    fn connect_drop(self: &Rc<Self>) {
        let target = gtk::DropTarget::new(glib::Type::INVALID, gdk::DragAction::COPY);
        target.set_types(&[gdk::FileList::static_type(), glib::Type::STRING]);
        let weak = Rc::downgrade(self);
        target.connect_enter(move |_, _, _| {
            if let Some(win) = weak.upgrade() {
                win.set_drop_hover(true);
            }
            gdk::DragAction::COPY
        });
        let weak = Rc::downgrade(self);
        target.connect_leave(move |_| {
            if let Some(win) = weak.upgrade() {
                win.set_drop_hover(false);
            }
        });
        let weak = Rc::downgrade(self);
        target.connect_drop(move |_, value, _, _| {
            let Some(win) = weak.upgrade() else {
                return false;
            };
            win.set_drop_hover(false);
            if let Ok(files) = value.get::<gdk::FileList>() {
                win.add_gfiles(&files.files());
                return true;
            }
            if let Ok(text) = value.get::<String>() {
                let urls: Vec<String> = text
                    .lines()
                    .filter(|line| !line.starts_with('#'))
                    .filter_map(crate::model::source::parse_web_address)
                    .collect();
                if !urls.is_empty() {
                    win.add_sources(Vec::new(), urls);
                    return true;
                }
            }
            false
        });
        self.drop_area.add_controller(target);
    }

    fn set_drop_hover(&self, on: bool) {
        if on {
            self.drop_area.add_css_class("drop-hover");
        } else {
            self.drop_area.remove_css_class("drop-hover");
        }
        let key = if on { "convert.drop.release" } else { "convert.drop.bar" };
        self.drop_bar_label.set_label(&tr(key));
        self.empty_status.set_title(&tr(if on {
            "convert.drop.release"
        } else {
            "convert.empty.title"
        }));
    }

    /// The output folder now, as conversions use it.
    pub fn output_folder(&self) -> PathBuf {
        self.settings.borrow().output_folder_or(&self.default_output)
    }

    pub fn display_path(&self, path: &std::path::Path) -> String {
        crate::model::format::home_abbreviated(path, Some(&glib::home_dir()))
    }

    pub fn refresh_output_folder(&self) {
        let folder = self.display_path(&self.output_folder());
        self.empty_status
            .set_description(Some(&trf("convert.empty.body", &[("folder", &folder)])));
    }

    pub fn save_settings(&self) {
        let _ = self.settings.borrow().save(&model::settings_path());
    }

    pub fn save_window_state(&self) {
        let (width, height) = self.window.default_size();
        {
            let mut settings = self.settings.borrow_mut();
            settings.window_maximized = self.window.is_maximized();
            if width > 0 && height > 0 {
                settings.window_width = width;
                settings.window_height = height;
            }
        }
        self.save_settings();
    }

    /// A toast, 5 s (DESIGN.md 6.7).
    pub fn notice(&self, text: &str) {
        self.toast(text, 5);
    }

    pub fn toast(&self, text: &str, seconds: u32) {
        let toast = adw::Toast::builder()
            .title(text)
            .timeout(seconds)
            .use_markup(false)
            .build();
        self.toasts.add_toast(toast);
    }

    pub fn announce(&self, text: &str) {
        self.window
            .upcast_ref::<gtk::Widget>()
            .announce(text, gtk::AccessibleAnnouncementPriority::Medium);
    }

    pub fn apply_zoom(&self) {
        let zoom = self.settings.borrow().zoom;
        self.shared.zoom.set(zoom);
        self.zoom_css.load_from_string(&format!(
            "textview.markdown-text {{ font-size: {:.2}pt; }}",
            10.0 * zoom
        ));
        self.convert_detail.apply_zoom();
        self.history_detail.apply_zoom();
    }

    /// The quit dialog while anything converts (DESIGN.md 6.7 and 7.3).
    pub fn ask_to_quit(self: &Rc<Self>) {
        // Asked already: a second Ctrl+Q or close leaves that dialog up.
        if self.quit_dialog.borrow().is_some() {
            return;
        }
        // Any other dialog (Clear History, Preferences, About) closes first,
        // as its Cancel or Close would.
        if let Some(other) = self.window.visible_dialog() {
            other.close();
        }
        let dialog = adw::AlertDialog::new(Some(&tr("quit.title")), None);
        dialog.add_responses(&[("cancel", &tr("common.cancel")), ("quit", &tr("quit.confirm"))]);
        dialog.set_response_appearance("quit", adw::ResponseAppearance::Destructive);
        dialog.set_default_response(Some("cancel"));
        dialog.set_close_response("cancel");
        let win = self.clone();
        dialog.connect_response(None, move |_, response| {
            win.quit_dialog.replace(None);
            if response == "quit" {
                win.discard_converting();
                win.save_window_state();
                win.app.quit();
            }
        });
        dialog.present(Some(&self.window));
        self.quit_dialog.replace(Some(dialog));
        self.follow_quit_dialog();
    }

    /// The quit dialog's count follows the queue; once nothing converts,
    /// nothing would be lost, so the app quits (DESIGN.md 7.3).
    pub fn follow_quit_dialog(&self) {
        let (converting, queued) = {
            let queue = self.queue.borrow();
            (queue.converting(), queue.queued())
        };
        if converting > 0 {
            if let Some(dialog) = self.quit_dialog.borrow().as_ref() {
                dialog.set_body(&super::convert::quit_body(crate::strings::lang(), converting, queued));
            }
            return;
        }
        let Some(dialog) = self.quit_dialog.take() else {
            return;
        };
        dialog.force_close();
        self.save_window_state();
        self.app.quit();
    }

    /// Quit confirmed (DESIGN.md 7.3): a run that ended on its worker before
    /// now reaches the list and History with its folder; the folders of
    /// items still Converting are deleted, and their conversions end with the
    /// process. `run` calls it again after the main loop stops, in case a
    /// conversion thread wrote into a folder after the first pass.
    pub fn discard_converting(self: &Rc<Self>) {
        for (id, folder, outcome) in self.runs.quit() {
            self.finish(id, folder, outcome);
        }
        for folder in self.queue.borrow().converting_folders() {
            let _ = std::fs::remove_dir_all(folder);
        }
    }

    pub fn quit(self: &Rc<Self>) {
        if self.queue.borrow().converting() > 0 {
            self.ask_to_quit();
        } else {
            self.save_window_state();
            self.app.quit();
        }
    }
}
