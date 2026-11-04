//! The app's settings (DESIGN.md 4.5), one flat JSON object in
//! `$XDG_CONFIG_HOME/dev.markitai.native/settings.json`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::write_atomically;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfigSource {
    #[default]
    User,
    Builtin,
}

/// The conversion switches an item captures when it is added or retried.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Options {
    pub llm: bool,
    pub ocr: bool,
    pub screenshot: bool,
    pub alt: bool,
    pub desc: bool,
    pub config: ConfigSource,
}

pub const ZOOM_MIN: f64 = 0.5;
pub const ZOOM_MAX: f64 = 3.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    #[serde(rename = "conversion.llm")]
    pub llm: bool,
    #[serde(rename = "conversion.ocr")]
    pub ocr: bool,
    #[serde(rename = "conversion.screenshot")]
    pub screenshot: bool,
    #[serde(rename = "conversion.desc")]
    pub desc: bool,
    #[serde(rename = "conversion.alt")]
    pub alt: bool,
    #[serde(rename = "output.folder")]
    pub output_folder: Option<PathBuf>,
    #[serde(rename = "configuration.source")]
    pub config: ConfigSource,
    #[serde(rename = "view.zoom")]
    pub zoom: f64,
    #[serde(rename = "window.width")]
    pub window_width: i32,
    #[serde(rename = "window.height")]
    pub window_height: i32,
    #[serde(rename = "window.maximized")]
    pub window_maximized: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            llm: false,
            ocr: false,
            screenshot: false,
            desc: false,
            alt: false,
            output_folder: None,
            config: ConfigSource::User,
            zoom: 1.0,
            window_width: 1200,
            window_height: 780,
            window_maximized: false,
        }
    }
}

impl Settings {
    /// A missing or unreadable file means defaults.
    pub fn load(path: &Path) -> Settings {
        let mut settings: Settings = std::fs::read(path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        settings.zoom = clamp_zoom(settings.zoom);
        if settings
            .output_folder
            .as_deref()
            .is_some_and(|folder| !folder.is_absolute())
        {
            settings.output_folder = None;
        }
        settings
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        text.push('\n');
        write_atomically(path, text.as_bytes())
    }

    pub fn options(&self) -> Options {
        Options {
            llm: self.llm,
            ocr: self.ocr,
            screenshot: self.screenshot,
            alt: self.alt,
            desc: self.desc,
            config: self.config,
        }
    }

    /// The output folder, or the platform default under Documents.
    pub fn output_folder_or(&self, default: &Path) -> PathBuf {
        self.output_folder
            .clone()
            .unwrap_or_else(|| default.to_path_buf())
    }
}

pub fn clamp_zoom(zoom: f64) -> f64 {
    if zoom.is_finite() {
        zoom.clamp(ZOOM_MIN, ZOOM_MAX)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_unreadable_files_mean_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        assert_eq!(Settings::load(&path), Settings::default());
        std::fs::write(&path, "{not json").unwrap();
        assert_eq!(Settings::load(&path), Settings::default());
    }

    #[test]
    fn missing_keys_keep_their_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        for text in ["", "{}"] {
            std::fs::write(&path, text).unwrap();
            assert_eq!(Settings::load(&path), Settings::default(), "{text:?}");
        }
        std::fs::write(&path, r#"{"conversion.ocr": true}"#).unwrap();
        let expected = Settings {
            ocr: true,
            ..Settings::default()
        };
        assert_eq!(Settings::load(&path), expected);
        assert_eq!(expected.zoom, 1.0);
        assert_eq!((expected.window_width, expected.window_height), (1200, 780));
        assert_eq!(expected.config, ConfigSource::User);
    }

    #[test]
    fn settings_round_trip_as_one_flat_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("settings.json");
        let settings = Settings {
            llm: true,
            alt: true,
            output_folder: Some("/srv/out".into()),
            config: ConfigSource::Builtin,
            zoom: 1.25,
            window_maximized: true,
            ..Settings::default()
        };
        settings.save(&path).unwrap();
        let written: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(written["conversion.llm"], true);
        assert_eq!(written["output.folder"], "/srv/out");
        assert_eq!(written["configuration.source"], "builtin");
        assert_eq!(written["view.zoom"], 1.25);
        assert_eq!(written["window.width"], 1200);
        assert_eq!(Settings::load(&path), settings);
        assert!(!dir.path().join("nested").join("settings.json.tmp").exists());
    }

    #[test]
    fn out_of_range_values_fall_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{"view.zoom": 9, "output.folder": "relative/dir", "conversion.ocr": true}"#).unwrap();
        let settings = Settings::load(&path);
        assert_eq!(settings.zoom, ZOOM_MAX);
        assert_eq!(settings.output_folder, None);
        assert!(settings.ocr);
    }
}
