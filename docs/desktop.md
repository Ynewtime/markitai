# Desktop versions

Markitai has three desktop versions. They run side by side on one computer.
Each has its own identifier and keeps its own app data. All three read the same
markitai configuration as the CLI (`~/.markitai`, or `MARKITAI_HOME`). No
release publishes any of them yet. Build them from a checkout.

| | Markitai Web | Markitai Native | Markitai Mygo |
|---|---|---|---|
| What it is | The [browser workbench](web-ui.md) in a window of its own, served by the bundled `markitai` | An app per system, drawn with that system's own controls, with the core in process | One app for every system, drawn by mygo's `ui` package, with the core in process |
| Minimum system | macOS 13 (Apple silicon), Windows 10 (x86-64), Linux (x86-64) | macOS 13, Windows 10 1809, GNOME 48 | macOS 13, Windows (x86-64), Linux (x86-64) |
| Identifier | `dev.markitai.web` | `dev.markitai.native` | `dev.markitai.mygo` |
| Source | `desktop/web/` | `desktop/native/macos/`, `desktop/native/windows/`, `desktop/native/linux/` | `desktop/mygo/` |
| Guide | [Markitai Web](desktop-web.md) | [Markitai Native](desktop-native.md) | [Markitai Mygo](desktop-mygo.md) |

Markitai Web has every workbench feature and converts exactly as the CLI of the
same version does. Markitai Native and Markitai Mygo share one product
specification, [`desktop/DESIGN.md`](../desktop/DESIGN.md), with a smaller
feature set: files and web pages, a preview, a history and five feature
switches. All three use the app icon `desktop/icon.png`.
