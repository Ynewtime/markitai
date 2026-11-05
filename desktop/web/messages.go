package main

import (
	"fmt"
	"runtime"

	"github.com/egoist/mygo"
)

// Texts are the app's own words, in the workbench's three languages: the
// status pages, the quit question, the save questions and errors, and what
// the app writes to serve.log.
type Texts struct {
	Lang    string // "en", "zh" or "ja", as the workbench names it; serve speaks Chinese or English
	HTMLTag string // the status page's lang, as the workbench's <html lang>

	Starting       string
	ExitedTitle    string
	ExitedBody     func(e Exited) string
	ExitedHint     string
	NotReadyTitle  string
	NotReadyBody   func(seconds int) string
	NoProgramTitle string
	NoProgramBody  func(path string) string
	SpawnTitle     string
	SpawnBody      func(err string) string
	StoppingTitle  string
	StoppingBody   string
	StoppingNow    string
	Retry          string
	ShowLog        string
	StopNow        string

	Unfinished   func(b busy) string
	QuitDetail   string
	QuitWhenDone string
	StopAndQuit  string
	Cancel       string

	SaveFailed     string
	DownloadFailed string
	Exists         func(name string) string
	ExistsDetail   string
	Replace        string

	// MenuItems name the web view's items a context menu keeps, by their
	// IDs (contextMenu).
	MenuItems map[string]string

	LoginEnvFailed func(reason string) string
}

// menuItems names a context menu's editing items, in their order. Emoji
// is named too: the web view would name it in its own language, which on
// Windows is Traditional Chinese where the app shows Simplified Chinese
// (languagesOn).
func menuItems(undo, redo, cut, copy, paste, pasteAndMatchStyle, del, selectAll, emoji string) map[string]string {
	return map[string]string{mygo.ContextMenuUndo: undo, mygo.ContextMenuRedo: redo, mygo.ContextMenuCut: cut,
		mygo.ContextMenuCopy: copy, mygo.ContextMenuPaste: paste, mygo.ContextMenuPasteAndMatchStyle: pasteAndMatchStyle,
		mygo.ContextMenuDelete: del, mygo.ContextMenuSelectAll: selectAll, mygo.ContextMenuEmoji: emoji}
}

// texts are the app's words by the workbench's language.
var texts = map[string]Texts{"en": en, "zh": zh, "ja": ja}

// exitText is how serve ended, for the Failed page: by a signal, or with a
// code as the system writes it.
func exitText(e Exited, signal, code func(string) string) string {
	if e.Signal != "" {
		return signal(e.Signal)
	}
	return code(codeText(runtime.GOOS, e.Code))
}

var en = Texts{
	Lang:        "en",
	HTMLTag:     "en",
	Starting:    "Starting Markitai Web…",
	ExitedTitle: "Markitai Web's server stopped",
	ExitedBody: func(e Exited) string {
		return exitText(e, func(s string) string { return "It was ended by " + s + "." },
			func(c string) string { return "It exited with code " + c + "." })
	},
	ExitedHint:    "If Markitai Web quit unexpectedly a moment ago, it may still be finishing conversions; try again shortly.",
	NotReadyTitle: "Markitai Web's server did not start",
	NotReadyBody: func(seconds int) string {
		return fmt.Sprintf("It did not answer within %d seconds and was stopped.", seconds)
	},
	NoProgramTitle: "This copy of Markitai Web is incomplete",
	NoProgramBody:  func(path string) string { return "It has no server program at " + path + "." },
	SpawnTitle:     "Markitai Web could not start its server",
	SpawnBody:      func(err string) string { return err },
	StoppingTitle:  "Finishing conversions…",
	StoppingBody:   "Markitai Web quits when the running conversions are done.",
	StoppingNow:    "Stopping…",
	Retry:          "Try Again",
	ShowLog:        "Show Log",
	StopNow:        "Stop Now and Quit",

	Unfinished: func(b busy) string {
		switch {
		case !b.Known:
			return "Conversions may be unfinished; Markitai Web could not count them."
		case b.N == 1:
			return "1 conversion is unfinished."
		}
		return fmt.Sprintf("%d conversions are unfinished.", b.N)
	},
	QuitDetail:   "Quit When Done lets running conversions finish and cancels queued ones; Stop and Quit stops them all now.",
	QuitWhenDone: "Quit When Done",
	StopAndQuit:  "Stop and Quit",
	Cancel:       "Cancel",

	SaveFailed:     "The PDF could not be saved.",
	DownloadFailed: "The download failed.",
	Exists:         func(name string) string { return "“" + name + "” already exists. Do you want to replace it?" },
	ExistsDetail:   "Replacing it overwrites what it holds.",
	Replace:        "Replace",

	MenuItems: menuItems("Undo", "Redo", "Cut", "Copy", "Paste", "Paste and Match Style", "Delete", "Select All", "Emoji"),

	LoginEnvFailed: func(reason string) string {
		return "Markitai Web could not read your login shell's environment (" + reason + "). The server runs " +
			"without it: programs on your PATH, exported keys and a MARKITAI_HOME set by your shell's startup " +
			"files are missing."
	},
}

var zh = Texts{
	Lang:        "zh",
	HTMLTag:     "zh-CN",
	Starting:    "正在启动 Markitai Web…",
	ExitedTitle: "Markitai Web 的服务已停止",
	ExitedBody: func(e Exited) string {
		return exitText(e, func(s string) string { return "服务被信号 " + s + " 终止。" },
			func(c string) string { return "服务以代码 " + c + " 退出。" })
	},
	ExitedHint:    "如果 Markitai Web 刚刚意外退出，它可能还在完成转换，请稍后再试。",
	NotReadyTitle: "Markitai Web 的服务未能启动",
	NotReadyBody: func(seconds int) string {
		return fmt.Sprintf("服务在 %d 秒内没有响应，已被停止。", seconds)
	},
	NoProgramTitle: "这份 Markitai Web 不完整",
	NoProgramBody:  func(path string) string { return "缺少服务程序：" + path },
	SpawnTitle:     "Markitai Web 无法启动服务",
	SpawnBody:      func(err string) string { return err },
	StoppingTitle:  "正在完成转换…",
	StoppingBody:   "正在运行的转换完成后，Markitai Web 就会退出。",
	StoppingNow:    "正在停止…",
	Retry:          "重试",
	ShowLog:        "显示日志",
	StopNow:        "立即停止并退出",

	Unfinished: func(b busy) string {
		if !b.Known {
			return "可能还有未完成的转换，Markitai Web 无法统计数量。"
		}
		return fmt.Sprintf("还有 %d 个转换未完成。", b.N)
	},
	QuitDetail:   "“完成后退出”会让正在运行的转换完成，并取消排队中的转换；“停止并退出”会立即全部停止。",
	QuitWhenDone: "完成后退出",
	StopAndQuit:  "停止并退出",
	Cancel:       "取消",

	SaveFailed:     "无法保存 PDF。",
	DownloadFailed: "下载失败。",
	Exists:         func(name string) string { return "“" + name + "”已存在。要替换它吗？" },
	ExistsDetail:   "替换会覆盖其中的内容。",
	Replace:        "替换",

	MenuItems: menuItems("撤销", "重做", "剪切", "复制", "粘贴", "粘贴并匹配样式", "删除", "全选", "表情符号"),

	LoginEnvFailed: func(reason string) string {
		return "Markitai Web 无法读取登录 shell 的环境（" + reason + "）。服务在没有它的情况下运行：" +
			"PATH 中的程序、导出的密钥，以及 shell 启动文件设置的 MARKITAI_HOME 都不可用。"
	},
}

var ja = Texts{
	Lang:        "ja",
	HTMLTag:     "ja",
	Starting:    "Markitai Web を起動しています…",
	ExitedTitle: "Markitai Web のサーバーが停止しました",
	ExitedBody: func(e Exited) string {
		return exitText(e, func(s string) string { return "シグナル " + s + " で終了しました。" },
			func(c string) string { return "コード " + c + " で終了しました。" })
	},
	ExitedHint:    "Markitai Web が少し前に予期せず終了した場合は、まだ変換を完了させている途中かもしれません。しばらくしてから再試行してください。",
	NotReadyTitle: "Markitai Web のサーバーが起動しませんでした",
	NotReadyBody: func(seconds int) string {
		return fmt.Sprintf("%d 秒以内に応答がなかったため、停止しました。", seconds)
	},
	NoProgramTitle: "この Markitai Web は不完全です",
	NoProgramBody:  func(path string) string { return "サーバープログラムがありません：" + path },
	SpawnTitle:     "Markitai Web はサーバーを起動できませんでした",
	SpawnBody:      func(err string) string { return err },
	StoppingTitle:  "変換を完了しています…",
	StoppingBody:   "実行中の変換が終わると、Markitai Web は終了します。",
	StoppingNow:    "停止しています…",
	Retry:          "再試行",
	ShowLog:        "ログを表示",
	StopNow:        "今すぐ停止して終了",

	Unfinished: func(b busy) string {
		if !b.Known {
			return "未完了の変換があるかもしれません。Markitai Web はその数を確認できませんでした。"
		}
		return fmt.Sprintf("未完了の変換が %d 件あります。", b.N)
	},
	QuitDetail:   "「完了後に終了」は実行中の変換を完了させ、待機中の変換を取り消します。「停止して終了」はすべてを今すぐ停止します。",
	QuitWhenDone: "完了後に終了",
	StopAndQuit:  "停止して終了",
	Cancel:       "キャンセル",

	SaveFailed:     "PDF を保存できませんでした。",
	DownloadFailed: "ダウンロードに失敗しました。",
	Exists: func(name string) string {
		return "「" + name + "」はすでに存在します。置き換えますか？"
	},
	ExistsDetail: "置き換えると、その内容は上書きされます。",
	Replace:      "置き換える",

	MenuItems: menuItems("取り消す", "やり直す", "カット", "コピー", "ペースト", "ペーストしてスタイルを合わせる", "削除",
		"すべてを選択", "絵文字"),

	LoginEnvFailed: func(reason string) string {
		return "Markitai Web はログインシェルの環境を読み取れませんでした（" + reason + "）。サーバーはその環境なしで動作します。" +
			"PATH 上のプログラム、エクスポートしたキー、シェルの起動ファイルで設定した MARKITAI_HOME は使えません。"
	},
}
