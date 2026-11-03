package main

import (
	"fmt"
	"strings"
)

// Texts are the app's own words, in the workbench's two languages: the
// status pages, the quit question and save errors.
type Texts struct {
	Lang string // "en" or "zh", the status page's lang and the child's LANG

	Starting       string
	ExitedTitle    string
	ExitedBody     func(code int) string
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

	StillRunning func(n int) string
	QuitDetail   string
	QuitWhenDone string
	StopAndQuit  string
	Cancel       string

	SaveFailed     string
	DownloadFailed string
}

var en = Texts{
	Lang:          "en",
	Starting:      "Starting Markitai…",
	ExitedTitle:   "Markitai's server stopped",
	ExitedBody:    func(code int) string { return fmt.Sprintf("It exited with code %d.", code) },
	ExitedHint:    "If Markitai quit unexpectedly a moment ago, it may still be finishing conversions; try again shortly.",
	NotReadyTitle: "Markitai's server did not start",
	NotReadyBody: func(seconds int) string {
		return fmt.Sprintf("It did not answer within %d seconds and was stopped.", seconds)
	},
	NoProgramTitle: "This copy of Markitai is incomplete",
	NoProgramBody:  func(path string) string { return "It has no server program at " + path + "." },
	SpawnTitle:     "Markitai could not start its server",
	SpawnBody:      func(err string) string { return err },
	StoppingTitle:  "Finishing conversions…",
	StoppingBody:   "Markitai quits when the running conversions are done.",
	StoppingNow:    "Stopping…",
	Retry:          "Try Again",
	ShowLog:        "Show Log",
	StopNow:        "Stop Now and Quit",

	StillRunning: func(n int) string {
		if n == 1 {
			return "1 conversion is still running."
		}
		return fmt.Sprintf("%d conversions are still running.", n)
	},
	QuitDetail:   "Quit When Done lets running conversions finish and cancels queued ones; Stop and Quit stops them all now.",
	QuitWhenDone: "Quit When Done",
	StopAndQuit:  "Stop and Quit",
	Cancel:       "Cancel",

	SaveFailed:     "The PDF could not be saved.",
	DownloadFailed: "The download failed.",
}

var zh = Texts{
	Lang:          "zh",
	Starting:      "正在启动 Markitai…",
	ExitedTitle:   "Markitai 的服务已停止",
	ExitedBody:    func(code int) string { return fmt.Sprintf("服务以代码 %d 退出。", code) },
	ExitedHint:    "如果 Markitai 刚刚意外退出，它可能还在完成转换，请稍后再试。",
	NotReadyTitle: "Markitai 的服务未能启动",
	NotReadyBody: func(seconds int) string {
		return fmt.Sprintf("服务在 %d 秒内没有响应，已被停止。", seconds)
	},
	NoProgramTitle: "这份 Markitai 不完整",
	NoProgramBody:  func(path string) string { return "缺少服务程序：" + path },
	SpawnTitle:     "Markitai 无法启动服务",
	SpawnBody:      func(err string) string { return err },
	StoppingTitle:  "正在完成转换…",
	StoppingBody:   "正在运行的转换完成后，Markitai 就会退出。",
	StoppingNow:    "正在停止…",
	Retry:          "重试",
	ShowLog:        "显示日志",
	StopNow:        "立即停止并退出",

	StillRunning: func(n int) string { return fmt.Sprintf("还有 %d 个转换正在进行。", n) },
	QuitDetail:   "“完成后退出”会让正在运行的转换完成，并取消排队中的转换；“停止并退出”会立即全部停止。",
	QuitWhenDone: "完成后退出",
	StopAndQuit:  "停止并退出",
	Cancel:       "取消",

	SaveFailed:     "无法保存 PDF。",
	DownloadFailed: "下载失败。",
}

// textsFor picks Chinese when the first preferred language is Chinese.
func textsFor(preferred []string) Texts {
	if len(preferred) > 0 && (preferred[0] == "zh" || strings.HasPrefix(preferred[0], "zh-")) {
		return zh
	}
	return en
}
