package main

import (
	"bytes"
	"errors"
	"path/filepath"
	"strings"
)

// childEnv is markitai serve's environment: the user's login environment, so
// what works after `markitai serve` in their terminal works here (PATH for
// Chrome, LibreOffice and the subscription CLIs, exported keys, proxies),
// without the variables only a parent markitai hands a server
// (MARKITAI_SERVE_*) and the app's own (MYGO_*, such as the automation
// socket), plus:
//
//   - MARKITAI_SERVE_TOKEN, this launch's token;
//   - MARKITAI_HOME as home, an absolute path, when the login environment
//     names one (it may be relative or start with ~);
//   - LANG for lang ("zh" or "en") when none of LANG, LC_ALL, LC_MESSAGES
//     and MARKITAI_LANG is set, so serve's messages, and the log tail a
//     failure shows, follow the system language.
func childEnv(login []string, token Token, lang, home string) []string {
	env := make([]string, 0, len(login)+2)
	for _, kv := range login {
		name, _, _ := strings.Cut(kv, "=")
		upper := strings.ToUpper(name)
		if strings.HasPrefix(upper, "MARKITAI_SERVE_") || strings.HasPrefix(upper, "MYGO_") || upper == "MARKITAI_HOME" {
			continue
		}
		env = append(env, kv)
	}
	if value(login, "MARKITAI_HOME") != "" {
		env = append(env, "MARKITAI_HOME="+home)
	}
	env = append(env, "MARKITAI_SERVE_TOKEN="+string(token))
	if value(login, "LANG") == "" && value(login, "LC_ALL") == "" && value(login, "LC_MESSAGES") == "" &&
		value(login, "MARKITAI_LANG") == "" {
		locale := "en_US.UTF-8"
		if lang == "zh" {
			locale = "zh_CN.UTF-8"
		}
		env = append(env, "LANG="+locale)
	}
	return env
}

// value is name's value in env, the last entry winning as in exec.
func value(env []string, name string) string {
	found := ""
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == name {
			found = v
		}
	}
	return found
}

// markitaiHome is the MARKITAI_HOME the CLI would use with env: its
// MARKITAI_HOME, ~ expanded and a relative one taken against the user's
// home (where a login shell starts), else ~/.markitai. The app and the CLI
// share it: configuration, keys, models, browsers and the workbench history.
func markitaiHome(env []string, userHome string) string {
	home := value(env, "MARKITAI_HOME")
	if home == "" {
		return filepath.Join(userHome, ".markitai")
	}
	if rest, ok := strings.CutPrefix(home, "~"); ok && (rest == "" || rest[0] == '/' || rest[0] == filepath.Separator) {
		return filepath.Join(userHome, rest)
	}
	if !filepath.IsAbs(home) {
		return filepath.Join(userHome, home)
	}
	return filepath.Clean(home)
}

// parseLoginEnv reads `env -0` output that a login shell printed between two
// markers, ignoring whatever its startup files printed around them.
func parseLoginEnv(out []byte, marker string) ([]string, error) {
	_, rest, ok := bytes.Cut(out, []byte(marker))
	if !ok {
		return nil, errors.New("the login shell printed no environment")
	}
	body, _, ok := bytes.Cut(rest, []byte(marker))
	if !ok {
		return nil, errors.New("the login shell's environment was cut short")
	}
	var env []string
	for _, kv := range bytes.Split(body, []byte{0}) {
		if bytes.IndexByte(kv, '=') > 0 {
			env = append(env, string(kv))
		}
	}
	if value(env, "PATH") == "" {
		return nil, errors.New("the login shell's environment has no PATH")
	}
	return env, nil
}
