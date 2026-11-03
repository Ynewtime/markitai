package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestChildEnv(t *testing.T) {
	login := []string{
		"PATH=/opt/homebrew/bin:/usr/bin",
		"OPENAI_API_KEY=k",
		"MARKITAI_SERVE_TOKEN=stale",
		"MARKITAI_SERVE_TOKEN_FD=3",
		"MARKITAI_SERVE_LISTEN_FD=4",
		"MYGO_AUTOMATION=e2e.sock",
	}
	env := childEnv(login, "fresh", "en", "/home-of-markitai")
	for _, want := range []string{"PATH=/opt/homebrew/bin:/usr/bin", "OPENAI_API_KEY=k",
		"MARKITAI_SERVE_TOKEN=fresh", "LANG=en_US.UTF-8"} {
		if !slices.Contains(env, want) {
			t.Errorf("childEnv lacks %s: %v", want, env)
		}
	}
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(name, "MYGO_") || (strings.HasPrefix(name, "MARKITAI_SERVE_") && kv != "MARKITAI_SERVE_TOKEN=fresh") {
			t.Errorf("childEnv passes %s on", kv)
		}
		if name == "MARKITAI_HOME" {
			t.Errorf("childEnv sets MARKITAI_HOME the login environment did not: %s", kv)
		}
	}
}

func TestChildEnvLanguage(t *testing.T) {
	if env := childEnv([]string{"PATH=/bin"}, "t", "zh", "/h"); !slices.Contains(env, "LANG=zh_CN.UTF-8") {
		t.Errorf("Chinese system: %v", env)
	}
	for _, set := range []string{"LANG=fr_FR.UTF-8", "LC_ALL=de_DE.UTF-8", "LC_MESSAGES=ja_JP.UTF-8", "MARKITAI_LANG=en"} {
		env := childEnv([]string{"PATH=/bin", set}, "t", "zh", "/h")
		if !slices.Contains(env, set) || slices.ContainsFunc(env, func(kv string) bool { return strings.HasPrefix(kv, "LANG=zh") }) {
			t.Errorf("with %s the user's choice must stand: %v", set, env)
		}
	}
}

func TestChildEnvNamesTheResolvedHome(t *testing.T) {
	env := childEnv([]string{"MARKITAI_HOME=~/state", "PATH=/bin"}, "t", "en", "/u/state")
	if value(env, "MARKITAI_HOME") != "/u/state" || slices.Contains(env, "MARKITAI_HOME=~/state") {
		t.Errorf("childEnv = %v", env)
	}
}

func TestMarkitaiHome(t *testing.T) {
	user := filepath.FromSlash("/u")
	cases := []struct {
		env  []string
		want string
	}{
		{nil, "/u/.markitai"},
		{[]string{"MARKITAI_HOME="}, "/u/.markitai"},
		{[]string{"MARKITAI_HOME=/srv/markitai/"}, "/srv/markitai"},
		{[]string{"MARKITAI_HOME=~/state"}, "/u/state"},
		{[]string{"MARKITAI_HOME=~"}, "/u"},
		{[]string{"MARKITAI_HOME=state"}, "/u/state"},
		{[]string{"MARKITAI_HOME=/a", "MARKITAI_HOME=/b"}, "/b"},
	}
	for _, tc := range cases {
		if got := markitaiHome(tc.env, user); got != filepath.FromSlash(tc.want) {
			t.Errorf("markitaiHome(%v) = %q; want %q", tc.env, got, tc.want)
		}
	}
}

func TestParseLoginEnv(t *testing.T) {
	out := []byte("Welcome!\nMK" + "PATH=/opt/homebrew/bin:/usr/bin\x00HOME=/u\x00MULTI=a\nb\x00" + "MKbye\n")
	env, err := parseLoginEnv(out, "MK")
	if err != nil || !slices.Equal(env, []string{"PATH=/opt/homebrew/bin:/usr/bin", "HOME=/u", "MULTI=a\nb"}) {
		t.Errorf("parseLoginEnv = %q, %v", env, err)
	}
	// dash -i sets $_ to the last word of the command before env -0: the
	// marker printf printed.
	underscore := []byte("MK" + "HOME=/u\x00_=MK\x00PATH=/bin\x00" + "MK")
	if env, err := parseLoginEnv(underscore, "MK"); err != nil || !slices.Equal(env, []string{"HOME=/u", "_=MK", "PATH=/bin"}) {
		t.Errorf("parseLoginEnv with $_ the marker = %q, %v", env, err)
	}
	for _, bad := range []string{"no markers", "MKPATH=/bin\x00", "MKHOME=/u\x00MK"} {
		if _, err := parseLoginEnv([]byte(bad), "MK"); err == nil {
			t.Errorf("parseLoginEnv(%q) accepted", bad)
		}
	}
}
