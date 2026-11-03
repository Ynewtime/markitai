//go:build unix

package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// launchedFromFinder runs loginEnv as a launch from Finder or a desktop
// launcher does: no TERM, the user's SHELL, and a HOME holding files.
func launchedFromFinder(t *testing.T, ctx context.Context, shell string, files map[string]string) ([]string, error) {
	t.Helper()
	if _, err := os.Stat(shell); err != nil {
		t.Skipf("%s is not installed", shell)
	}
	home := t.TempDir()
	for name, body := range files {
		path := filepath.Join(home, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("HOME", home)
	t.Setenv("MARKITAI_HOME", filepath.Join(home, "markitai-home"))
	t.Setenv("SHELL", shell)
	t.Setenv("TERM", "")
	for _, name := range []string{"ENV", "ZDOTDIR", "BASH_ENV"} {
		unsetenv(t, name)
	}
	t.Setenv("MARKITAI_PROBE", "")
	return loginEnv(ctx)
}

func unsetenv(t *testing.T, name string) {
	t.Setenv(name, "")
	if err := os.Unsetenv(name); err != nil {
		t.Fatal(err)
	}
}

func TestLoginEnvReadsEachShellsStartupFiles(t *testing.T) {
	cases := []struct {
		shell string
		files map[string]string
		want  []string
	}{
		{"/bin/zsh", map[string]string{".zprofile": "export MARKITAI_LOGIN=zprofile\n",
			".zshrc": "export MARKITAI_PROBE=zshrc\n"}, []string{"MARKITAI_LOGIN=zprofile", "MARKITAI_PROBE=zshrc"}},
		{"/bin/bash", map[string]string{".bash_profile": "export MARKITAI_PROBE=bash_profile\n"},
			[]string{"MARKITAI_PROBE=bash_profile"}},
		{"/bin/dash", map[string]string{".profile": "export MARKITAI_PROBE=profile\n"},
			[]string{"MARKITAI_PROBE=profile"}},
		{"/usr/bin/dash", map[string]string{".profile": "export MARKITAI_PROBE=profile\n"},
			[]string{"MARKITAI_PROBE=profile"}},
		{"/bin/tcsh", map[string]string{".cshrc": "setenv MARKITAI_PROBE cshrc\n", ".login": "setenv MARKITAI_LOGIN login\n"},
			[]string{"MARKITAI_PROBE=cshrc", "MARKITAI_LOGIN=login"}},
		{"/bin/csh", map[string]string{".cshrc": "setenv MARKITAI_PROBE cshrc\n", ".login": "setenv MARKITAI_LOGIN login\n"},
			[]string{"MARKITAI_PROBE=cshrc", "MARKITAI_LOGIN=login"}},
		{"/usr/bin/fish", map[string]string{".config/fish/config.fish": "set -gx MARKITAI_PROBE config\n"},
			[]string{"MARKITAI_PROBE=config"}},
	}
	for _, tc := range cases {
		t.Run(tc.shell, func(t *testing.T) {
			env, err := launchedFromFinder(t, context.Background(), tc.shell, tc.files)
			if err != nil {
				t.Fatalf("loginEnv: %v", err)
			}
			for _, want := range tc.want {
				if !slices.Contains(env, want) {
					t.Errorf("the login environment lacks %s: %q", want, env)
				}
			}
			if value(env, "PATH") == "" || value(env, "HOME") != os.Getenv("HOME") {
				t.Errorf("the login environment lost PATH or HOME: %q", env)
			}
		})
	}
}

func TestLoginEnvGivesUpOnASlowShell(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	started := time.Now()
	env, err := launchedFromFinder(t, ctx, "/bin/bash", map[string]string{".bash_profile": "sleep 30\n"})
	if err == nil || !strings.Contains(err.Error(), "/bin/bash") {
		t.Errorf("loginEnv = %v; want an error naming the shell", err)
	}
	if took := time.Since(started); took > 5*time.Second {
		t.Errorf("loginEnv took %v past its deadline", took)
	}
	if value(env, "PATH") == "" {
		t.Errorf("the fallback environment has no PATH: %q", env)
	}
}
