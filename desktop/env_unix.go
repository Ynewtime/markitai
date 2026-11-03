//go:build unix

package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"syscall"
	"time"
)

// loginTimeout bounds the shell: a startup file that waits for input or the
// network must not hold the window on Starting.
const loginTimeout = 5 * time.Second

// loginEnv is the environment of the user's login shell. Finder and desktop
// launchers start apps with PATH=/usr/bin:/bin:/usr/sbin:/sbin and none of
// the user's startup files, which would leave serve without Homebrew,
// ~/.local/bin, the subscription CLIs and exported keys. An app started from
// a terminal (TERM is set) already has that environment. On failure it
// returns fallbackEnv and the reason.
func loginEnv(ctx context.Context) ([]string, error) {
	if os.Getenv("TERM") != "" {
		return os.Environ(), nil
	}
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh"
		if runtime.GOOS == "darwin" {
			shell = "/bin/zsh"
		}
	}
	marker := "markitai-env-" + string(newToken())[:16]
	script := "printf %s " + marker + "; env -0; printf %s " + marker
	ctx, cancel := context.WithTimeout(ctx, loginTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, shell, "-l", "-i", "-c", script)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = time.Second
	out, err := cmd.Output()
	if err == nil {
		var env []string
		if env, err = parseLoginEnv(out, marker); err == nil {
			return env, nil
		}
	}
	return fallbackEnv(), err
}

// fallbackEnv is the app's own environment with the usual tool directories
// added to PATH.
func fallbackEnv() []string {
	env := os.Environ()
	path := filepath.SplitList(value(env, "PATH"))
	for _, dir := range []string{"/opt/homebrew/bin", "/usr/local/bin", filepath.Join(os.Getenv("HOME"), ".local", "bin")} {
		if !slices.Contains(path, dir) {
			path = append(path, dir)
		}
	}
	return append(env, "PATH="+strings.Join(path, string(filepath.ListSeparator)))
}
