//go:build unix

package main

import (
	"context"
	"errors"
	"fmt"
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
// network must not hold the window on Starting. VS Code waits as long for
// the same environment (application.shellEnvironmentResolutionTimeout).
const loginTimeout = 10 * time.Second

// loginEnv is the environment of the user's login shell. Finder and desktop
// launchers start apps with PATH=/usr/bin:/bin:/usr/sbin:/sbin and none of
// the user's startup files, which would leave serve without Homebrew,
// ~/.local/bin, the subscription CLIs, exported keys and MARKITAI_HOME. An
// app started from a terminal (TERM is set) already has that environment.
// On failure it returns fallbackEnv and why, naming the shell.
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
	env, err := shellEnv(ctx, shell)
	if err != nil {
		return fallbackEnv(), fmt.Errorf("%s: %w", shell, err)
	}
	return env, nil
}

// shellEnv runs shell as an interactive login shell, so that it reads the
// startup files a terminal's does, and has it print its environment.
func shellEnv(ctx context.Context, shell string) ([]string, error) {
	marker := "markitai-env-" + string(newToken())[:16]
	script := "printf %s " + marker + "; env -0; printf %s " + marker
	ctx, cancel := context.WithTimeout(ctx, loginTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, shell, "-l", "-i", "-c", script)
	if name := filepath.Base(shell); name == "csh" || name == "tcsh" {
		// They take -l only as their one flag, read .cshrc in every shell,
		// and the script from stdin.
		cmd = exec.CommandContext(ctx, shell, "-l")
		cmd.Stdin = strings.NewReader(script + "\n")
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = time.Second
	out, err := cmd.Output()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, fmt.Errorf("its startup files did not finish within %v", loginTimeout)
	}
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			if lines := strings.Split(strings.TrimSpace(string(exit.Stderr)), "\n"); lines[len(lines)-1] != "" {
				return nil, fmt.Errorf("%w: %s", err, lines[len(lines)-1])
			}
		}
		return nil, err
	}
	return parseLoginEnv(out, marker)
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
