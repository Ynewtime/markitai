//go:build unix

package main

import (
	"bytes"
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
// startup files a terminal's does, and has it print its environment. A job a
// startup file leaves running (`sleep 3 &!`, a daemon) may hold the output
// open after the shell exits: once the shell has exited the job goes with its
// process group, and one in a session of its own is given a second more. What
// the shell printed between the markers by then is its environment.
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
	stdout, err := newOutput()
	if err != nil {
		return nil, err
	}
	defer stdout.r.Close()
	stderr, err := newOutput()
	if err != nil {
		stdout.w.Close()
		return nil, err
	}
	defer stderr.r.Close()
	cmd.Stdout, cmd.Stderr = stdout.w, stderr.w
	err = cmd.Start()
	stdout.w.Close()
	stderr.w.Close()
	if err != nil {
		return nil, err
	}
	err = cmd.Wait()
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, fmt.Errorf("its startup files did not finish within %v", loginTimeout)
	}
	end := time.Now().Add(time.Second)
	out, errOut := stdout.until(end), stderr.until(end)
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			if lines := strings.Split(strings.TrimSpace(string(errOut)), "\n"); lines[len(lines)-1] != "" {
				return nil, fmt.Errorf("%w: %s", err, lines[len(lines)-1])
			}
		}
		return nil, err
	}
	return parseLoginEnv(out, marker)
}

// output is a pipe for a child's output, read to its end on a goroutine of
// its own, so that a holder that keeps it open holds nobody.
type output struct {
	r, w *os.File
	buf  bytes.Buffer
	done chan struct{}
}

func newOutput() (*output, error) {
	r, w, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	o := &output{r: r, w: w, done: make(chan struct{})}
	go func() {
		_, _ = o.buf.ReadFrom(r)
		close(o.done)
	}()
	return o, nil
}

// until is what was read once the pipe ended, or at end if it had not.
func (o *output) until(end time.Time) []byte {
	_ = o.r.SetReadDeadline(end)
	<-o.done
	return o.buf.Bytes()
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
