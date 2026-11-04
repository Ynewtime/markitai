package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// Port is a loopback TCP port the app chose for its child; never 0, so the
// origin is known before the child starts.
type Port uint16

// cliPort is `markitai serve`'s default; the app keeps off it so a server
// the user starts in a terminal never meets the app's on its port.
const cliPort Port = 3600

func (p Port) String() string { return strconv.Itoa(int(p)) }

// Token is the child's access token for one launch. It lives in the child's
// environment and the window's URL fragment; String keeps it out of logs.
type Token string

func (Token) String() string { return "[redacted]" }

func newToken() Token {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return Token(hex.EncodeToString(b))
}

// Memory is what the app keeps between launches in PathUserData/web.json:
// the port, whose origin keeps the workbench's localStorage, and the theme
// the workbench last reported (Look).
type Memory struct {
	Port  Port   `json:"port,omitempty"`
	Theme string `json:"theme,omitempty"`
}

// memory is web.json, read once at launch. The sidecar writes the port and
// the window the theme; each change rewrites the whole file under the lock,
// so neither undoes the other's.
type memory struct {
	path  string
	mu    sync.Mutex
	m     Memory
	saved Memory // what the file holds, as far as the app knows
}

// openMemory reads path; a missing or unreadable file is the zero Memory,
// which costs one new port.
func openMemory(path string) *memory {
	m := &memory{path: path}
	if b, err := os.ReadFile(path); err == nil && json.Unmarshal(b, &m.m) != nil {
		m.m = Memory{}
	}
	m.saved = m.m
	return m
}

func (s *memory) Get() Memory {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.m
}

// Update applies change and saves the result when it differs from what the
// file holds: a save that failed is made by the next Update.
func (s *memory) Update(change func(*Memory)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	change(&s.m)
	if s.m == s.saved {
		return nil
	}
	b, err := json.Marshal(s.m)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	if err := replace(tmp, s.path); err != nil {
		return err
	}
	s.saved = s.m
	return nil
}

// replace renames from to to. Windows refuses to replace a file another
// program has open, as one reading web.json does (a virus scan, a backup):
// it tries again for a few seconds while Windows says the file is in use,
// as mygo does for its own files.
func replace(from, to string) error {
	for i := 1; ; i++ {
		err := os.Rename(from, to)
		if err == nil || i == 10 || runtime.GOOS != "windows" ||
			// ERROR_SHARING_VIOLATION, or ERROR_ACCESS_DENIED for a file
			// that is open.
			!errors.Is(err, syscall.Errno(32)) && !errors.Is(err, syscall.Errno(5)) {
			return err
		}
		time.Sleep(time.Duration(i) * 100 * time.Millisecond)
	}
}

// firstPorts is where the app looks for a port it has none: 23600 and the
// 99 after it, below the ephemeral ranges from which macOS and Windows
// (49152-65535) and Linux (32768-60999) hand ports to outgoing connections
// and Listen(":0"), so no other program's connection holds it by chance
// while the app is closed. The first free one is the same on every launch.
const (
	firstPort  Port = 23600
	firstPorts      = 100
)

// pickPort keeps the remembered port while it is free, else takes the first
// free one of firstPorts, else one the kernel hands out. Another program may
// still take it before the child binds; the child then exits with serve's
// message and Retry picks again.
func pickPort(remembered Port) (Port, error) {
	if remembered != 0 && remembered != cliPort && free(remembered) {
		return remembered, nil
	}
	for p := firstPort; p < firstPort+firstPorts; p++ {
		if free(p) {
			return p, nil
		}
	}
	for range 8 {
		l, err := net.Listen("tcp4", "127.0.0.1:0")
		if err != nil {
			return 0, err
		}
		p := Port(l.Addr().(*net.TCPAddr).Port)
		l.Close()
		if p != cliPort && free(p) {
			return p, nil
		}
	}
	return 0, errors.New("no free loopback port")
}
