package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strconv"
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

// Memory is what the app keeps between launches in PathUserData/desktop.json.
// The workbench keeps its theme, language and options in localStorage, which
// belongs to an origin: the same port keeps them.
type Memory struct {
	Port Port `json:"port,omitempty"`
}

// loadMemory returns the zero Memory for a missing or unreadable file; the
// cost is one new port.
func loadMemory(path string) Memory {
	b, err := os.ReadFile(path)
	if err != nil {
		return Memory{}
	}
	var m Memory
	if json.Unmarshal(b, &m) != nil {
		return Memory{}
	}
	return m
}

func (m Memory) save(path string) error {
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
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

// free reports whether no program listens on p, on loopback or on every
// address. A listener on 0.0.0.0 or [::] does not keep serve from binding
// 127.0.0.1 on macOS (SO_REUSEADDR), where serve would then take the
// connections other programs make to that program on 127.0.0.1.
func free(p Port) bool {
	for _, a := range []struct{ network, host string }{{"tcp4", "127.0.0.1"}, {"tcp4", "0.0.0.0"}, {"tcp", ""}} {
		l, err := net.Listen(a.network, net.JoinHostPort(a.host, p.String()))
		if err != nil {
			return false
		}
		l.Close()
	}
	return true
}
