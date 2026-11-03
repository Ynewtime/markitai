package main

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPickPortKeepsAFreeRememberedPort(t *testing.T) {
	remembered, err := pickPort(0)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := pickPort(remembered); err != nil || got != remembered {
		t.Errorf("pickPort(%d) = %d, %v", remembered, got, err)
	}
}

// The ephemeral ranges the systems hand out for outgoing connections and
// Listen(":0"): macOS and Windows 49152-65535, Linux 32768-60999.
func ephemeral(p Port) bool { return p >= 32768 }

func TestPickPortStartsOutsideEphemeralRanges(t *testing.T) {
	p, err := pickPort(0)
	if err != nil || p == 0 || ephemeral(p) || p == cliPort {
		t.Errorf("pickPort(0) = %d, %v; want a port below 32768", p, err)
	}
	again, err := pickPort(0)
	if err != nil || again != p {
		t.Errorf("a second first launch takes %d, the first %d", again, p)
	}
}

// Another program listening on loopback, or on every address as a dev
// server on 0.0.0.0 or [::] does, keeps its port.
func TestPickPortLeavesATakenPort(t *testing.T) {
	for _, a := range []struct{ network, address string }{
		{"tcp4", "127.0.0.1:0"}, {"tcp4", "0.0.0.0:0"}, {"tcp", ":0"}, {"tcp6", "[::]:0"},
	} {
		l, err := net.Listen(a.network, a.address)
		if err != nil {
			t.Logf("%s %s: %v", a.network, a.address, err)
			continue
		}
		taken := Port(l.Addr().(*net.TCPAddr).Port)
		got, err := pickPort(taken)
		if err != nil || got == taken || got == 0 || got == cliPort {
			t.Errorf("with %s %s listening on %d, pickPort(%d) = %d, %v", a.network, a.address, taken, taken, got, err)
		}
		l.Close()
	}
}

func TestPickPortTakesTheNextOfTheFirstPorts(t *testing.T) {
	first, err := pickPort(0)
	if err != nil {
		t.Fatal(err)
	}
	l, err := net.Listen("tcp", "127.0.0.1:"+first.String())
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if got, err := pickPort(0); err != nil || got == first || ephemeral(got) {
		t.Errorf("with %d taken, pickPort(0) = %d, %v; want another port below 32768", first, got, err)
	}
}

func TestPickPortNeverTakesTheCLIPort(t *testing.T) {
	if got, err := pickPort(cliPort); err != nil || got == cliPort || got == 0 {
		t.Errorf("pickPort(3600) = %d, %v", got, err)
	}
}

func TestMemoryRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "data", "desktop.json")
	if m := loadMemory(path); m != (Memory{}) {
		t.Errorf("a missing file loads %v", m)
	}
	if err := (Memory{Port: 51234}).save(path); err != nil {
		t.Fatal(err)
	}
	if m := loadMemory(path); m.Port != 51234 {
		t.Errorf("loadMemory = %v", m)
	}
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if m := loadMemory(path); m != (Memory{}) {
		t.Errorf("a broken file loads %v", m)
	}
}

func TestTokenStaysOutOfLogs(t *testing.T) {
	token := newToken()
	if len(token) != 64 {
		t.Errorf("token of %d characters", len(token))
	}
	c := child{PID: 1, Port: 2, Token: token}
	for _, printed := range []string{fmt.Sprint(token), fmt.Sprintf("%v %+v", c, Ready{Child: c})} {
		if strings.Contains(printed, string(token)) {
			t.Errorf("printed the token: %s", printed)
		}
	}
}
