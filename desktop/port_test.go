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

func TestPickPortLeavesATakenPort(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	taken := Port(l.Addr().(*net.TCPAddr).Port)
	got, err := pickPort(taken)
	if err != nil || got == taken || got == 0 || got == cliPort {
		t.Errorf("pickPort(%d) = %d, %v", taken, got, err)
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
