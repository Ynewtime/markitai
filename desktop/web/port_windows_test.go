//go:build windows

package main

import (
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// A program that has web.json open without sharing its deletion, as a
// virus scan may, keeps it from being replaced until it closes it.
func TestMemoryReplacesAFileAnotherProgramHolds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "web.json")
	m := openMemory(path)
	if err := m.Update(func(m *Memory) { m.Port = 51234 }); err != nil {
		t.Fatal(err)
	}
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		t.Fatal(err)
	}
	h, err := syscall.CreateFile(name, syscall.GENERIC_READ, syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING,
		syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	released := make(chan struct{})
	go func() {
		time.Sleep(300 * time.Millisecond)
		syscall.CloseHandle(h)
		close(released)
	}()
	if err := m.Update(func(m *Memory) { m.Theme = "dark" }); err != nil {
		t.Fatalf("with web.json held for 300 ms: %v", err)
	}
	<-released
	if got := openMemory(path).Get(); got != (Memory{Port: 51234, Theme: "dark"}) {
		t.Errorf("the file holds %+v", got)
	}
}
