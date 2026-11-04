package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestLaunchAsUser starts the programs the app opens with the user's own
// values of the variables set for the app's toolkit (section 10.5), by the
// app or by mygo, and leaves the app's as they are. A handler of a scheme of the test's own
// records the environment it got.
func TestLaunchAsUser(t *testing.T) {
	if err := loadGLib(); err != nil {
		t.Skip(err)
	}
	dir := t.TempDir()
	t.Setenv("XDG_DATA_HOME", filepath.Join(dir, "data"))
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(dir, "config"))
	out := filepath.Join(dir, "env.txt")
	record := filepath.Join(dir, "record.sh")
	os.WriteFile(record, []byte("#!/bin/sh\nenv > "+out+".part && mv "+out+".part "+out+"\n"), 0o755)
	apps := filepath.Join(dir, "data", "applications")
	os.MkdirAll(apps, 0o755)
	os.WriteFile(filepath.Join(apps, "markitai-record.desktop"), []byte("[Desktop Entry]\nType=Application\nName=Record\nNoDisplay=true\n"+
		"Exec="+record+" %u\nMimeType=x-scheme-handler/markitai-test;\n"), 0o644)
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	os.WriteFile(filepath.Join(dir, "config", "mimeapps.list"), []byte("[Default Applications]\nx-scheme-handler/markitai-test=markitai-record.desktop\n"), 0o644)

	saved := appEnv
	t.Cleanup(func() { appEnv = saved })
	appEnv = map[string]*string{}
	t.Setenv("LANGUAGE", "fr")
	os.Unsetenv("GTK_USE_PORTAL")
	t.Cleanup(func() { os.Unsetenv("GTK_USE_PORTAL") })
	setAppEnv("LANGUAGE", "zh_CN")
	setAppEnv("GTK_USE_PORTAL", "1")
	// mygo's, set as it starts (App.Run).
	t.Setenv("WEBKIT_DISABLE_DMABUF_RENDERER", "")
	os.Unsetenv("WEBKIT_DISABLE_DMABUF_RENDERER")
	noteUserEnv("WEBKIT_DISABLE_DMABUF_RENDERER")
	os.Setenv("WEBKIT_DISABLE_DMABUF_RENDERER", "1")

	if err := launchAsUser("markitai-test:hello"); err != nil {
		t.Fatal(err)
	}
	var env string
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if b, err := os.ReadFile(out); err == nil {
			env = string(b)
			break
		}
	}
	if env == "" {
		t.Fatal("the handler did not run")
	}
	lines := strings.Split(env, "\n")
	has := func(line string) bool {
		for _, l := range lines {
			if l == line {
				return true
			}
		}
		return false
	}
	if !has("LANGUAGE=fr") || strings.Contains(env, "GTK_USE_PORTAL=") || strings.Contains(env, "WEBKIT_DISABLE_DMABUF_RENDERER=") {
		t.Errorf("the program got LANGUAGE=fr %v, GTK_USE_PORTAL %v, WEBKIT_DISABLE_DMABUF_RENDERER %v", has("LANGUAGE=fr"),
			strings.Contains(env, "GTK_USE_PORTAL="), strings.Contains(env, "WEBKIT_DISABLE_DMABUF_RENDERER="))
	}
	if os.Getenv("LANGUAGE") != "zh_CN" || os.Getenv("GTK_USE_PORTAL") != "1" {
		t.Errorf("the app's own: LANGUAGE %q, GTK_USE_PORTAL %q", os.Getenv("LANGUAGE"), os.Getenv("GTK_USE_PORTAL"))
	}
}
