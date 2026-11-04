// Package core calls the markitai Rust core in process through
// libmarkitai_app, the private C interface of desktop/DESIGN.md 3.1 and
// 3.3. It loads the library at run time with purego: no cgo.
package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ABIVersion is the MARKITAI_APP_ABI_VERSION this app was written against.
const ABIVersion = 1

// ErrABI reports a library whose ABI version is not ABIVersion.
var ErrABI = errors.New("core: ABI version mismatch")

// buffer is MarkitaiAppBuffer: len bytes of UTF-8 JSON the library owns.
type buffer struct {
	data *byte
	len  uintptr
}

// Lib is a loaded libmarkitai_app. Its methods are safe for concurrent use:
// each call is independent, as the library's are.
type Lib struct {
	Path string

	abiVersion         func() uint32
	version            func() string
	prepareEnvironment func(*byte, uintptr) buffer
	convertJSON        func(*byte, uintptr) buffer
	previewHTML        func(*byte, uintptr) buffer
	claimOutputFolder  func(*byte, uintptr) buffer
	configFile         func() buffer
	bufferFree         func(*buffer)
}

// FileName is the library's file name on this platform.
func FileName() string {
	switch runtime.GOOS {
	case "windows":
		return "markitai_app.dll"
	case "darwin":
		return "libmarkitai_app.dylib"
	default:
		return "libmarkitai_app.so"
	}
}

// Find returns the library's path: MARKITAI_APP_LIB when set, else the
// file in resources, or in its platform directory as the project's
// resources hold it under go run and go test.
func Find(resources string) string {
	if p := os.Getenv("MARKITAI_APP_LIB"); p != "" {
		return p
	}
	for _, p := range []string{
		filepath.Join(resources, FileName()),
		filepath.Join(resources, runtime.GOOS+"-"+runtime.GOARCH, FileName()),
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return filepath.Join(resources, FileName())
}

var loaded struct {
	sync.Mutex
	libs map[string]*Lib
}

// Load opens the library at path once per process and checks its ABI
// version: a mismatch returns the Lib with ErrABI, which the app shows as
// the fatal.abi message.
func Load(path string) (*Lib, error) {
	loaded.Lock()
	defer loaded.Unlock()
	if l := loaded.libs[path]; l != nil {
		return l, nil
	}
	h, err := open(path)
	if err != nil {
		return nil, fmt.Errorf("core: open %s: %w", path, err)
	}
	l := &Lib{Path: path}
	for _, f := range []struct {
		name string
		fn   any
	}{
		{"markitai_app_abi_version", &l.abiVersion},
		{"markitai_app_version", &l.version},
		{"markitai_app_prepare_environment", &l.prepareEnvironment},
		{"markitai_app_convert_json", &l.convertJSON},
		{"markitai_app_preview_html", &l.previewHTML},
		{"markitai_app_claim_output_folder", &l.claimOutputFolder},
		{"markitai_app_config_file", &l.configFile},
		{"markitai_app_buffer_free", &l.bufferFree},
	} {
		addr, err := sym(h, f.name)
		if err != nil || addr == 0 {
			return nil, fmt.Errorf("core: %s: missing %s: %v", path, f.name, err)
		}
		purego.RegisterFunc(f.fn, addr)
	}
	if v := l.abiVersion(); v != ABIVersion {
		return l, fmt.Errorf("%w: library %d, app %d", ErrABI, v, ABIVersion)
	}
	if loaded.libs == nil {
		loaded.libs = map[string]*Lib{}
	}
	loaded.libs[path] = l
	return l, nil
}

// Version is the markitai version the library was built from.
func (l *Lib) Version() string { return l.version() }

// Error is an error envelope's error, or one the app makes with the same
// codes (output_unavailable, skipped).
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// envelope is every call's response.
type envelope struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result"`
	Error  *Error          `json:"error"`
}

// take copies a response out of the library's memory, frees it, and
// parses the envelope: the result's JSON, or its error.
func (l *Lib) take(b buffer) (json.RawMessage, error) {
	raw := []byte(nil)
	if b.data != nil {
		raw = append(raw, unsafe.Slice(b.data, b.len)...)
	}
	l.bufferFree(&b)
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, &Error{Code: "invalid_json", Message: err.Error()}
	}
	if !env.OK {
		if env.Error == nil {
			return nil, &Error{Code: "invalid_json", Message: "an error envelope without an error"}
		}
		return nil, env.Error
	}
	return env.Result, nil
}

func (l *Lib) call(fn func(*byte, uintptr) buffer, request any) (json.RawMessage, error) {
	req, err := json.Marshal(request)
	if err != nil {
		return nil, &Error{Code: "invalid_input", Message: err.Error()}
	}
	b := fn(unsafe.SliceData(req), uintptr(len(req)))
	runtime.KeepAlive(req)
	return l.take(b)
}

// Environment is prepare_environment's result.
type Environment struct {
	LoginEnvironment string  `json:"login_environment"`
	Detail           *string `json:"detail"`
}

// PrepareEnvironment creates dataDir, makes it the working directory and,
// launched without a terminal on macOS and Linux, imports the login
// shell's environment. Call it once, first.
func (l *Lib) PrepareEnvironment(dataDir string) (Environment, error) {
	var env Environment
	raw, err := l.call(l.prepareEnvironment, map[string]string{"data_dir": dataDir})
	if err == nil {
		err = json.Unmarshal(raw, &env)
	}
	return env, err
}

// ConvertJSON runs one conversion and blocks until it ends: call it off the
// UI thread. request is markitai_convert_json's request; the result is its
// envelope's result.
func (l *Lib) ConvertJSON(request any) (json.RawMessage, error) {
	return l.call(l.convertJSON, request)
}

// PreviewHTML renders Markdown to the preview document: theme is "light"
// or "dark", and images load only from inside baseDir.
func (l *Lib) PreviewHTML(markdown, theme string, baseDir *string) (string, error) {
	raw, err := l.call(l.previewHTML, map[string]any{"markdown": markdown, "theme": theme, "base_dir": baseDir})
	if err != nil {
		return "", err
	}
	var out struct {
		HTML string `json:"html"`
	}
	err = json.Unmarshal(raw, &out)
	return out.HTML, err
}

// Folder is a conversion's output folder, claimed and created.
type Folder struct {
	Path string `json:"path"`
	Name string `json:"name"`
}

// ClaimOutputFolder creates the folder a conversion of source writes into,
// under parent.
func (l *Lib) ClaimOutputFolder(parent, source string) (Folder, error) {
	var f Folder
	raw, err := l.call(l.claimOutputFolder, map[string]string{"parent": parent, "source": source})
	if err == nil {
		err = json.Unmarshal(raw, &f)
	}
	return f, err
}

// ConfigFile is the markitai configuration file the core reads.
type ConfigFile struct {
	Path   string `json:"path"`
	Exists bool   `json:"exists"`
}

// ConfigFile reports the configuration file the core reads and whether it
// exists now.
func (l *Lib) ConfigFile() (ConfigFile, error) {
	var f ConfigFile
	raw, err := l.take(l.configFile())
	if err == nil {
		err = json.Unmarshal(raw, &f)
	}
	return f, err
}
