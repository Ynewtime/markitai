package markitai

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNativeConversion(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MARKITAI_HOME", filepath.Join(dir, "home"))
	path := filepath.Join(dir, "文档.md")
	if err := os.WriteFile(path, []byte("# Go\n\n你好 🌍\n"), 0600); err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	for i := 0; i < 24; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			output, err := Convert(path, &Options{Config: map[string]any{}, LLM: Bool(false)})
			if err != nil {
				t.Error(err)
				return
			}
			if !strings.Contains(output.Markdown, "你好 🌍") || output.OutputPath != nil {
				t.Errorf("unexpected in-memory result: %+v", output)
			}
		}()
	}
	group.Wait()
	output, err := Convert(path, &Options{Config: map[string]any{}, OutputDir: filepath.Join(dir, "out"), LLM: Bool(false)})
	if err != nil {
		t.Fatal(err)
	}
	if output.OutputPath == nil {
		t.Fatal("missing output path")
	}
	if _, err := os.Stat(*output.OutputPath); err != nil {
		t.Fatal(err)
	}
}

func TestErrorsAndMalformedJSON(t *testing.T) {
	t.Setenv("MARKITAI_HOME", t.TempDir())
	_, err := Convert(filepath.Join(t.TempDir(), "missing.md"), &Options{Config: map[string]any{}, LLM: Bool(false)})
	var conversionError *ConversionError
	if !errors.As(err, &conversionError) || conversionError.Code == "" {
		t.Fatalf("expected structured conversion error, got %v", err)
	}
	for _, request := range [][]byte{nil, []byte("{"), {0xff}} {
		response, err := ConvertJSON(request)
		if err != nil {
			t.Fatal(err)
		}
		var envelope struct {
			OK bool `json:"ok"`
		}
		if err := json.Unmarshal(response, &envelope); err != nil || envelope.OK {
			t.Fatalf("invalid error envelope: %s (%v)", response, err)
		}
	}
	if Version() == "" {
		t.Fatal("empty native version")
	}
}

func TestNativeFetchCache(t *testing.T) {
	t.Setenv("MARKITAI_HOME", t.TempDir())
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html><article><h1>Native page</h1><p>Go HTTP 世界.</p></article></html>"))
	}))
	defer server.Close()
	options := &Options{Config: map[string]any{}, LLM: Bool(false)}
	first, err := Convert(server.URL, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Convert(server.URL, options)
	if err != nil {
		t.Fatal(err)
	}
	if first.Markdown != second.Markdown || !strings.Contains(second.Markdown, "Go HTTP 世界.") {
		t.Fatal("cached Markdown differs from fetched result")
	}
	if requests.Load() != 1 {
		t.Fatalf("expected one HTTP request, got %d", requests.Load())
	}
	options.Config["cache"] = map[string]any{"enabled": false}
	if _, err := Convert(server.URL, options); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 {
		t.Fatalf("disabled cache did not refetch: %d", requests.Load())
	}
}

func TestLegacyErrorUsageIsOptional(t *testing.T) {
	legacy := &ConversionError{Code: "old_code", Message: "old message"}
	if legacy.Error() != "old message" || legacy.Usage != nil {
		t.Fatalf("legacy error constructor changed: %+v", legacy)
	}
	for _, raw := range []string{
		`{"code":"conversion_error","message":"original message"}`,
		`{"code":"conversion_error","message":"original message","usage":null}`,
	} {
		var failure ConversionError
		if err := json.Unmarshal([]byte(raw), &failure); err != nil {
			t.Fatal(err)
		}
		if failure.Error() != "original message" || failure.Code != "conversion_error" || failure.Usage != nil {
			t.Fatalf("older producer contract changed: %+v", failure)
		}
	}
	var recorded ConversionError
	if err := json.Unmarshal([]byte(`{"code":"conversion_error","message":"original message","usage":{"cost_usd":0,"requests":1,"input_tokens":0,"output_tokens":0,"by_model":{"fixture":{"requests":1}}}}`), &recorded); err != nil {
		t.Fatal(err)
	}
	if recorded.Usage == nil || recorded.Usage.Requests != 1 || recorded.Usage.InputTokens != 0 || recorded.Usage.ByModel["fixture"]["requests"] != float64(1) {
		t.Fatalf("known zero-token response was lost: %+v", recorded)
	}
}

func TestPaidNativeFailuresKeepDocumentScopesSeparate(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MARKITAI_HOME", filepath.Join(dir, "state"))
	t.Setenv("NO_PROXY", "127.0.0.1,localhost")
	files := map[string]string{}
	for _, name := range []string{"TERMALPHA", "TERMBETA", "TERMZERO"} {
		path := filepath.Join(dir, name+".md")
		if err := os.WriteFile(path, []byte("# "+name+"\n\nComplete independent source document "+name+".\n"), 0600); err != nil {
			t.Fatal(err)
		}
		files[name] = path
	}
	var mu sync.Mutex
	requests := map[string]int{}
	var entered atomic.Int32
	gate := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(io.LimitReader(r.Body, 1024*1024+1))
		if err != nil || len(raw) > 1024*1024 {
			t.Errorf("invalid bounded request: %v", err)
			http.Error(w, "bad fixture request", http.StatusBadRequest)
			return
		}
		name := ""
		for candidate := range files {
			if strings.Contains(string(raw), candidate) {
				name = candidate
				break
			}
		}
		if name == "" {
			t.Error("request lost the independent source marker")
			http.Error(w, "missing fixture marker", http.StatusBadRequest)
			return
		}
		mu.Lock()
		requests[name]++
		mu.Unlock()
		if name != "TERMZERO" {
			if entered.Add(1) == 2 {
				close(gate)
			}
			select {
			case <-gate:
			case <-time.After(10 * time.Second):
				t.Error("both document requests did not reach the server together")
			case <-r.Context().Done():
				return
			}
		}
		tokens := map[string][2]int{"TERMALPHA": {11, 3}, "TERMBETA": {29, 7}, "TERMZERO": {0, 0}}[name]
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		if err := json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "PRIVATE RESPONSE SECRET"}, "model": name,
			"usage": map[string]any{"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
		}); err != nil {
			t.Error(err)
		}
	}))
	defer server.Close()
	config := map[string]any{
		"cache":   map[string]any{"enabled": false, "global_dir": filepath.Join(dir, "cache")},
		"prompts": map[string]any{"dir": filepath.Join(dir, "prompts")},
		"history": map[string]any{"record": false}, "ocr": map[string]any{"enabled": false},
		"image": map[string]any{"alt_enabled": false, "desc_enabled": false},
		"llm": map[string]any{"enabled": true, "on_failure": "fail", "max_requests_per_document": 1,
			"router_settings": map[string]any{"num_retries": 0, "timeout": 15},
			"model_list": []any{map[string]any{"model_name": "fixture", "litellm_params": map[string]any{
				"model": "openai/fixture", "api_key": "synthetic-key", "api_base": server.URL + "/v1",
			}}},
		},
	}
	type answer struct {
		name string
		err  error
	}
	failures := make(chan answer, 2)
	for _, name := range []string{"TERMALPHA", "TERMBETA"} {
		go func(name string) {
			_, err := Convert(files[name], &Options{Config: config})
			failures <- answer{name, err}
		}(name)
	}
	results := []answer{<-failures, <-failures}
	_, zero := Convert(files["TERMZERO"], &Options{Config: config})
	results = append(results, answer{"TERMZERO", zero})
	for _, result := range results {
		var failure *ConversionError
		if !errors.As(result.err, &failure) {
			t.Fatalf("%s: existing ConversionError category lost: %v", result.name, result.err)
		}
		if failure.Code != "conversion_error" || !strings.Contains(failure.Message, "HTTP 401") || strings.Contains(failure.Message, "PRIVATE RESPONSE SECRET") {
			t.Fatalf("%s: unexpected public diagnostic: %+v", result.name, failure)
		}
		tokens := map[string][2]uint64{"TERMALPHA": {11, 3}, "TERMBETA": {29, 7}, "TERMZERO": {0, 0}}[result.name]
		if failure.Usage == nil || failure.Usage.Requests != 1 || failure.Usage.InputTokens != tokens[0] || failure.Usage.OutputTokens != tokens[1] || len(failure.Usage.ByModel) != 1 || failure.Usage.ByModel[result.name] == nil {
			t.Fatalf("%s: missing or mixed accounting: %+v", result.name, failure.Usage)
		}
		coverage := failure.Usage.ByModel[result.name]
		if coverage["priced_requests"] != float64(0) || coverage["unpriced_requests"] != float64(1) || coverage["cost_status"] != "unknown" || coverage["pricing_snapshot"] != nil || failure.Usage.CostUSD != 0 {
			t.Fatalf("%s: custom endpoint must retain unknown price coverage: %+v", result.name, failure.Usage)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 3 || requests["TERMALPHA"] != 1 || requests["TERMBETA"] != 1 || requests["TERMZERO"] != 1 {
		t.Fatalf("terminal errors retried unexpectedly: %+v", requests)
	}
	_, err := Convert(filepath.Join(dir, "missing.md"), &Options{Config: map[string]any{}, LLM: Bool(false)})
	var early *ConversionError
	if !errors.As(err, &early) || early.Usage != nil {
		t.Fatalf("early failure should not invent accounting: %v", err)
	}
}

var numbersFixtures = []struct{ name, hash string }{
	{"test-1.numbers", "b9e9772b2d2866c26d773fe46a173c373c7dc1dd3df6cefc7f0253b6ab50d4c3"},
	{"test-formats.numbers", "9b3ba4b52b2eb3ffd1e05602ab7da954abb2da7f37e68b147725d31777e1cda3"},
}

func numbersFixtureRoot(t *testing.T) string {
	t.Helper()
	root := os.Getenv("MARKITAI_TEST_NUMBERS_FIXTURES")
	if root == "" {
		root = "../../crates/markitai-core/src/formats/numbers/fixtures"
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		t.Fatal(err)
	}
	return absolute
}

func numbersBundle(t *testing.T, fixtureRoot, targetRoot, name, expectedHash string) (string, string) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(fixtureRoot, name))
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != expectedHash {
		t.Fatal("independent MIT Numbers fixture identity changed")
	}
	zipped := filepath.Join(targetRoot, name)
	if err := os.WriteFile(zipped, data, 0600); err != nil {
		t.Fatal(err)
	}
	bundle := filepath.Join(targetRoot, "目录-"+strings.ToUpper(name))
	if err := os.Mkdir(bundle, 0700); err != nil {
		t.Fatal(err)
	}
	archive, err := zip.OpenReader(zipped)
	if err != nil {
		t.Fatal(err)
	}
	defer archive.Close()
	seen := map[string]bool{}
	for _, entry := range archive.File {
		if !filepath.IsLocal(entry.Name) || strings.Contains(entry.Name, "\\") || seen[entry.Name] || entry.UncompressedSize64 > 1024*1024 {
			t.Fatal("unexpected pinned fixture entry")
		}
		seen[entry.Name] = true
		destination := filepath.Join(bundle, filepath.FromSlash(entry.Name))
		if entry.FileInfo().IsDir() {
			if err := os.MkdirAll(destination, 0700); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if !entry.Mode().IsRegular() {
			t.Fatal("pinned fixture entry is not regular")
		}
		input, err := entry.Open()
		if err != nil {
			t.Fatal(err)
		}
		body, readErr := io.ReadAll(io.LimitReader(input, 1024*1024+1))
		closeErr := input.Close()
		if readErr != nil || closeErr != nil || uint64(len(body)) != entry.UncompressedSize64 {
			t.Fatalf("invalid pinned ZIP entry: read=%v close=%v", readErr, closeErr)
		}
		if err := os.MkdirAll(filepath.Dir(destination), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(destination, body, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return zipped, bundle
}

func numbersOptions(root string) *Options {
	return &Options{LLM: Bool(false), OCR: Bool(false), Screenshot: Bool(false), Alt: Bool(false), Desc: Bool(false), Config: map[string]any{
		"llm": map[string]any{"enabled": false}, "ocr": map[string]any{"enabled": false},
		"screenshot": map[string]any{"enabled": false}, "cache": map[string]any{"enabled": false},
		"image":   map[string]any{"alt_enabled": false, "desc_enabled": false},
		"history": map[string]any{"record": false}, "prompts": map[string]any{"dir": filepath.Join(root, "prompts")},
	}}
}

func privateNumbersCwd(t *testing.T, directory string) {
	t.Helper()
	previous, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(directory); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previous); err != nil {
			t.Error(err)
		}
	})
	t.Setenv("MARKITAI_HOME", filepath.Join(directory, "state"))
}

func numbersJSON(t *testing.T, source string, options *Options) (bool, *ConversionOutput, *ConversionError) {
	t.Helper()
	request, err := json.Marshal(map[string]any{"source": source, "options": options})
	if err != nil {
		t.Fatal(err)
	}
	response, err := ConvertJSON(request)
	if err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		OK     bool              `json:"ok"`
		Result *ConversionOutput `json:"result"`
		Error  *ConversionError  `json:"error"`
	}
	if err := json.Unmarshal(response, &envelope); err != nil {
		t.Fatal(err)
	}
	return envelope.OK, envelope.Result, envelope.Error
}

func TestNumbersDirectoryPackagesMatchPinnedZIPAcrossTypedAndJSONCalls(t *testing.T) {
	fixtures := numbersFixtureRoot(t)
	dir := t.TempDir()
	originalHome := os.Getenv("HOME")
	privateNumbersCwd(t, dir)
	for _, fixture := range numbersFixtures {
		zipped, bundle := numbersBundle(t, fixtures, dir, fixture.name, fixture.hash)
		options := numbersOptions(dir)
		baseline, err := Convert(zipped, options)
		if err != nil {
			t.Fatal(err)
		}
		result, err := Convert(bundle, options)
		if err != nil {
			t.Fatal(err)
		}
		ok, raw, failure := numbersJSON(t, bundle, options)
		if !ok || raw == nil || failure != nil {
			t.Fatalf("directory JSON call failed: %+v", failure)
		}
		for _, output := range []*ConversionOutput{result, raw} {
			if baseline.Markdown == "" || output.Markdown != baseline.Markdown || output.Source != bundle || !reflect.DeepEqual(output.Warnings, baseline.Warnings) || output.OutputPath != nil || output.Usage.Requests != 0 || len(output.Assets) != 0 {
				t.Fatal("directory result differs from the independent ZIP fixture")
			}
		}
		options.OutputDir = filepath.Join(dir, "out-"+fixture.name)
		written, err := Convert(bundle, options)
		if err != nil {
			t.Fatal(err)
		}
		if written.OutputPath == nil || filepath.Base(*written.OutputPath) != filepath.Base(bundle)+".md" {
			t.Fatal("directory did not publish one correctly named Markdown file")
		}
		body, err := os.ReadFile(*written.OutputPath)
		if err != nil || !strings.HasSuffix(string(body), baseline.Markdown) {
			t.Fatalf("published body differs from ZIP: %v", err)
		}
	}
	if os.Getenv("HOME") != originalHome {
		t.Fatal("HOME changed during directory package conversion")
	}
}

func TestNumbersPackagesKeepOtherDirectoriesXMLAndVisualModesExplicit(t *testing.T) {
	fixtures := numbersFixtureRoot(t)
	dir := t.TempDir()
	privateNumbersCwd(t, dir)
	_, bundle := numbersBundle(t, fixtures, dir, numbersFixtures[0].name, numbersFixtures[0].hash)
	ordinary, legacy := filepath.Join(dir, "ordinary"), filepath.Join(dir, "old.numbers")
	for _, directory := range []string{ordinary, legacy} {
		if err := os.Mkdir(directory, 0700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(legacy, "index.xml"), []byte("<document><table>OLD_XML_MUST_NOT_BE_BATCHED</table></document>"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"directory", "xml", "ocr", "screenshot"} {
		options := numbersOptions(dir)
		source := bundle
		switch mode {
		case "directory":
			source = ordinary
		case "xml":
			source = legacy
		case "ocr":
			options.OCR = Bool(true)
		case "screenshot":
			options.Screenshot = Bool(true)
		}
		_, err := Convert(source, options)
		var typed *ConversionError
		if !errors.As(err, &typed) {
			t.Fatalf("%s: expected typed conversion failure: %v", mode, err)
		}
		ok, result, raw := numbersJSON(t, source, options)
		if ok || result != nil || raw == nil || raw.Code != typed.Code || raw.Message != typed.Message || raw.Usage != nil || typed.Usage != nil {
			t.Fatalf("%s: raw and typed failure contracts differ", mode)
		}
		if mode == "directory" {
			if typed.Code != "is_directory" {
				t.Fatalf("ordinary directory category changed: %s", typed.Code)
			}
		} else if mode == "ocr" || mode == "screenshot" {
			if typed.Code != "unsupported" || typed.Message != "Numbers complete-sheet screenshots are not supported by this LibreOffice adapter; native Numbers table reading remains available without screenshot or OCR options" {
				t.Fatalf("%s: Numbers visual-mode rejection changed: %+v", mode, typed)
			}
		} else if !strings.Contains(typed.Message, "Numbers") || typed.Code != "unsupported" {
			t.Fatalf("%s: not an explicit Numbers rejection: %+v", mode, typed)
		}
	}
}
