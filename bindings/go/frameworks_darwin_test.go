//go:build darwin && !markitai_static

package markitai

import (
	"bytes"
	"debug/macho"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

// From macOS 15 (Darwin 24) dyld postpones the initialization of an image
// linked delay-initialized, and DYLD_PRINT_LIBRARIES reports each image it
// maps, postpones, and initializes later. The dynamic library links the media
// frameworks that way; the static archive leaves linkage to its consumer.

var mediaFrameworks = []string{"CoreFoundation", "Foundation", "CoreGraphics", "ImageIO", "Vision"}

const (
	frameworkRequest = "MARKITAI_TEST_FRAMEWORK_REQUEST"
	frameworkOutput  = "MARKITAI_TEST_FRAMEWORK_OUTPUT"
	frameworkMarker  = "markitai test: library loaded\n"
)

var dyldLine = regexp.MustCompile(`^dyld\[\d+\]: (?:<[0-9A-F-]+> (.+)|move (loaded to delayed|delayed to loaded): (.+))$`)

type dyldImages struct {
	paths                               map[string]string
	mapped, postponed, initializedLater map[string]bool
}

func parseDyld(trace string) dyldImages {
	images := dyldImages{map[string]string{}, map[string]bool{}, map[string]bool{}, map[string]bool{}}
	for _, line := range strings.Split(trace, "\n") {
		match := dyldLine.FindStringSubmatch(line)
		switch {
		case match == nil:
		case match[1] != "":
			images.paths[filepath.Base(match[1])] = match[1]
			images.mapped[filepath.Base(match[1])] = true
		case match[2] == "loaded to delayed":
			images.postponed[match[3]] = true
		default:
			images.initializedLater[match[3]] = true
		}
	}
	return images
}

// delayedDependencies lists a Mach-O image's dependencies whose
// dylib_use_command carries DYLIB_USE_DELAYED_INIT.
func delayedDependencies(t *testing.T, path string) map[string]bool {
	t.Helper()
	image, err := macho.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer image.Close()
	delayed := map[string]bool{}
	for _, load := range image.Loads {
		dylib, ok := load.(*macho.Dylib)
		raw := load.Raw()
		if ok && dylib.Time == 0x1a741800 && len(raw) >= 28 && image.ByteOrder.Uint32(raw[24:])&0x8 != 0 {
			delayed[filepath.Base(dylib.Name)] = true
		}
	}
	return delayed
}

func postponesImages(t *testing.T) bool {
	t.Helper()
	release, err := syscall.Sysctl("kern.osrelease")
	if err != nil {
		t.Fatal(err)
	}
	major, err := strconv.Atoi(strings.SplitN(release, ".", 2)[0])
	if err != nil {
		t.Fatal(err)
	}
	return major >= 24
}

// TestFrameworkHelperProcess converts one request as the child process of
// TestLoadingPostponesMediaFrameworksUntilAConversionNeedsThem.
func TestFrameworkHelperProcess(t *testing.T) {
	request := os.Getenv(frameworkRequest)
	if request == "" {
		t.Skip("runs only as the child of the framework test")
	}
	os.Stderr.WriteString(frameworkMarker)
	response, err := ConvertJSON([]byte(request))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv(frameworkOutput), response, 0600); err != nil {
		t.Fatal(err)
	}
}

// pagePDF is one page holding a filled rectangle and no text.
func pagePDF() []byte {
	content := "0 0.4 0.8 rg 20 20 160 60 re f"
	objects := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
		fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(content), content),
	}
	var pdf bytes.Buffer
	pdf.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objects))
	for index, body := range objects {
		offsets[index] = pdf.Len()
		fmt.Fprintf(&pdf, "%d 0 obj\n%s\nendobj\n", index+1, body)
	}
	xref := pdf.Len()
	fmt.Fprintf(&pdf, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
	for _, offset := range offsets {
		fmt.Fprintf(&pdf, "%010d 00000 n \n", offset)
	}
	fmt.Fprintf(&pdf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
	return pdf.Bytes()
}

func TestLoadingPostponesMediaFrameworksUntilAConversionNeedsThem(t *testing.T) {
	if !postponesImages(t) {
		t.Skip("dyld postpones delay-initialized images from macOS 15")
	}
	// A Go executable that links a media framework itself (crypto/x509 links
	// CoreFoundation) initializes it and its dependencies at launch, as Node
	// and Python do; only Vision is then left for the library to postpone.
	executable, err := macho.Open(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	imported, err := executable.ImportedLibraries()
	executable.Close()
	if err != nil {
		t.Fatal(err)
	}
	linksMedia := false
	for _, library := range imported {
		for _, name := range mediaFrameworks {
			linksMedia = linksMedia || filepath.Base(library) == name
		}
	}
	expected := mediaFrameworks
	if linksMedia {
		expected = []string{"Vision"}
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "page.pdf")
	if err := os.WriteFile(source, pagePDF(), 0600); err != nil {
		t.Fatal(err)
	}
	request, err := json.Marshal(map[string]any{"source": source, "options": map[string]any{
		"config": map[string]any{
			"llm": map[string]any{"enabled": false}, "cache": map[string]any{"enabled": false},
			"history": map[string]any{"record": false},
		},
		"output_dir": filepath.Join(dir, "out"), "llm": false, "ocr": false, "screenshot": true,
		"alt": false, "desc": false,
	}})
	if err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(dir, "response.json")
	child := exec.Command(os.Args[0], "-test.run=^TestFrameworkHelperProcess$", "-test.count=1")
	child.Env = append(os.Environ(), "DYLD_PRINT_LIBRARIES=1", "MARKITAI_HOME="+filepath.Join(dir, "home"),
		frameworkRequest+"="+string(request), frameworkOutput+"="+output)
	var stderr bytes.Buffer
	child.Stderr = &stderr
	if err := child.Run(); err != nil {
		t.Fatalf("%v: %s", err, stderr.String())
	}
	loading, converting, found := strings.Cut(stderr.String(), frameworkMarker)
	if !found {
		t.Fatalf("no marker separates loading from converting: %s", stderr.String())
	}
	loaded := parseDyld(loading)
	if len(loaded.mapped) == 0 {
		t.Skip("this process drops dyld's diagnostic variables")
	}
	library, found := loaded.paths["libmarkitai_ffi.dylib"]
	if !found {
		t.Fatal("the child did not load libmarkitai_ffi.dylib")
	}
	delayed := delayedDependencies(t, library)
	for _, name := range mediaFrameworks {
		if !delayed[name] {
			t.Errorf("%s does not link %s delay-initialized", library, name)
		}
	}
	for _, name := range expected {
		if !loaded.postponed[name] {
			t.Errorf("loading the library initialized %s", name)
		}
	}
	if len(loaded.initializedLater) != 0 {
		t.Errorf("loading the library initialized postponed images %v", loaded.initializedLater)
	}
	// Page rendering opens CoreGraphics on first use.
	body, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		OK     bool             `json:"ok"`
		Result ConversionOutput `json:"result"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || !envelope.OK || len(envelope.Result.Screenshots) != 1 {
		t.Fatalf("page rendering failed: %v %s", err, body)
	}
	image, err := os.ReadFile(envelope.Result.Screenshots[0])
	if err != nil || !bytes.HasPrefix(image, []byte{0xff, 0xd8, 0xff}) {
		t.Fatalf("screenshot is not a JPEG image: %v", err)
	}
	if !linksMedia && !parseDyld(converting).initializedLater["CoreGraphics"] {
		t.Error("page rendering did not initialize CoreGraphics")
	}
}
