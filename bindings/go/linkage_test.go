package markitai

import (
	"go/build"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Every target links exactly one way: the development dynamic library by
// default and, with markitai_static, either the archive packaged for that
// target or the file that rejects the build. File-name suffixes constrain the
// files as well as their build lines, and the ios and android targets also
// satisfy the darwin and linux tags.
func TestEachTargetSelectsExactlyOneLinkage(t *testing.T) {
	names, err := filepath.Glob("link_*.go")
	if err != nil {
		t.Fatal(err)
	}
	packaged := map[string]string{
		"darwin/arm64": "link_static_darwin_arm64.go",
		"linux/amd64":  "link_static_linux_amd64.go",
	}
	for _, name := range packaged {
		if !slices.Contains(names, name) {
			t.Fatalf("missing linkage file %s among %v", name, names)
		}
	}
	targets := []string{"darwin/arm64", "darwin/amd64", "ios/arm64", "linux/amd64", "linux/arm64",
		"linux/386", "android/amd64", "android/arm64", "freebsd/amd64", "windows/amd64"}
	for _, target := range targets {
		goos, goarch, _ := strings.Cut(target, "/")
		for _, static := range []bool{false, true} {
			context := build.Default
			context.GOOS, context.GOARCH, context.CgoEnabled = goos, goarch, true
			context.BuildTags = nil
			want := "link_dynamic.go"
			if static {
				context.BuildTags = []string{"markitai_static"}
				want = "link_static_unsupported.go"
				if name, ok := packaged[target]; ok {
					want = name
				}
			}
			var selected []string
			for _, name := range names {
				match, err := context.MatchFile(".", name)
				if err != nil {
					t.Fatal(err)
				}
				if match {
					selected = append(selected, name)
				}
			}
			if len(selected) != 1 || selected[0] != want {
				t.Errorf("%s static=%v selects %v, want [%s]", target, static, selected, want)
			}
		}
	}
}
