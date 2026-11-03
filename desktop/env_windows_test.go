//go:build windows

package main

import "testing"

// The CLI joins MARKITAI_HOME to the folder it starts in as Rust's PathBuf
// does: a path from the root without a drive stays on that folder's drive.
func TestMarkitaiHomeOnTheUsersDrive(t *testing.T) {
	cases := map[string]string{
		`\srv\markitai`:        `C:\srv\markitai`,
		`/srv/markitai/`:       `C:\srv\markitai`,
		`D:\data\markitai`:     `D:\data\markitai`,
		`\\nas\share\markitai`: `\\nas\share\markitai`,
		`state`:                `C:\Users\u\state`,
		`~\state`:              `C:\Users\u\state`,
		`~/state`:              `C:\Users\u\state`,
	}
	for home, want := range cases {
		if got := markitaiHome([]string{"MARKITAI_HOME=" + home}, `C:\Users\u`); got != want {
			t.Errorf("markitaiHome(%s) = %q; want %q", home, got, want)
		}
	}
}
