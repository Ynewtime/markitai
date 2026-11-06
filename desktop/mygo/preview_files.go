package main

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"os"
	"strings"
	"sync"
)

// previewScheme is the scheme the macOS preview reads a result folder at.
const previewScheme = "markitai-preview"

// previewFolders serves result folders to the macOS preview, each at a host
// of its own (section 8.1). Only a file load WebKit can show gives a page
// read access to a folder, and on macOS 15 it shows no Markdown file: it
// downloads it instead. A folder the app serves needs no such load.
type previewFolders struct {
	mu      sync.Mutex
	folders map[string]string
}

// url is the URL of folder, the same each time, with a trailing slash.
func (f *previewFolders) url(folder string) string {
	sum := sha256.Sum256([]byte(folder))
	host := hex.EncodeToString(sum[:8])
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.folders == nil {
		f.folders = map[string]string{}
	}
	f.folders[host] = folder
	return previewScheme + "://" + host + "/"
}

// ServeHTTP answers with the file the request names inside its folder.
// os.Root refuses a path or a link that leaves the folder.
func (f *previewFolders) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	folder, ok := f.folders[r.URL.Host]
	f.mu.Unlock()
	if !ok {
		http.NotFound(w, r)
		return
	}
	root, err := os.OpenRoot(folder)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer root.Close()
	name := strings.TrimPrefix(r.URL.Path, "/")
	file, err := root.Open(name)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.IsDir() {
		http.NotFound(w, r)
		return
	}
	http.ServeContent(w, r, name, info.ModTime(), file)
}
