package main

import (
	"os"
	"path/filepath"
	"strings"
)

// appID is Markitai Mygo's identifier (section 2).
const appID = "dev.markitai.mygo"

// mygoFolders are section 2.1's data and cache folders and settings file
// on goos, for a user whose home folder is home.
func mygoFolders(goos, home string, getenv func(string) string) (data, cache, settings string) {
	switch goos {
	case "darwin":
		data = filepath.Join(home, "Library", "Application Support", appID)
		return data, filepath.Join(home, "Library", "Caches", appID), filepath.Join(data, "settings.json")
	case "windows":
		local := getenv("LOCALAPPDATA")
		if local == "" {
			local = filepath.Join(home, "AppData", "Local")
		}
		data = filepath.Join(local, "Markitai Mygo")
		return data, filepath.Join(data, "Cache"), filepath.Join(data, "settings.json")
	}
	xdg := func(key, fallback string) string {
		if dir := getenv(key); filepath.IsAbs(dir) {
			return dir
		}
		return filepath.Join(home, fallback)
	}
	return filepath.Join(xdg("XDG_DATA_HOME", filepath.Join(".local", "share")), appID),
		filepath.Join(xdg("XDG_CACHE_HOME", ".cache"), appID),
		filepath.Join(xdg("XDG_CONFIG_HOME", ".config"), appID, "settings.json")
}

// fileArgs keeps the command line's arguments that are existing paths,
// relative ones read from wd. macOS adds its own, as -psn_ and
// -AppleLanguages with a value, which are not.
func fileArgs(args []string, wd string) []string {
	var paths []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-") {
			if strings.HasPrefix(arg, "-Apple") || strings.HasPrefix(arg, "-NS") {
				i++
			}
			continue
		}
		if !filepath.IsAbs(arg) && wd != "" {
			arg = filepath.Join(wd, arg)
		}
		if _, err := os.Stat(arg); err == nil {
			paths = append(paths, arg)
		}
	}
	return paths
}
