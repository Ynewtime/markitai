package main

import "os"

// appEnv holds the variables the app sets for its own toolkit, each with
// the value the user's environment gave it, nil for none: the programs
// the app opens get those back (section 10.5).
var appEnv = map[string]*string{}

// setAppEnv sets key for the app's own use, before anything else runs.
func setAppEnv(key, value string) {
	noteUserEnv(key)
	os.Setenv(key, value)
}

// noteUserEnv keeps the user's value of key, which the toolkit is about
// to set for itself, for the programs the app opens.
func noteUserEnv(key string) {
	if _, ok := appEnv[key]; ok {
		return
	}
	if v, ok := os.LookupEnv(key); ok {
		appEnv[key] = &v
	} else {
		appEnv[key] = nil
	}
}
