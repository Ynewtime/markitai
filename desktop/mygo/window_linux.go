package main

import "github.com/egoist/mygo"

// contentMinimum has nothing to add on Linux: mygo gives GTK MinWidth and
// MinHeight as the content's least size (section 6.10).
func contentMinimum(*mygo.Window) {}
