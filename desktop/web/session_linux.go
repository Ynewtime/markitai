package main

import (
	"os"
	"os/signal"
	"syscall"
)

// holdTermination keeps SIGTERM from mygo, which would quit as Cmd+Q does,
// asking about unfinished conversions: a session that ends sends SIGTERM
// and follows with SIGKILL, and asks nobody. Before App.Run, which leaves
// a signal ignored to the app.
func holdTermination() { signal.Ignore(syscall.SIGTERM) }

// takeTermination calls end on the app's goroutine of signals for each
// SIGTERM. Once the app runs, before it starts a child, which would
// inherit SIGTERM ignored.
func takeTermination(end func()) {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGTERM)
	go func() {
		for range ch {
			end()
		}
	}()
}
