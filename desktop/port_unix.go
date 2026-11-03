//go:build unix

package main

import "net"

// free reports whether no program listens on p, on loopback or on every
// address. A listener on 0.0.0.0 or [::] does not keep serve from binding
// 127.0.0.1 on macOS (SO_REUSEADDR), where serve would then take the
// connections other programs make to that program on 127.0.0.1.
func free(p Port) bool {
	for _, a := range []struct{ network, host string }{{"tcp4", "127.0.0.1"}, {"tcp4", "0.0.0.0"}, {"tcp", ""}} {
		l, err := net.Listen(a.network, net.JoinHostPort(a.host, p.String()))
		if err != nil {
			return false
		}
		l.Close()
	}
	return true
}
