//go:build windows

package main

import (
	"encoding/binary"
	"syscall"
	"unsafe"
)

// free reports whether no program listens on p, on any address, and serve
// could bind 127.0.0.1 there. It reads the system's tables of listeners
// rather than listening to find out, as on macOS and Linux: Windows lets a
// socket listen on every address of a port that another holds for IPv6
// alone, and a listen on every address is what makes Windows Defender
// Firewall ask the user to let the app accept connections from the network.
func free(p Port) bool {
	return !listened(p) && bindable(p)
}

// bindable reports whether no socket holds 127.0.0.1:p, by binding it
// without listening.
func bindable(p Port) bool {
	s, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
	if err != nil {
		return false
	}
	defer syscall.Closesocket(s)
	return syscall.Bind(s, &syscall.SockaddrInet4{Port: int(p), Addr: [4]byte{127, 0, 0, 1}}) == nil
}

// iphlpapi.dll is one of Go's own system DLLs, which syscall loads from
// System32 alone.
var getExtendedTCPTable = syscall.NewLazyDLL("iphlpapi.dll").NewProc("GetExtendedTcpTable")

// listened reports whether a TCP listener, IPv4 or IPv6, has the local port
// p. Rows of MIB_TCPROW_OWNER_PID and MIB_TCP6ROW_OWNER_PID hold the port in
// network byte order, at offsets 8 and 20. A table it cannot read counts as
// empty, leaving bindable to judge.
func listened(p Port) bool {
	for _, t := range []struct{ family, row, port int }{{syscall.AF_INET, 24, 8}, {syscall.AF_INET6, 56, 20}} {
		table := listeners(t.family)
		if len(table) < 4 {
			continue
		}
		for i := range int(binary.LittleEndian.Uint32(table)) {
			at := 4 + i*t.row + t.port
			if at+2 <= len(table) && Port(binary.BigEndian.Uint16(table[at:])) == p {
				return true
			}
		}
	}
	return false
}

// listeners is the system's table of TCP listeners of family
// (TCP_TABLE_OWNER_PID_LISTENER), or nil.
func listeners(family int) []byte {
	const ownerPIDListener = 3
	if getExtendedTCPTable.Find() != nil {
		return nil
	}
	size := uint32(16 << 10)
	for range 4 {
		table := make([]byte, size)
		r, _, _ := getExtendedTCPTable.Call(uintptr(unsafe.Pointer(&table[0])), uintptr(unsafe.Pointer(&size)), 0,
			uintptr(family), ownerPIDListener, 0)
		switch syscall.Errno(r) {
		case 0:
			return table
		case syscall.ERROR_INSUFFICIENT_BUFFER:
			continue // size is now the table's
		}
		return nil
	}
	return nil
}
