package main

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// glib holds what the app calls in GLib and GIO. mygo loads the same
// libraries; loading them again shares them.
var glib struct {
	once sync.Once
	err  error

	errorFree        func(e *gError)
	variantNewString func(s string) unsafe.Pointer
	variantNewTuple  func(children *unsafe.Pointer, n uintptr) unsafe.Pointer
	variantUnref     func(v unsafe.Pointer)
	objectUnref      func(o unsafe.Pointer)

	busGetSync          func(busType int32, cancellable unsafe.Pointer, gerr **gError) unsafe.Pointer
	dbusCallSync        func(conn unsafe.Pointer, name, path, iface, method string, params, replyType unsafe.Pointer, flags, timeout int32, cancellable unsafe.Pointer, gerr **gError) unsafe.Pointer
	launchContextNew    func() unsafe.Pointer
	launchContextSetenv func(ctx unsafe.Pointer, key, value string)
	launchContextUnset  func(ctx unsafe.Pointer, key string)
	launchDefaultForURI func(uri string, ctx unsafe.Pointer, gerr **gError) int32
}

// gError is GLib's GError.
type gError struct {
	domain  uint32
	code    int32
	message *byte
}

// loadGLib loads glib's functions once, and says whether it could.
func loadGLib() error {
	glib.once.Do(func() {
		libs := map[string]uintptr{}
		for _, name := range []string{"libglib-2.0.so.0", "libgobject-2.0.so.0", "libgio-2.0.so.0"} {
			h, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err != nil {
				glib.err = err
				return
			}
			libs[name] = h
		}
		g, o, io := libs["libglib-2.0.so.0"], libs["libgobject-2.0.so.0"], libs["libgio-2.0.so.0"]
		for _, f := range []struct {
			fn   any
			lib  uintptr
			name string
		}{
			{&glib.errorFree, g, "g_error_free"},
			{&glib.variantNewString, g, "g_variant_new_string"},
			{&glib.variantNewTuple, g, "g_variant_new_tuple"},
			{&glib.variantUnref, g, "g_variant_unref"},
			{&glib.objectUnref, o, "g_object_unref"},
			{&glib.busGetSync, io, "g_bus_get_sync"},
			{&glib.dbusCallSync, io, "g_dbus_connection_call_sync"},
			{&glib.launchContextNew, io, "g_app_launch_context_new"},
			{&glib.launchContextSetenv, io, "g_app_launch_context_setenv"},
			{&glib.launchContextUnset, io, "g_app_launch_context_unsetenv"},
			{&glib.launchDefaultForURI, io, "g_app_info_launch_default_for_uri"},
		} {
			purego.RegisterLibFunc(f.fn, f.lib, f.name)
		}
	})
	return glib.err
}

// cString copies the C string p.
func cString(p *byte) string {
	var b []byte
	for ; p != nil && *p != 0; p = (*byte)(unsafe.Add(unsafe.Pointer(p), 1)) {
		b = append(b, *p)
	}
	return string(b)
}

// takeError is the GError e, which it frees.
func takeError(e *gError) error {
	if e == nil {
		return errors.New("no error given")
	}
	err := errors.New(cString(e.message))
	glib.errorFree(e)
	return err
}

// sessionCall calls method on the session bus and drops its answer.
func sessionCall(name, path, iface, method string, params unsafe.Pointer, timeoutMS int32) error {
	if err := loadGLib(); err != nil {
		return err
	}
	var gerr *gError
	conn := glib.busGetSync(2, nil, &gerr) // G_BUS_TYPE_SESSION
	if conn == nil {
		return takeError(gerr)
	}
	defer glib.objectUnref(conn)
	reply := glib.dbusCallSync(conn, name, path, iface, method, params, nil, 0, timeoutMS, nil, &gerr)
	if reply == nil {
		return takeError(gerr)
	}
	glib.variantUnref(reply)
	return nil
}

// tuple is a GVariant tuple of children, which it takes.
func tuple(children ...unsafe.Pointer) unsafe.Pointer {
	return glib.variantNewTuple(&children[0], uintptr(len(children)))
}

// launchAsUser opens uri with its default program, in the user's
// environment (appEnv). Main thread only.
func launchAsUser(uri string) error {
	if err := loadGLib(); err != nil {
		return err
	}
	ctx := glib.launchContextNew()
	defer glib.objectUnref(ctx)
	for key, user := range appEnv {
		if user == nil {
			glib.launchContextUnset(ctx, key)
		} else {
			glib.launchContextSetenv(ctx, key, *user)
		}
	}
	var gerr *gError
	if glib.launchDefaultForURI(uri, ctx, &gerr) == 0 {
		return takeError(gerr)
	}
	return nil
}
