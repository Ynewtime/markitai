package main

import (
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// gtk holds what the app calls in GTK and GObject itself, beside mygo,
// which loads the same libraries; loading them again shares them. Main
// thread only, but for loadGTK.
var gtk struct {
	once sync.Once
	err  error

	widgetType        func() uintptr
	containerType     func() uintptr
	entryType         func() uintptr
	fileChooserType   func() uintptr
	messageDialogType func() uintptr
	classRef          func(t uintptr) unsafe.Pointer
	signalLookup      func(name string, owner uintptr) uint32
	addEmissionHook   func(signal, detail uint32, hook, data, destroy uintptr) uint64
	isA               func(instance unsafe.Pointer, t uintptr) int32
	valueGetObject    func(v unsafe.Pointer) unsafe.Pointer
	valueGetBoxed     func(v unsafe.Pointer) unsafe.Pointer
	objectGetData     func(o unsafe.Pointer, key string) uintptr
	objectSetData     func(o unsafe.Pointer, key string, data uintptr)
	signalConnect     func(instance unsafe.Pointer, signal string, handler, data, destroy uintptr, flags int32) uint64
	selectionURIs     func(sel unsafe.Pointer) **byte
	transientFor      func(w unsafe.Pointer) unsafe.Pointer
	windowSize        func(w unsafe.Pointer, width, height *int32)
	widgetWindow      func(w unsafe.Pointer) unsafe.Pointer
	displayDefault    func() unsafe.Pointer
	monitorAtWindow   func(d, w unsafe.Pointer) unsafe.Pointer
	monitorAt         func(d unsafe.Pointer, n int32) unsafe.Pointer
	workarea          func(m unsafe.Pointer, r *gdkRectangle)
	setGeometryHints  func(w, geometryWidget unsafe.Pointer, g *gdkGeometry, mask int32)
	preferredWidth    func(w unsafe.Pointer, least, natural *int32)
	forall            func(c unsafe.Pointer, callback, data uintptr)
	entryWidthChars   func(e unsafe.Pointer) int32
	setEntryWidth     func(e unsafe.Pointer, chars int32)
	listToplevels     func() *gList
	listFree          func(l *gList)
	dialogResponse    func(d unsafe.Pointer, response int32)
}

// gList is GLib's GList.
type gList struct {
	data       unsafe.Pointer
	next, prev *gList
}

// loadGTK loads gtk's functions once, and says whether it could.
func loadGTK() error {
	gtk.once.Do(func() {
		libs := map[string]uintptr{}
		for _, name := range []string{"libglib-2.0.so.0", "libgobject-2.0.so.0", "libgtk-3.so.0"} {
			h, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err != nil {
				gtk.err = err
				return
			}
			libs[name] = h
		}
		g, o, t := libs["libglib-2.0.so.0"], libs["libgobject-2.0.so.0"], libs["libgtk-3.so.0"]
		for _, f := range []struct {
			fn   any
			lib  uintptr
			name string
		}{
			{&gtk.widgetType, t, "gtk_widget_get_type"},
			{&gtk.containerType, t, "gtk_container_get_type"},
			{&gtk.entryType, t, "gtk_entry_get_type"},
			{&gtk.fileChooserType, t, "gtk_file_chooser_dialog_get_type"},
			{&gtk.messageDialogType, t, "gtk_message_dialog_get_type"},
			{&gtk.classRef, o, "g_type_class_ref"},
			{&gtk.signalLookup, o, "g_signal_lookup"},
			{&gtk.addEmissionHook, o, "g_signal_add_emission_hook"},
			{&gtk.isA, o, "g_type_check_instance_is_a"},
			{&gtk.valueGetObject, o, "g_value_get_object"},
			{&gtk.valueGetBoxed, o, "g_value_get_boxed"},
			{&gtk.objectGetData, o, "g_object_get_data"},
			{&gtk.objectSetData, o, "g_object_set_data"},
			{&gtk.signalConnect, o, "g_signal_connect_data"},
			{&gtk.selectionURIs, t, "gtk_selection_data_get_uris"},
			{&gtk.transientFor, t, "gtk_window_get_transient_for"},
			{&gtk.windowSize, t, "gtk_window_get_size"},
			{&gtk.widgetWindow, t, "gtk_widget_get_window"},
			{&gtk.displayDefault, t, "gdk_display_get_default"},
			{&gtk.monitorAtWindow, t, "gdk_display_get_monitor_at_window"},
			{&gtk.monitorAt, t, "gdk_display_get_monitor"},
			{&gtk.workarea, t, "gdk_monitor_get_workarea"},
			{&gtk.setGeometryHints, t, "gtk_window_set_geometry_hints"},
			{&gtk.preferredWidth, t, "gtk_widget_get_preferred_width"},
			{&gtk.forall, t, "gtk_container_forall"},
			{&gtk.entryWidthChars, t, "gtk_entry_get_width_chars"},
			{&gtk.setEntryWidth, t, "gtk_entry_set_width_chars"},
			{&gtk.listToplevels, t, "gtk_window_list_toplevels"},
			{&gtk.listFree, g, "g_list_free"},
			{&gtk.dialogResponse, t, "gtk_dialog_response"},
		} {
			purego.RegisterLibFunc(f.fn, f.lib, f.name)
		}
	})
	return gtk.err
}
