//go:build markitai_static && linux && amd64 && !android

package markitai

/*
#cgo CFLAGS: -I${SRCDIR}/native/include
#cgo LDFLAGS: ${SRCDIR}/native/linux_amd64/libmarkitai_ffi.a -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc
*/
import "C"
