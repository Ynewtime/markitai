//go:build markitai_static && darwin && arm64 && !ios

package markitai

/*
#cgo CFLAGS: -I${SRCDIR}/native/include
#cgo LDFLAGS: ${SRCDIR}/native/darwin_arm64/libmarkitai_ffi.a -framework Vision -framework Foundation -framework ImageIO -framework CoreGraphics -framework CoreFoundation -lobjc -liconv -lSystem -lc -lm
*/
import "C"
