//go:build markitai_static && !(darwin && arm64 && !ios) && !(linux && amd64 && !android)

package markitai

/*
#cgo CFLAGS: -I${SRCDIR}/native/include -I${SRCDIR}/../c
#error markitai_static is currently packaged only for darwin/arm64 and linux/amd64 (glibc); use the default dynamic binding on other targets
*/
import "C"
