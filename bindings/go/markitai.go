// Package markitai converts documents with the in-process Rust engine.
package markitai

/*
#include "markitai.h"
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"runtime"
	"unsafe"
)

// Options preserves the shared conversion option names in its JSON form.
// Nil booleans inherit configuration; Bool(false) explicitly disables a feature.
type Options struct {
	OutputDir  string         `json:"output_dir,omitempty"`
	Config     map[string]any `json:"config"`
	LLM        *bool          `json:"llm,omitempty"`
	OCR        *bool          `json:"ocr,omitempty"`
	Screenshot *bool          `json:"screenshot,omitempty"`
	Alt        *bool          `json:"alt,omitempty"`
	Desc       *bool          `json:"desc,omitempty"`
	Profile    string         `json:"profile,omitempty"`
}

type ConversionUsage struct {
	CostUSD      float64                   `json:"cost_usd"`
	Requests     uint64                    `json:"requests"`
	InputTokens  uint64                    `json:"input_tokens"`
	OutputTokens uint64                    `json:"output_tokens"`
	ByModel      map[string]map[string]any `json:"by_model"`
}

type ConversionOutput struct {
	Source        string           `json:"source"`
	Markdown      string           `json:"markdown"`
	LLMMarkdown   *string          `json:"llm_markdown"`
	Frontmatter   map[string]any   `json:"frontmatter"`
	OutputPath    *string          `json:"output_path"`
	LLMOutputPath *string          `json:"llm_output_path"`
	Assets        []string         `json:"assets"`
	Screenshots   []string         `json:"screenshots"`
	Images        []map[string]any `json:"images"`
	Usage         ConversionUsage  `json:"usage"`
	SkipReason    *string          `json:"skip_reason"`
	Duration      float64          `json:"duration"`
	Warnings      []string         `json:"warnings"`
	// A PDF's pages, each with where its text came from; empty otherwise.
	PageSources []PageSource `json:"page_sources"`
}

// PageSource is where one PDF page's text came from: Source is "text" (the
// text layer), "ocr" or "mixed", Reason a short English note.
type PageSource struct {
	Page   int    `json:"page"`
	Source string `json:"source"`
	Reason string `json:"reason"`
}

type ConversionError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	// Nil means no accounting was supplied; it does not establish a free call.
	Usage *ConversionUsage `json:"usage,omitempty"`
}

func (err *ConversionError) Error() string { return err.Message }

func Bool(value bool) *bool { return &value }

func Version() string { return C.GoString(C.markitai_version()) }

// Convert blocks until the conversion finishes. Concurrent calls are allowed.
// Built-in conversion calls Rust in process; optional provider/browser/Office runtimes may start children.
func Convert(source string, options *Options) (*ConversionOutput, error) {
	request, err := json.Marshal(struct {
		Source  string   `json:"source"`
		Options *Options `json:"options,omitempty"`
	}{source, options})
	if err != nil {
		return nil, fmt.Errorf("encode conversion request: %w", err)
	}
	response, err := ConvertJSON(request)
	if err != nil {
		return nil, err
	}
	var envelope struct {
		OK     bool              `json:"ok"`
		Result *ConversionOutput `json:"result"`
		Error  *ConversionError  `json:"error"`
	}
	if err = json.Unmarshal(response, &envelope); err != nil {
		return nil, fmt.Errorf("decode native conversion response: %w", err)
	}
	if !envelope.OK {
		if envelope.Error == nil {
			return nil, fmt.Errorf("native response is missing its error")
		}
		return nil, envelope.Error
	}
	if envelope.Result == nil {
		return nil, fmt.Errorf("native response is missing its result")
	}
	return envelope.Result, nil
}

// ConvertJSON returns the raw shared JSON envelope, copied into Go memory.
// Conversion failures remain inside that envelope; errors here indicate an ABI
// failure. The native response is released before this function returns.
func ConvertJSON(request []byte) ([]byte, error) {
	if C.markitai_abi_version() != 1 {
		return nil, fmt.Errorf("unsupported Markitai ABI version")
	}
	var data *C.uint8_t
	if len(request) > 0 {
		data = (*C.uint8_t)(unsafe.Pointer(&request[0]))
	}
	buffer := C.markitai_convert_json(data, C.size_t(len(request)))
	runtime.KeepAlive(request)
	defer C.markitai_buffer_free(&buffer)
	if buffer.data == nil {
		return nil, fmt.Errorf("native conversion returned a null buffer")
	}
	if uint64(buffer.len) > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("native response exceeds Go slice capacity")
	}
	view := unsafe.Slice((*byte)(unsafe.Pointer(buffer.data)), int(buffer.len))
	return append([]byte(nil), view...), nil
}
