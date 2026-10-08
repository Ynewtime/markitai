//! Owned JSON responses for C-compatible callers. See `bindings/c/markitai.h`.

use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;

#[repr(C)]
pub struct MarkitaiBuffer {
    pub data: *mut u8,
    pub len: usize,
}

impl MarkitaiBuffer {
    fn from_string(value: String) -> Self {
        let bytes = value.into_bytes().into_boxed_slice();
        let len = bytes.len();
        Self {
            data: Box::into_raw(bytes).cast(),
            len,
        }
    }
}

fn error(code: &str, message: &str) -> String {
    serde_json::json!({"ok": false, "error": {"code": code, "message": message}}).to_string()
}

#[unsafe(no_mangle)]
pub extern "C" fn markitai_abi_version() -> u32 {
    1
}

/// A process-lifetime UTF-8 string. The caller must not modify or free it.
#[unsafe(no_mangle)]
pub extern "C" fn markitai_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Convert a JSON request and return an owned UTF-8 JSON response.
///
/// # Safety
/// `request` must point to `len` readable bytes for the duration of this call.
/// Null is accepted only for length zero. The response must be released exactly
/// once with `markitai_buffer_free`; it has no trailing NUL byte.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_convert_json(request: *const u8, len: usize) -> MarkitaiBuffer {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if len > MAX_REQUEST_BYTES {
            return error("invalid_input", "JSON request exceeds the 64 MiB ABI limit");
        }
        if request.is_null() && len != 0 {
            return error(
                "invalid_input",
                "Request pointer is null with nonzero length",
            );
        }
        let bytes = if len == 0 {
            &[][..]
        } else {
            // The caller owns this allocation and keeps it alive until return.
            unsafe { std::slice::from_raw_parts(request, len) }
        };
        match std::str::from_utf8(bytes) {
            Ok(request) => markitai_core::convert_json(request),
            Err(_) => error("invalid_input", "JSON request is not valid UTF-8"),
        }
    }))
    .unwrap_or_else(|_| markitai_core::internal_error_json());
    MarkitaiBuffer::from_string(response)
}

/// Release a response and clear its fields. Calling again on the same cleared
/// struct or passing null is harmless. Never free a copied ownership handle.
///
/// # Safety
/// `buffer` must be null or point to a writable `MarkitaiBuffer` returned by
/// `markitai_convert_json`, with its original pointer and length unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_buffer_free(buffer: *mut MarkitaiBuffer) {
    if buffer.is_null() {
        return;
    }
    let buffer = unsafe { &mut *buffer };
    if !buffer.data.is_null() {
        let allocation = ptr::slice_from_raw_parts_mut(buffer.data, buffer.len);
        buffer.data = ptr::null_mut();
        buffer.len = 0;
        unsafe { drop(Box::from_raw(allocation)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    unsafe fn decode_and_free(mut buffer: MarkitaiBuffer) -> Value {
        let bytes = unsafe { std::slice::from_raw_parts(buffer.data, buffer.len) };
        let result = serde_json::from_slice(bytes).unwrap();
        unsafe { markitai_buffer_free(&mut buffer) };
        assert!(buffer.data.is_null());
        assert_eq!(buffer.len, 0);
        unsafe { markitai_buffer_free(&mut buffer) };
        result
    }

    #[test]
    fn invalid_buffers_return_structured_errors() {
        unsafe {
            assert_eq!(
                decode_and_free(markitai_convert_json(ptr::null(), 1))["error"]["code"],
                "invalid_input"
            );
            assert_eq!(
                decode_and_free(markitai_convert_json(ptr::null(), 0))["ok"],
                false
            );
            assert_eq!(
                decode_and_free(markitai_convert_json([0xff].as_ptr(), 1))["error"]["code"],
                "invalid_input"
            );
            assert_eq!(
                decode_and_free(markitai_convert_json(ptr::null(), MAX_REQUEST_BYTES + 1))["ok"],
                false
            );
            markitai_buffer_free(ptr::null_mut());
        }
    }

    #[test]
    fn repeated_unicode_conversion_and_ownership() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("文档.md");
        std::fs::write(&source, "# Native\n\n你好 🌍\n").unwrap();
        let request =
            serde_json::json!({"source": source, "options": {"config": {}, "llm": false}})
                .to_string();
        for _ in 0..64 {
            let response =
                unsafe { decode_and_free(markitai_convert_json(request.as_ptr(), request.len())) };
            assert_eq!(response["ok"], true, "{response}");
            assert!(
                response["result"]["markdown"]
                    .as_str()
                    .unwrap()
                    .contains("你好 🌍")
            );
        }
    }
}
