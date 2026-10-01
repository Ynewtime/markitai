//! Explicit first use of the macOS frameworks behind OCR, HEIF/AVIF decoding
//! and PDF rasterization.
//!
//! The command-line executables and this crate's tests link these frameworks
//! delay-initialized (`build.rs`). On macOS 15 and later dyld then postpones
//! their initializers and Objective-C class registration until first use, so a
//! conversion that never needs them does not pay for that work at launch. A C
//! call into a delayed framework completes the initialization first, but an
//! Objective-C class lookup by name does not and fails instead. Each backend
//! therefore opens its framework here before its first call. Where the
//! framework was initialized at launch (older systems, the language bindings),
//! opening it only takes another reference.

use std::ffi::CStr;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub(crate) enum Framework {
    CoreGraphics,
    ImageIO,
    Vision,
}

impl Framework {
    fn name(self) -> &'static str {
        match self {
            Self::CoreGraphics => "CoreGraphics",
            Self::ImageIO => "ImageIO",
            Self::Vision => "Vision",
        }
    }

    /// The install name recorded in the executable's load commands.
    fn path(self) -> &'static CStr {
        match self {
            Self::CoreGraphics => {
                c"/System/Library/Frameworks/CoreGraphics.framework/Versions/A/CoreGraphics"
            }
            Self::ImageIO => c"/System/Library/Frameworks/ImageIO.framework/Versions/A/ImageIO",
            Self::Vision => c"/System/Library/Frameworks/Vision.framework/Versions/A/Vision",
        }
    }
}

/// Initializes `framework` and its dependencies once per process. The handle
/// is never closed, so the framework stays loaded for the process lifetime.
pub(crate) fn open(framework: Framework) -> Result<(), String> {
    static OPENED: [OnceLock<bool>; 3] = [const { OnceLock::new() }; 3];
    let opened = *OPENED[framework as usize].get_or_init(|| {
        // SAFETY: The path is a NUL-terminated constant and dlopen is
        // thread-safe. The returned handle is intentionally leaked.
        let handle = unsafe {
            libc::dlopen(
                framework.path().as_ptr(),
                libc::RTLD_LAZY | libc::RTLD_LOCAL,
            )
        };
        !handle.is_null()
    });
    if opened {
        Ok(())
    } else {
        Err(format!(
            "cannot load the system {} framework",
            framework.name()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2::runtime::AnyClass;

    #[test]
    fn opened_frameworks_register_their_classes_and_stay_open() {
        for framework in [
            Framework::CoreGraphics,
            Framework::ImageIO,
            Framework::Vision,
        ] {
            open(framework).unwrap();
            open(framework).unwrap();
        }
        // Name lookups are what delay-initialized frameworks cannot satisfy
        // before they are opened; Foundation arrives as Vision's dependency.
        for class in [
            c"VNRecognizeTextRequest",
            c"VNImageRequestHandler",
            c"NSString",
        ] {
            assert!(AnyClass::get(class).is_some(), "{class:?}");
        }
    }
}
