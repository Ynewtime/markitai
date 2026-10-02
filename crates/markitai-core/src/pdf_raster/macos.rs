use super::{RasterSize, Result, failure, raster_size};
use crate::system_frameworks::{self, Framework};
use image::RgbImage;
use objc2_core_foundation::{CFData, CFRetained, CGAffineTransform, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGDataProvider, CGImageAlphaInfo,
    CGImageByteOrderInfo, CGPDFBox, CGPDFDocument, CGPDFPage,
};
use std::marker::PhantomData;
use std::rc::Rc;

/// A document and its immutable backing data stay alive through every draw.
/// The session is deliberately confined to its caller's thread.
pub(crate) struct PdfRasterSession {
    document: CFRetained<CGPDFDocument>,
    _provider: CFRetained<CGDataProvider>,
    _data: CFRetained<CFData>,
    pages: usize,
    _thread: PhantomData<Rc<()>>,
}

struct PageFrame {
    page: CFRetained<CGPDFPage>,
    crop: CGRect,
    size: RasterSize,
    target: CGRect,
    transform: CGAffineTransform,
}

fn checked_rect(rect: CGRect) -> Result<CGRect> {
    let values = [
        rect.origin.x,
        rect.origin.y,
        rect.size.width,
        rect.size.height,
    ];
    if values.iter().any(|value| !value.is_finite())
        || rect.size.width <= 0.
        || rect.size.height <= 0.
        || !(rect.origin.x + rect.size.width).is_finite()
        || !(rect.origin.y + rect.size.height).is_finite()
    {
        return Err(failure("page box is empty or has invalid coordinates"));
    }
    Ok(rect)
}

fn intersection(first: CGRect, second: CGRect) -> Result<CGRect> {
    let first = checked_rect(first)?;
    let second = checked_rect(second)?;
    let left = first.origin.x.max(second.origin.x);
    let bottom = first.origin.y.max(second.origin.y);
    let right = (first.origin.x + first.size.width).min(second.origin.x + second.size.width);
    let top = (first.origin.y + first.size.height).min(second.origin.y + second.size.height);
    checked_rect(CGRect::new(
        CGPoint::new(left, bottom),
        CGSize::new(right - left, top - bottom),
    ))
}

impl PdfRasterSession {
    /// `bytes` has passed the shared input checks (size bound, PDF header).
    pub(crate) fn open(bytes: &[u8]) -> Result<Self> {
        system_frameworks::open(Framework::CoreGraphics).map_err(|message| failure(&message))?;
        // SAFETY: The slice is valid for its checked length; CFDataCreate copies
        // it before returning. The caller's buffer may then be dropped.
        let data = unsafe { CFData::new(None, bytes.as_ptr(), bytes.len() as isize) }
            .ok_or_else(|| failure("cannot retain document bytes"))?;
        let provider = CGDataProvider::with_cf_data(Some(&data))
            .ok_or_else(|| failure("cannot create the document data provider"))?;
        let document = CGPDFDocument::with_provider(Some(&provider))
            .ok_or_else(|| failure("cannot open the PDF document"))?;
        if !CGPDFDocument::is_unlocked(Some(&document)) {
            return Err(failure(
                "PDF is locked; password-protected rendering is not supported",
            ));
        }
        let pages = CGPDFDocument::number_of_pages(Some(&document));
        if pages == 0 {
            return Err(failure("document contains no readable pages"));
        }
        Ok(Self {
            document,
            _provider: provider,
            _data: data,
            pages,
            _thread: PhantomData,
        })
    }

    pub(crate) fn pages(&self) -> usize {
        self.pages
    }

    fn frame(&self, page_1_based: usize, dpi: f64) -> Result<PageFrame> {
        if page_1_based == 0 || page_1_based > self.pages {
            return Err(failure("page number is outside the document"));
        }
        let page = CGPDFDocument::page(Some(&self.document), page_1_based)
            .ok_or_else(|| failure("cannot access the requested page"))?;
        let crop = intersection(
            CGPDFPage::box_rect(Some(&page), CGPDFBox::MediaBox),
            CGPDFPage::box_rect(Some(&page), CGPDFBox::CropBox),
        )?;
        let rotation = CGPDFPage::rotation_angle(Some(&page)).rem_euclid(360);
        if rotation % 90 != 0 {
            return Err(failure("page rotation is not a multiple of 90 degrees"));
        }
        let (width, height) = if matches!(rotation, 90 | 270) {
            (crop.size.height, crop.size.width)
        } else {
            (crop.size.width, crop.size.height)
        };
        let size = raster_size(width, height, dpi)?;
        let target = CGRect::new(
            CGPoint::new(0., 0.),
            CGSize::new(f64::from(size.width), f64::from(size.height)),
        );
        // Quartz fits oversized pages down, but does not upscale a page merely
        // because the destination is larger. Ask it only for the point-space
        // crop/rotation transform, then explicitly apply the requested DPI.
        let page_target = CGRect::new(CGPoint::new(0., 0.), CGSize::new(width, height));
        let mut transform =
            CGPDFPage::drawing_transform(Some(&page), CGPDFBox::CropBox, page_target, 0, true);
        let scale = dpi / 72.;
        transform.a *= scale;
        transform.b *= scale;
        transform.c *= scale;
        transform.d *= scale;
        transform.tx = transform.tx * scale + (f64::from(size.width) - width * scale) / 2.;
        transform.ty = transform.ty * scale + (f64::from(size.height) - height * scale) / 2.;
        let determinant = transform.a * transform.d - transform.b * transform.c;
        if [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.tx,
            transform.ty,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || !determinant.is_finite()
            || determinant == 0.
        {
            return Err(failure("page drawing transform is invalid"));
        }
        Ok(PageFrame {
            page,
            crop,
            size,
            target,
            transform,
        })
    }

    /// Checks the complete page geometry without allocating its pixel buffers.
    pub(crate) fn dimensions(&self, page_1_based: usize, dpi: f64) -> Result<(u32, u32)> {
        let frame = self.frame(page_1_based, dpi)?;
        Ok((frame.size.width, frame.size.height))
    }

    pub(crate) fn render(&self, page_1_based: usize, dpi: f64) -> Result<RgbImage> {
        let PageFrame {
            page,
            crop,
            size,
            target,
            transform,
        } = self.frame(page_1_based, dpi)?;
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(size.rgba_bytes)
            .map_err(|_| failure("cannot allocate the bounded page bitmap"))?;
        pixels.resize(size.rgba_bytes, 255u8);
        let color = CGColorSpace::new_device_rgb()
            .ok_or_else(|| failure("cannot create the page color space"))?;
        // SAFETY: The owned allocation is exactly row_bytes * height, remains
        // immovable/unread while Quartz uses it, and outlives the context. RGBX
        // byte order is explicit so no platform-native channel inference occurs.
        let context = unsafe {
            CGBitmapContextCreate(
                pixels.as_mut_ptr().cast(),
                size.width as usize,
                size.height as usize,
                8,
                size.row_bytes,
                Some(&color),
                CGImageByteOrderInfo::Order32Big.0 | CGImageAlphaInfo::NoneSkipLast.0,
            )
        }
        .ok_or_else(|| failure("cannot create the page bitmap context"))?;
        CGContext::set_rgb_fill_color(Some(&context), 1., 1., 1., 1.);
        CGContext::fill_rect(Some(&context), target);
        CGContext::concat_ctm(Some(&context), transform);
        CGContext::clip_to_rect(Some(&context), crop);
        CGContext::draw_pdf_page(Some(&context), Some(&page));
        CGContext::flush(Some(&context));
        drop(context);
        // The opaque RGBX context has already composited all PDF transparency
        // onto white. Drop its unused fourth channel without interpreting alpha.
        let mut rgb = Vec::new();
        rgb.try_reserve_exact(size.rgb_bytes)
            .map_err(|_| failure("cannot allocate the bounded RGB result"))?;
        for pixel in pixels.as_chunks::<4>().0 {
            rgb.extend_from_slice(&pixel[..3]);
        }
        RgbImage::from_raw(size.width, size.height, rgb)
            .ok_or_else(|| failure("rendered RGB dimensions are inconsistent"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_box_geometry_is_rejected_before_sdk_drawing() {
        for rect in [
            CGRect::new(CGPoint::new(0., 0.), CGSize::new(0., 1.)),
            CGRect::new(CGPoint::new(f64::NAN, 0.), CGSize::new(1., 1.)),
            CGRect::new(CGPoint::new(0., 0.), CGSize::new(f64::INFINITY, 1.)),
            CGRect::new(CGPoint::new(f64::MAX, 0.), CGSize::new(f64::MAX, 1.)),
        ] {
            assert!(checked_rect(rect).is_err());
        }
        let first = CGRect::new(CGPoint::new(0., 0.), CGSize::new(10., 10.));
        let disjoint = CGRect::new(CGPoint::new(20., 20.), CGSize::new(10., 10.));
        assert!(intersection(first, disjoint).is_err());
    }
}
