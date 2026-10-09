//! AVIF decoding without the operating system, on Windows and Linux: the
//! container is read by `heifer`'s ISO BMFF parser and each AV1 image item by
//! rav1d (the Rust port of dav1d, vendored without its assembly). Colour
//! conversion and the crop, rotation and mirror transforms are heifer's, as
//! for HEIC. Coded items, grids of them and alpha planes are read; overlays
//! and other derived images are unsupported errors.

use heifer::Image;
use heifer::color::{ColorParams, frame_to_rgba};
use heifer::heifer_hevc_dec::recon::Frame;
use heifer::heifer_isobmff::boxes::{ColorInfo, Property};
use heifer::heifer_isobmff::{HeifFile, ItemId};
use rav1d::pixel::YUVRange;
use rav1d::{Decoder, PixelLayout, PlanarImageComponent, Rav1dError, Settings};

use super::portable::unsupported;
use crate::{Error, Result};

const MAX_PIXELS: u64 = super::super::MAX_PIXELS;
/// Derived images within derived images, as heifer allows.
const MAX_DEPTH: u32 = 4;
/// More tiles than a 32-megapixel image of 256 by 256 tiles needs.
const MAX_TILES: usize = 1024;
const ALPHA: [&str; 2] = [
    "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha",
    "urn:mpeg:hevc:2015:auxid:1",
];

fn failure(message: impl std::fmt::Display) -> Error {
    super::super::error(format!("AVIF: {message}"))
}

/// Whether an item is AV1-coded or derived from AV1-coded items.
pub(super) fn is_av1(file: &HeifFile<'_>, id: ItemId) -> bool {
    let mut id = id;
    for _ in 0..=MAX_DEPTH {
        let Ok(item) = file.item(id) else {
            return false;
        };
        match &item.item_type.0 {
            b"av01" => return true,
            b"grid" | b"iovl" => match file.referenced_items(id, b"dimg").first() {
                Some(&input) => id = input,
                None => return false,
            },
            _ => return false,
        }
    }
    false
}

/// The primary image with its alpha plane and transforms.
pub(super) fn decode(file: &HeifFile<'_>) -> Result<Image> {
    item(file, file.primary_id, 0, false)
}

fn item(file: &HeifFile<'_>, id: ItemId, depth: u32, alpha: bool) -> Result<Image> {
    if depth > MAX_DEPTH {
        return Err(failure("derived images are nested too deeply"));
    }
    if let Some((width, height)) = file.image_size(id).map_err(failure)?
        && u64::from(width) * u64::from(height) > MAX_PIXELS
    {
        return Err(failure("the image exceeds 32 million pixels"));
    }
    let mut image = match &file.item(id).map_err(failure)?.item_type.0 {
        b"av01" => coded(file, id, alpha)?,
        b"grid" => grid(file, id, depth, alpha)?,
        other => {
            return Err(unsupported(&format!(
                "This AVIF image uses a derived image the portable decoder does not read ({})",
                String::from_utf8_lossy(other)
            )));
        }
    };
    // An alpha plane belongs to the image it is attached to, before that
    // image's transforms; its own transforms repeat them.
    if alpha {
        return Ok(image);
    }
    if let Some(plane) = alpha_item(file, id)? {
        let plane = item(file, plane, depth + 1, true)?;
        attach_alpha(&mut image, &plane)?;
    }
    for property in file.item_properties(id).map_err(failure)? {
        image = match *property {
            Property::CleanAperture {
                width,
                height,
                horiz_offset,
                vert_offset,
            } => clean_aperture(&image, width, height, horiz_offset, vert_offset)?,
            Property::Rotation(quarters) => image.rotate_ccw(quarters),
            Property::Mirror(top_bottom) => image.mirror(top_bottom),
            _ => image,
        };
    }
    Ok(image)
}

fn alpha_item(file: &HeifFile<'_>, id: ItemId) -> Result<Option<ItemId>> {
    for auxiliary in file.referencing_items(id, b"auxl") {
        if file
            .item_properties(auxiliary)
            .map_err(failure)?
            .any(|property| matches!(property, Property::AuxiliaryType(kind) if ALPHA.contains(&kind.as_str())))
        {
            return Ok(Some(auxiliary));
        }
    }
    Ok(None)
}

/// Copies the alpha plane (its red channel) into the image's alpha channel.
fn attach_alpha(image: &mut Image, plane: &Image) -> Result<()> {
    if (plane.width, plane.height) != (image.width, image.height) {
        return Err(failure("the alpha plane and the image differ in size"));
    }
    let (from, to) = (u32::from(plane.max_value()), u32::from(image.max_value()));
    for (pixel, value) in image
        .data
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(plane.data.as_chunks::<4>().0)
    {
        pixel[3] = ((u32::from(value[0]) * to + from / 2) / from) as u16;
    }
    image.has_alpha = true;
    Ok(())
}

/// The `colr` (nclx) colour parameters of an item, if it has them; otherwise
/// the AV1 sequence header's apply.
fn nclx(file: &HeifFile<'_>, id: ItemId) -> Result<Option<ColorParams>> {
    Ok(file
        .item_properties(id)
        .map_err(failure)?
        .find_map(|property| match property {
            Property::Color(ColorInfo::Nclx {
                matrix, full_range, ..
            }) => Some(ColorParams {
                matrix: *matrix,
                full_range: *full_range,
            }),
            _ => None,
        }))
}

fn coded(file: &HeifFile<'_>, id: ItemId, alpha: bool) -> Result<Image> {
    let nclx = nclx(file, id)?;
    let (frame, width, height) = picture(&file.item_data(id).map_err(failure)?)?;
    let params = nclx.unwrap_or_else(|| ColorParams::from_frame(&frame));
    let mut image = if alpha {
        alpha_plane(&frame, params.full_range)
    } else {
        frame_to_rgba(&frame, params)
    };
    // `ispe` is authoritative; the coded picture may be larger.
    let (width, height) = file
        .image_size(id)
        .map_err(failure)?
        .unwrap_or((width, height));
    if width > image.width || height > image.height {
        return Err(failure("the coded image is smaller than its declared size"));
    }
    if (width, height) != (image.width, image.height) {
        image = image.crop(0, 0, width, height);
    }
    Ok(image)
}

/// The luma of an alpha image, scaled to its full range, in the red channel.
fn alpha_plane(frame: &Frame, full_range: bool) -> Image {
    let depth = frame.bit_depth[0];
    let max = (1u32 << depth) - 1;
    let (offset, span) = if full_range {
        (0, max)
    } else {
        (16 << (depth - 8), 219 << (depth - 8))
    };
    let data = frame.planes[0]
        .iter()
        .flat_map(|&value| {
            let value = (u32::from(value).saturating_sub(offset) * max + span / 2) / span;
            [value.min(max) as u16, 0, 0, max as u16]
        })
        .collect();
    Image {
        width: frame.widths[0],
        height: frame.heights[0],
        bit_depth: depth,
        has_alpha: false,
        data,
    }
}

/// Decodes one AV1 image item payload. The planes are padded to whole chroma
/// samples (heifer's conversion expects even sizes for subsampled chroma);
/// returns them with the picture's own size.
fn picture(data: &[u8]) -> Result<(Frame, u32, u32)> {
    let av1 = |error: Rav1dError| failure(format!("the AV1 data cannot be decoded ({error})"));
    if data.is_empty() {
        return Err(failure("an image item has no AV1 data"));
    }
    let mut settings = Settings::new();
    // One thread: no worker threads, so a failure stays on this thread.
    settings.set_n_threads(1);
    settings.set_max_frame_delay(1);
    settings.set_all_layers(false);
    settings.set_frame_size_limit(MAX_PIXELS as u32);
    // A damaged stream is this image's error, not lines on the host's stderr.
    settings.set_logging(false);
    let mut decoder = Decoder::with_settings(&settings).map_err(av1)?;
    let mut sent = match decoder.send_data(data.into(), None, None, None) {
        Ok(()) => true,
        Err(Rav1dError::TryAgain) => false,
        Err(error) => return Err(av1(error)),
    };
    // A still image is the first shown frame; each round consumes input.
    let mut picture = None;
    for _ in 0..64 {
        match decoder.get_picture() {
            Ok(shown) => {
                picture = Some(shown);
                break;
            }
            Err(Rav1dError::TryAgain) if !sent => {}
            Err(Rav1dError::TryAgain) => break,
            Err(error) => return Err(av1(error)),
        }
        sent = match decoder.send_pending_data() {
            Ok(()) => true,
            Err(Rav1dError::TryAgain) => false,
            Err(error) => return Err(av1(error)),
        };
    }
    let picture = picture.ok_or_else(|| failure("the AV1 data has no image"))?;
    let (width, height) = (picture.width(), picture.height());
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(failure("the image is empty or exceeds 32 million pixels"));
    }
    let depth = picture
        .bits_per_component()
        .ok_or_else(|| failure("AV1 bit depth is invalid"))?
        .0;
    let (sx, sy, chroma) = match picture.pixel_layout() {
        PixelLayout::I400 => (1, 1, false),
        PixelLayout::I420 => (2, 2, true),
        PixelLayout::I422 => (2, 1, true),
        PixelLayout::I444 => (1, 1, true),
    };
    let wide = picture.bit_depth() > 8;
    // Rows and columns past the picture repeat its last ones.
    let read = |component, (columns, rows): (u32, u32), (width, height): (u32, u32)| {
        let plane = picture.plane(component);
        let stride = picture.stride(component) as usize;
        let mut samples = Vec::with_capacity(columns as usize * rows as usize);
        for y in 0..rows {
            let start = y.min(height - 1) as usize * stride;
            let row = plane
                .get(start..)
                .ok_or_else(|| failure("AV1 plane is shorter than its size"))?;
            for x in 0..columns {
                let x = x.min(width - 1) as usize;
                let sample = if wide {
                    row.get(2 * x..2 * x + 2)
                        .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]))
                } else {
                    row.get(x).map(|&byte| u16::from(byte))
                };
                samples.push(sample.ok_or_else(|| failure("AV1 plane is shorter than its size"))?);
            }
        }
        Ok::<_, Error>(samples)
    };
    let padded = (width.div_ceil(sx) * sx, height.div_ceil(sy) * sy);
    let luma = read(PlanarImageComponent::Y, padded, (width, height))?;
    let (chroma_size, u, v) = if chroma {
        let size = (padded.0 / sx, padded.1 / sy);
        let coded = (width.div_ceil(sx), height.div_ceil(sy));
        (
            size,
            read(PlanarImageComponent::U, size, coded)?,
            read(PlanarImageComponent::V, size, coded)?,
        )
    } else {
        ((0, 0), Vec::new(), Vec::new())
    };
    let frame = Frame {
        widths: [padded.0, chroma_size.0, chroma_size.0],
        heights: [padded.1, chroma_size.1, chroma_size.1],
        planes: [luma, u, v],
        bit_depth: [depth, depth],
        chroma_format: match (chroma, sx, sy) {
            (false, ..) => 0,
            (true, 2, 2) => 1,
            (true, 2, _) => 2,
            _ => 3,
        },
        full_range: picture.color_range() == YUVRange::Full,
        matrix_coeffs: picture.matrix_coefficients() as u8,
    };
    Ok((frame, width, height))
}

fn grid(file: &HeifFile<'_>, id: ItemId, depth: u32, alpha: bool) -> Result<Image> {
    let grid = file.grid(id).map_err(failure)?;
    let tiles = file.referenced_items(id, b"dimg");
    let columns = u32::from(grid.columns);
    if tiles.is_empty()
        || tiles.len() > MAX_TILES
        || tiles.len() != usize::from(grid.rows) * usize::from(grid.columns)
    {
        return Err(failure(
            "the grid's tiles do not match its rows and columns or are too many",
        ));
    }
    let (width, height) = (grid.output_width, grid.output_height);
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(failure("the grid is empty or exceeds 32 million pixels"));
    }
    let mut canvas: Option<Image> = None;
    for (index, &tile) in tiles.iter().enumerate() {
        let tile = item(file, tile, depth + 1, alpha)?;
        let canvas =
            canvas.get_or_insert_with(|| Image::filled(width, height, tile.bit_depth, [0; 4]));
        let (column, row) = (index as u32 % columns, index as u32 / columns);
        place(canvas, &tile, column, row, columns, grid.rows.into())?;
    }
    canvas.ok_or_else(|| failure("the grid has no tiles"))
}

/// Copies a tile into the canvas without blending; tiles must cover the
/// canvas, and the last row and column may extend past it.
fn place(
    canvas: &mut Image,
    tile: &Image,
    column: u32,
    row: u32,
    columns: u32,
    rows: u32,
) -> Result<()> {
    let (width, height) = (tile.width, tile.height);
    if tile.bit_depth != canvas.bit_depth
        || u64::from(width) * u64::from(columns) < u64::from(canvas.width)
        || u64::from(height) * u64::from(rows) < u64::from(canvas.height)
        || u64::from(width) * u64::from(columns - 1) >= u64::from(canvas.width)
        || u64::from(height) * u64::from(rows - 1) >= u64::from(canvas.height)
    {
        return Err(failure(
            "the grid's tiles differ or do not cover its canvas",
        ));
    }
    let (x, y) = ((column * width) as usize, (row * height) as usize);
    let stride = canvas.width as usize;
    let copied = (width as usize).min(stride - x);
    for line in 0..(height as usize).min(canvas.height as usize - y) {
        let source = &tile.data[line * width as usize * 4..][..copied * 4];
        canvas.data[((y + line) * stride + x) * 4..][..copied * 4].copy_from_slice(source);
    }
    canvas.has_alpha |= tile.has_alpha;
    Ok(())
}

/// A `clap` crop, centred as ISO/IEC 14496-12 specifies (heifer's rule).
fn clean_aperture(
    image: &Image,
    width: (u32, u32),
    height: (u32, u32),
    horizontal: (i32, u32),
    vertical: (i32, u32),
) -> Result<Image> {
    if width.1 == 0 || height.1 == 0 || horizontal.1 == 0 || vertical.1 == 0 {
        return Err(failure("the clean aperture has a zero denominator"));
    }
    let crop_width = f64::from(width.0) / f64::from(width.1);
    let crop_height = f64::from(height.0) / f64::from(height.1);
    let center_x =
        f64::from(horizontal.0) / f64::from(horizontal.1) + (f64::from(image.width) - 1.0) / 2.0;
    let center_y =
        f64::from(vertical.0) / f64::from(vertical.1) + (f64::from(image.height) - 1.0) / 2.0;
    let left = (center_x - (crop_width - 1.0) / 2.0).round().max(0.0) as u32;
    let top = (center_y - (crop_height - 1.0) / 2.0).round().max(0.0) as u32;
    let cropped = image.crop(
        left,
        top,
        crop_width.round() as u32,
        crop_height.round() as u32,
    );
    if cropped.width == 0 || cropped.height == 0 {
        return Err(failure("the clean aperture is empty"));
    }
    Ok(cropped)
}

#[cfg(test)]
mod tests {
    use super::super::{Decoded, decode};

    fn rgba(decoded: &Decoded, x: u32, y: u32) -> [u8; 4] {
        decoded.image.to_rgba8().get_pixel(x, y).0
    }

    fn near(actual: [u8; 4], expected: [u8; 4]) -> bool {
        actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 4)
    }

    /// Expected pixels are Pillow 12.3's (libavif 1.4.2 with dav1d), after
    /// its EXIF-orientation transpose.
    #[test]
    fn avif_odd_sizes_ten_bits_alpha_and_grids_decode_without_the_os() {
        // Red, green / magenta, blue quadrants of 121 by 81, 4:2:0, turned a
        // quarter clockwise by `irot` (generate-avif-quadrants.py).
        let turned = decode(include_bytes!(
            "../fixtures/heif/quadrants-121x81-orientation6.avif"
        ))
        .unwrap();
        assert_eq!((turned.image.width(), turned.image.height()), (81, 121));
        assert_eq!(
            (turned.info.format, turned.info.images, turned.info.primary),
            ("AVIF", 1, 0)
        );
        for ((x, y), color) in [
            ((20, 30), [255, 0, 255, 255]),
            ((60, 30), [255, 0, 0, 255]),
            ((20, 90), [0, 0, 255, 255]),
            ((60, 90), [0, 255, 0, 255]),
            // The last column and row of the odd-sized chroma planes.
            ((0, 0), [255, 0, 255, 255]),
            ((80, 120), [0, 255, 0, 255]),
        ] {
            assert!(
                near(rgba(&turned, x, y), color),
                "{x},{y}: {:?}",
                rgba(&turned, x, y)
            );
        }
        // 10-bit 12 by 34 with an alpha plane and `irot`; its `clop` and
        // `imor` are unknown boxes, not crop and mirror.
        let ten = decode(include_bytes!("../fixtures/heif/clop_irot_imor.avif")).unwrap();
        assert_eq!((ten.image.width(), ten.image.height()), (34, 12));
        assert_eq!(ten.info.images, 1);
        assert!(near(rgba(&ten, 0, 0), [0, 132, 0, 64]));
        assert!(near(rgba(&ten, 5, 5), [0, 132, 0, 66]));
        // A 4 by 3 grid of 10-bit tiles with a 4 by 3 grid alpha; the gain
        // map is not an image of its own.
        let grid = decode(include_bytes!(
            "../fixtures/heif/color_grid_alpha_grid_gainmap_nogrid.avif"
        ))
        .unwrap();
        assert_eq!((grid.image.width(), grid.image.height()), (512, 600));
        assert_eq!((grid.info.images, grid.info.primary), (1, 0));
        for ((x, y), color) in [
            ((0, 0), [0, 136, 0, 1]),
            ((100, 100), [195, 128, 205, 157]),
            ((300, 300), [91, 128, 85, 112]),
            ((10, 590), [198, 126, 208, 157]),
            ((511, 599), [255, 120, 255, 255]),
        ] {
            assert!(
                near(rgba(&grid, x, y), color),
                "{x},{y}: {:?}",
                rgba(&grid, x, y)
            );
        }
        let white = decode(include_bytes!("../fixtures/heif/white_1x1.avif")).unwrap();
        assert!(near(rgba(&white, 0, 0), [253, 253, 253, 255]));
        let circle = decode(include_bytes!(
            "../fixtures/heif/circle_custom_properties.avif"
        ))
        .unwrap();
        assert_eq!(rgba(&circle, 0, 0)[3], 0);
    }

    #[test]
    fn damaged_avif_is_an_error_not_an_empty_image() {
        let whole = include_bytes!("../fixtures/heif/quadrants-121x81-orientation6.avif");
        for cut in [whole.len() / 2, whole.len() - 40, whole.len() - 1] {
            assert!(decode(&whole[..cut]).is_err(), "cut {cut}");
        }
        // The AV1 payload (the `mdat` box ends the file) without its frame.
        let mut header_only = whole.to_vec();
        let end = header_only.len();
        header_only[end - 200..].fill(0);
        assert!(decode(&header_only).is_err());
    }
}
