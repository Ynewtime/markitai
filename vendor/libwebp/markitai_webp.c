// markitai's entry point to libwebp (MIT, like the rest of markitai; libwebp
// itself is BSD-3-Clause, see COPYING).
//
// Lossy WebP encoding as Pillow's WebPEncode does it (README.md): libwebp's
// default configuration at `quality`, method 4, lossless alpha
// (alpha_quality 100), transparent pixels' colours not kept, ARGB input, no
// metadata chunks.
#include <stddef.h>
#include <stdint.h>

#include "src/webp/encode.h"

typedef int (*markitai_webp_sink)(const uint8_t* data, size_t size,
                                  void* context);

typedef struct {
  markitai_webp_sink sink;
  void* context;
} Writer;

static int Write(const uint8_t* data, size_t size, const WebPPicture* picture) {
  const Writer* const writer = (const Writer*)picture->custom_ptr;
  return writer->sink(data, size, writer->context);
}

// Encodes `width` x `height` pixels, RGBA when `alpha` is non-zero, else RGB,
// rows `stride` bytes apart, and hands the file to `sink` in pieces; a sink
// returning 0 stops the encoder. Returns 0, or libwebp's WebPEncodingError.
int markitai_webp_encode(const uint8_t* pixels, int width, int height,
                         int stride, int alpha, float quality,
                         markitai_webp_sink sink, void* context) {
  WebPConfig config;
  WebPPicture picture;
  Writer writer;
  int ok;
  if (!WebPConfigInit(&config) || !WebPPictureInit(&picture)) {
    return VP8_ENC_ERROR_INVALID_CONFIGURATION;
  }
  config.lossless = 0;
  config.quality = quality;
  config.alpha_quality = 100;
  config.method = 4;
  config.exact = 0;
  if (!WebPValidateConfig(&config)) return VP8_ENC_ERROR_INVALID_CONFIGURATION;
  picture.width = width;
  picture.height = height;
  picture.use_argb = 1;
  ok = alpha ? WebPPictureImportRGBA(&picture, pixels, stride)
             : WebPPictureImportRGB(&picture, pixels, stride);
  if (!ok) {
    const int error = picture.error_code;
    WebPPictureFree(&picture);
    return error ? error : VP8_ENC_ERROR_OUT_OF_MEMORY;
  }
  writer.sink = sink;
  writer.context = context;
  picture.writer = Write;
  picture.custom_ptr = &writer;
  ok = WebPEncode(&config, &picture);
  WebPPictureFree(&picture);
  return ok ? VP8_ENC_OK : picture.error_code;
}
