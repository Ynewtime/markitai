# PaddleOCR models for the portable OCR engine

The portable local OCR engine (Windows, Linux, and macOS builds with the
`portable-media` feature) runs PaddleOCR's PP-OCR models in process with the
pure-Rust `tract` inference engine. The models are not part of any Markitai
executable, library or package. `markitai doctor --fix`, or the first OCR that
needs one, downloads each model file from the official mirror listed in
[`crates/markitai-core/src/ocr/paddle/models.json`](../../crates/markitai-core/src/ocr/paddle/models.json)
into `MARKITAI_HOME/models/ocr/<name>-<sha8>/`, and every load checks its
published size and SHA-256.

- **Models**: PaddleOCR, by the PaddlePaddle Authors, Apache License 2.0
  ([PaddleOCR-LICENSE](PaddleOCR-LICENSE), retrieved 2026-10-02 from
  `https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE`).
- **ONNX conversion and mirror**: RapidOCR, by RapidAI, Apache License 2.0
  ([RapidOCR-LICENSE](RapidOCR-LICENSE), retrieved 2026-10-02 from
  `https://raw.githubusercontent.com/RapidAI/RapidOCR/main/LICENSE`). The
  files are those RapidOCR 3.9.2 itself downloads (its `default_models.yaml`),
  from `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/`;
  the reference implementation's RapidOCR downloads the same detector,
  classifier and multilingual recognizer by default.

| Model (manifest name) | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `ppocrv6-det-small` | `PP-OCRv6_det_small.onnx` | 9,929,594 | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `ppocr-cls-mobile-v2` | `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | 585,532 | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |
| `ppocrv6-rec-small` | `PP-OCRv6_rec_small.onnx` | 21,234,383 | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `korean-ppocrv5-rec-mobile` | `korean_PP-OCRv5_rec_mobile.onnx` | 13,488,748 | `cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b` |
| `latin-ppocrv5-rec-mobile` | `latin_PP-OCRv5_rec_mobile.onnx` | 7,904,513 | `b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a` |
| `eslav-ppocrv5-rec-mobile` | `eslav_PP-OCRv5_rec_mobile.onnx` | 7,911,802 | `08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab` |
| `cyrillic-ppocrv5-rec-mobile` | `cyrillic_PP-OCRv5_rec_mobile.onnx` | 8,074,092 | `90f761b4bfcce0c8c561c0cb5c887b0971d3ec01c32164bdf7374a35b0982711` |
| `th-ppocrv5-rec-mobile` | `th_PP-OCRv5_rec_mobile.onnx` | 7,915,294 | `de541dd83161c241ff426f7ecfd602a0ba77d686cf3ab9a6c255ea82fd08006e` |
| `el-ppocrv5-rec-mobile` | `el_PP-OCRv5_rec_mobile.onnx` | 7,832,350 | `b4368bccd557123c702b7549fee6cd1e94b581337d1c9b65310f109131542b7f` |
| `arabic-ppocrv5-rec-mobile` | `arabic_PP-OCRv5_rec_mobile.onnx` | 8,023,828 | `c1192e632d0baa9146ae5b756a0e635e3dc63c1733737ebfd1629e87144e9295` |
| `devanagari-ppocrv5-rec-mobile` | `devanagari_PP-OCRv5_rec_mobile.onnx` | 7,940,361 | `d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e` |
| `ta-ppocrv5-rec-mobile` | `ta_PP-OCRv5_rec_mobile.onnx` | 7,909,926 | `a42448808b7dea87597336f12438935f40353f1949e8360acd9e06b4da21bfe1` |
| `te-ppocrv5-rec-mobile` | `te_PP-OCRv5_rec_mobile.onnx` | 7,923,102 | `a3690451b50028a09a3316a1274f7c05728151ea3f8fd392696397a7fefcbd92` |

The first four form the default set (`doctor --fix` installs them, about
45.2 MB); the others are downloaded only when `ocr.lang` names their language.
Every digest above equals the one RapidOCR 3.9.2 publishes for the file.

The inference engine itself (`tract-onnx` 0.23.8 and its `tract-*` crates,
MIT OR Apache-2.0) is compiled into the executable and is covered by the
dependency license inventory of each package. This record is mechanical
provenance, not a legal review.
