"""Validate and archive reviewed local upstream notices without network access.

A full text and a license notice are deliberately different states. Reviewed
historical originals can supply full terms while retaining historical provenance.
Hash checks do not establish the legal sufficiency of a distribution.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile

MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 32 * 1024 * 1024
MAX_FILES = 512


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("Duplicate license evidence JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("Invalid license evidence path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise RuntimeError("License evidence path escapes its directory")
    return path


def _read(root, name, expected=None):
    relative = _relative(name)
    path = root
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("License evidence root must be a real directory")
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise RuntimeError("License evidence must not contain symlinks")
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > MAX_FILE:
        raise RuntimeError("License evidence must be a bounded regular file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE + 1)
    if len(raw) > MAX_FILE:
        raise RuntimeError("License evidence file exceeds its bound")
    if expected is not None:
        size, digest = expected.get("bytes"), expected.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("Invalid license evidence identity")
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError("License evidence byte identity differs")
    return raw


def _repo(value):
    if not isinstance(value, str) or not value.startswith("https://github.com/"):
        raise RuntimeError("Unsupported upstream evidence repository")
    repo = value.removeprefix("https://github.com/").rstrip("/").removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise RuntimeError("Invalid upstream evidence repository")
    return repo



# Only these reviewed package identities may add externally hosted license terms.
_AUTHORITY_TERMS = {
    "Apache-2.0": {
        "url": "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "link": b"[Apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0",
        "repository": "madsmtm/objc2",
        "packages": {
            ("dispatch2", "0.3.1", "8852b424193ca41602281b3d7540d7c8ed51e49a"),
            ("objc2-core-foundation", "0.3.2", "7b1abfd750a2cacaea71d6a56ecfb83cb7de560b"),
            ("objc2-core-graphics", "0.3.2", "7b1abfd750a2cacaea71d6a56ecfb83cb7de560b"),
            ("objc2-image-io", "0.3.2", "7b1abfd750a2cacaea71d6a56ecfb83cb7de560b"),
            ("objc2-vision", "0.3.2", "7b1abfd750a2cacaea71d6a56ecfb83cb7de560b"),
        },
    },
    "MPL-2.0": {
        "url": "https://www.mozilla.org/media/MPL/2.0/index.f75d2927d3c1.txt",
        "sha256": "3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04",
        "link": b"https://mozilla.org/MPL/2.0/",
        "repository": "servo/stylo",
        "packages": {("selectors", "0.38.0", "572ecba2d1600e7c3d490586692a209faf703baa"),
                     ("selectors", "0.41.0", "34d25f741765a0bdb894f8271852e633299352d8")},
    },
}
_MIT_ORIGINAL_URL = "https://raw.githubusercontent.com/madsmtm/objc2/9961247c1a82027d6edbe6c516011b1363b9354c/LICENSE.txt"
_MIT_ORIGINAL_SHA = "e353f37b12aefbb9f9b29490e837cfee05d9bda70804b3562839a3285c1df1e5"
_MIT_TARGETS = {
    ("objc2", "0.6.4"): "8852b424193ca41602281b3d7540d7c8ed51e49a",
    ("objc2", "0.6.5"): "d7d2fa23ceaa5e6096c923b081040e5d81b3b9df",
    ("objc2-encode", "4.1.0"): "8d214f5477365ffcbcbb7de058c86ed9a518efb7",
    ("objc2-foundation", "0.3.2"): "7b1abfd750a2cacaea71d6a56ecfb83cb7de560b",
}
_MIT_PROOFS = {'https://github.com/madsmtm/objc2/commits/8d214f5477365ffcbcbb7de058c86ed9a518efb7/LICENSE.txt': '0a1efed6cc84aac1d931fe09facf2883ae2fb13caa4e131ddc3bf61aca0d9e3a', 'https://github.com/madsmtm/objc2/commit/cfb199226661ec6e4939fcd7cd7531b3c5453076.patch': 'b4ecc7d6db95a78e1cf777636049b4a9156783f326edf016a1461b24c84d1872', 'https://github.com/madsmtm/objc2/commit/9961247c1a82027d6edbe6c516011b1363b9354c.patch': '13127f39976ceecdadb86870d2448867fba8601ad21a376724f242fbd67e86fb', 'https://github.com/madsmtm/objc2/commits/8852b424193ca41602281b3d7540d7c8ed51e49a/LICENSE.txt': 'cce3f980b9088bf6533102997510603f50a5fbdda6fd34c055f797777fc20e0b', 'https://github.com/madsmtm/objc2/commits/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.txt': 'ab00bc3db74112a44b17ac776fab3a16bc6122ffc8ccf2708a80e5d96206b4de', 'https://github.com/madsmtm/objc2/commits/d7d2fa23ceaa5e6096c923b081040e5d81b3b9df/LICENSE.txt': '21a492cb3a3a28562fcdcf08ba386c60f21937d28db7697dbb414294ff95f907'}
_SELECTORS_ARCHIVE_SHA = "8adfa1c298912827b8a28b223b3b874357397ae706e6190acd9bf28cee99114d"
# Published source and the one version-stamp exception are fixed review facts,
# never a general permission to normalize manifests or accept dirty sources.
_SOURCE_PACKAGES = {
    ("selectors", "0.38.0", "572ecba2d1600e7c3d490586692a209faf703baa"): {
        "sha256": _SELECTORS_ARCHIVE_SHA, "path_in_vcs": "selectors", "members": 22,
    },
    ("selectors", "0.41.0", "34d25f741765a0bdb894f8271852e633299352d8"): {
        "sha256": "8322aebea94a6cc288ff884e8a49bdbceef60206c400c6c46d22e62fc4fd91a6",
        "path_in_vcs": "selectors", "members": 21,
    },
    ("nom-language", "0.1.0", "2cec1b3e4c9ccac62c902d60c00de6d1549ccbe1"): {
        "sha256": "2de2bc5b451bfedaef92c90b8939a8fff5770bdcc1fafd6239d086aab8fa6b29",
        "path_in_vcs": "nom-language", "members": 7,
        "repository": "rust-bakery/nom", "declared_license": "MIT",
        "rust": {
            "src/error.rs": "e0df3aecd5d1bce88d5a3ace27653d61dac66c4d1adffc5f8f2dd14872d3daf0",
            "src/lib.rs": "67d772d35d00ac350cc12496aaa8badc6c51b64ae1cba495d803437e4f6300b1",
            "src/precedence/mod.rs": "c9f56d9c2256e43af883f773b9065372f6fa95137030eb66a90c3e236c6c92d8",
            "src/precedence/tests.rs": "bec5f416d2e300e664ba1d974d3fdc1e731fca98d87cb32d9d7b29b462df14ec",
        },
        "terms": {"LICENSE": "4dbda04344456f09a7a588140455413a9ac59b6b26a1ef7cdf9c800c012d87f0"},
    },
    ("tract-extra", "0.23.8", "248335349c0f59a5772d737480b8b39250097374"): {
        "sha256": "2a05c86ece0c47880c7726e08dabb72d68e9cba0cd36d44bc0aa90e2291a05db",
        "path_in_vcs": "extra", "members": 6,
        "repository": "sonos/tract", "declared_license": "MIT OR Apache-2.0",
        "rust": {
            "src/exp_unit_norm.rs": "47a3ccdb9e03c4187a0789b624a5d5e2e83ba5833d90cac7d8970afcab4a54c2",
            "src/lib.rs": "6067a911c32c79b2310351fa79a82bced196460a012e61c30efd623952dcc245",
        },
        "terms": {
            "LICENSE": "f7ef673bf046d823dcd775bdd0768432bd8855f81d0e5e1290a0a48c42e2dca3",
            "LICENSE-MIT": "23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3",
            "LICENSE-APACHE": "a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2",
        },
    },
}
# Exact new Node-binding tuples. These children declare MIT directly;
# they do not inherit workspace.package.license. Archives retain published
# VCS/original manifests while fixed parent bytes bind the complete notice.
_SOURCE_PACKAGES.update({
    ('napi', '3.14.0', '37109f133043aff379a38edfb8b1548e7e9309e5'): {'sha256': '15301c22171e8397a7b17beb3c75bf486bf67ca3bdad8ea347835b770df61969', 'path_in_vcs': 'crates/napi', 'members': 88, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': '046046115eecec251c9eaed985e0f5cc36815f6660adccd9410dfc28d1bb4f35', 'vcs_sha256': '39e3cd2d05c1dc6eb219f18bbff425b6ab3ec8d4cb52edde206e8698279881b4'},
    ('napi-build', '2.6.0', 'a94cef33f62e74682963dd085e283f52dcec2600'): {'sha256': 'b899b545d3aa6dca985939059f258c5488d34e4ecf39c274e20009748f4b846d', 'path_in_vcs': 'crates/build', 'members': 11, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': 'db03f6809e261fef67c4fad525bb43e7474c393845b7a129956dbac21cfed148', 'vcs_sha256': 'ab0db254baa56192392dfa8a058f15a6e6d306194faeb4858b7cbb6f71276685'},
    ('napi-derive', '3.6.10', '37109f133043aff379a38edfb8b1548e7e9309e5'): {'sha256': 'd11174f7507d7f6cdd69874f541eff194e81c95652b775e69036bcf8d5b7a805', 'path_in_vcs': 'crates/macro', 'members': 16, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': 'b35f32f10608df47a94f28d391c0b5d0cd88c3b51823131e07efa2cc028f6ad3', 'vcs_sha256': '4dc7f324179df88dfeef2476de23487cc069d14b2cef9881e7e5dafb363be64e'},
    ('napi-sys', '3.4.0', '37109f133043aff379a38edfb8b1548e7e9309e5'): {'sha256': 'e22a4f25c16a5c5411d987cd6fbd48313522ae1789ade2d6dd3efdc6d40a0fc8', 'path_in_vcs': 'crates/sys', 'members': 11, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': 'fd25505b1dcd4c6886df2f42e5072af24da3a8b026bc0a202f680a07dbf18c02', 'vcs_sha256': 'e83464a4c378004e398379bcf42e57189e8840f671bc3ed57d12917a7d4dd9f1'},
    # The same parent and LICENSE bytes at the commit napi 3.15 was published from.
    ('napi', '3.15.0', 'd9cb526c15e63ba30e06b4d169068c9854180595'): {'sha256': '6a746ad50efbfb632ade1de4025519a477decbbd798cea1a6ac41d7c44f80083', 'path_in_vcs': 'crates/napi', 'members': 89, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': 'ed67c43de5350454692d7e279e2dddcb4520a023214a01fe4ef0aac8d3b674ef', 'vcs_sha256': '8f417bbb7137f0c1d8b45d31d8a9764ce9424639c38250f84ffb0679ecc4ebfd'},
    ('napi-derive', '3.6.13', 'd9cb526c15e63ba30e06b4d169068c9854180595'): {'sha256': '3ff48271a0f61abb83f3e2d1989fa71169143db95f6f948c3a8903961f2855dc', 'path_in_vcs': 'crates/macro', 'members': 16, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': '84f5ae907e86619814aad37c03c22253a547e12fd2cc3d8c500b26c2aedc5a95', 'vcs_sha256': '9af016540388a0b6e0d14bebc10fff6951ac43412cf878738ae47607cdc75124'},
    ('napi-sys', '3.4.1', 'd9cb526c15e63ba30e06b4d169068c9854180595'): {'sha256': 'd153beb7d5c973f4b9395be8a83054a28968b61188f1dd1a2cc2321af22daea2', 'path_in_vcs': 'crates/sys', 'members': 11, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': '82b7eef9b33624a95256db45c4736805c2a77ae37319c34a4b0286c87694ae10', 'vcs_sha256': '2a3eb2bd1135dbf07a332c93f714fc0ccf768187c7ece05d35009c3ccebb54fc'},
    ('napi-derive-backend', '6.1.6', 'd9cb526c15e63ba30e06b4d169068c9854180595'): {'sha256': 'f2639052c5aff9956a21d7b6202fc61d5e89c5f19c72d289b7a0641ea94e8159', 'path_in_vcs': 'crates/backend', 'members': 22, 'repository': 'napi-rs/napi-rs', 'declared_license': 'MIT', 'terms': {'LICENSE': '3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215'}, 'parent_manifest_sha256': 'f2110059925ada6800f0d3e8a7976f6967803686d6a87d68f2c67ce5974b6261', 'package_manifest_sha256': '310cae4da3ac464443046c8acb5e563e60da2b5ab502f824dfd07f8c65c14772', 'vcs_sha256': '35c192e47381e0c1a3f74c4231ae9a7b01aa5051c564f9b31bae5c4127334dce'},
})
# Lock-refresh tuples of 2026-10-23. jni alone inherits its license from the
# pinned parent workspace manifest; objc2 takes the reviewed historical MIT text.
_SOURCE_PACKAGES.update({
    ('alloc-stdlib', '0.3.0', '0a81fd6928ea3b33c8cd484aa4575d50ffb98012'): {'sha256': '0b5c1865780388bfa186411ab5f247819487fc4864c6e9c3106611fa347586e1', 'path_in_vcs': 'alloc-stdlib', 'members': 12, 'repository': 'dropbox/rust-alloc-no-stdlib', 'declared_license': 'BSD-3-Clause', 'terms': {'LICENSE': 'c0c56f26d9c051cac4d200c34c84e7ae9aaa853e01a982a1df08b09931e518ae'}, 'package_manifest_sha256': '13c6e8d5ec2e75ae6b66542498e049e1100e71ef25e29ccb9cb083b40d6279ea', 'vcs_sha256': '2bfc044805e0e75273a7e4d06c9d72f4e029a6cb926e4645c382bde160ecfd95'},
    ('jni', '0.22.4', '5ae9458a4ec44c5318f37ddc7569c1d4ae8a69e7'): {'sha256': '5efd9a482cf3a427f00d6b35f14332adc7902ce91efb778580e180ff90fa3498', 'path_in_vcs': 'crates/jni', 'members': 76, 'repository': 'jni-rs/jni-rs', 'declared_license': 'MIT OR Apache-2.0', 'terms': {'LICENSE-MIT': 'fea1d5bf3dd71605ce5d7d2ff695c1837e914c77195e523a86b5391716477960', 'LICENSE-APACHE': 'a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2'}, 'license_inherited_from_workspace': True, 'parent_manifest_sha256': '83fb01ea19de610d7c177f50959f5b76ae1cf83571a838aff43dc4914af870bc', 'package_manifest_sha256': '420d0c6c04c3579946386ceea541bf5dd5f564b6b190e76568cce83bad0b9db3', 'vcs_sha256': '0619bb2725fac7b26a1935d1379d5eea7427eb987731a982500389de5efd5b8b'},
    ('jni-macros', '0.22.4', '33045a124105c939d1e2cbdcb5a39e5d868ffa03'): {'sha256': 'a00109accc170f0bdb141fed3e393c565b6f5e072365c3bd58f5b062591560a3', 'path_in_vcs': 'crates/jni-macros', 'members': 14, 'repository': 'jni-rs/jni-rs', 'declared_license': 'MIT OR Apache-2.0', 'terms': {'LICENSE-MIT': 'fea1d5bf3dd71605ce5d7d2ff695c1837e914c77195e523a86b5391716477960', 'LICENSE-APACHE': 'a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2'}, 'parent_manifest_sha256': '1d950cf4f7fb6584187a3705c37611f9b5b5cd1cdcaf3a3326a4b60abc29aff5', 'package_manifest_sha256': 'cfa3ca54bc23baa582a42df863f397debe8c4312a80ead221e8e049b834f93b5', 'vcs_sha256': '5b271175fb5d736bf6e71f696a8829f37c6c00391065111526d7e43fdeb537e1'},
    ('jni-sys-macros', '0.4.1', '64d77b7a5f119d7b55b4e2c169a4668067ff59e6'): {'sha256': '38c0b942f458fe50cdac086d2f946512305e5631e720728f2a61aabcd47a6264', 'path_in_vcs': 'jni-sys-macros', 'members': 6, 'repository': 'jni-rs/jni-sys', 'declared_license': 'MIT OR Apache-2.0', 'terms': {'LICENSE-MIT': '1d85bd754b04ceec93e98e890edd1fa3c6a22e81bcb32135806beeccefa51cd1', 'LICENSE-APACHE': 'c6596eb7be8581c18be736c846fb9173b69eccf6ef94c5135893ec56bd92ba08'}, 'parent_manifest_sha256': '193ab6fe89ccbbed7ea68c3bf611cf098253e811824c7079fba6979737fff1e7', 'package_manifest_sha256': '6501ef93eff9b37ab65ec846bd06992d79cc28d8ef1f02f13b0488ff441ed90a', 'vcs_sha256': '1824b83bc25a1e245bcb97a6503105bfe07078a47cddc924c6b7db91e8852f45'},
    ('objc2', '0.6.5', 'd7d2fa23ceaa5e6096c923b081040e5d81b3b9df'): {'sha256': '08849bbd4767dfae9457696856ae1c84fe4e0281bbe4a7abff2d0e06fb7981f8', 'path_in_vcs': 'crates/objc2', 'members': 120, 'repository': 'madsmtm/objc2', 'declared_license': 'MIT', 'parent_manifest_sha256': 'f93fce3a2814c16597140137b109c53042c0ae74909fd123a8a4d1f7d35090da', 'package_manifest_sha256': '4a1e1561612efc4a8aea4dd574c781f6215fa5416696425227558fc782353cba', 'vcs_sha256': '5f7c8e095d3933c477ff1f71021b551cbfe497a62346a1dc8615debd9846e257'},
    ('rustls-platform-verifier-android', '0.2.0', '252e25161a91af476cbea620e29d277592d27ec7'): {'sha256': 'eec689c0bc40ff2458a5977b6619cb718087084a18e02a131c599b62d05e1a5f', 'path_in_vcs': 'android-release-support', 'members': 5, 'repository': 'rustls/rustls-platform-verifier', 'declared_license': 'MIT OR Apache-2.0', 'terms': {'LICENSE-MIT': '1c7cf76689c837a68ed8d704994e52a0f2940c087958f860d17f3186afbdcc0c', 'LICENSE-APACHE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'}, 'parent_manifest_sha256': '5c47afb1a10db977965be6be81fe4e17a5fdea95a506dfc5cfe8daf5f5708fc1', 'package_manifest_sha256': 'e78f460d378e8fbf3cb90e96a249c8d23cd7cfc97e83cd7b95d979472de4adad', 'vcs_sha256': 'f9de66ae77438abeb19e82916812a0266cfdbcd6a8b1ca0404f5f167daf0a43e'},
})
_TRACT_IDENTITY = ("tract-extra", "0.23.8", "248335349c0f59a5772d737480b8b39250097374")
_TRACT_STAMP = {"kind": "reviewed_publication_version_stamp", "upstream_version": "0.23.8-pre",
                "published_version": "0.23.8", "legal_review": "not_performed"}


def _same_commit_source(asset, data, root, repo, commit):
    source_path = str(_relative(asset["source_path"]))
    if not source_path.startswith(root) or source_path not in data:
        raise RuntimeError("License source is outside the attributed commit")
    expected_url = "https://raw.githubusercontent.com/" + repo + "/" + commit + "/" + source_path[len(root):]
    full = data[source_path]
    if asset["source_url"] != expected_url or hashlib.sha256(full).hexdigest() != asset["source_sha256"]:
        raise RuntimeError("License source URL or hash differs")
    bounds = asset.get("source_byte_range")
    if bounds is not None:
        start, end = bounds.get("start_inclusive"), bounds.get("end_exclusive")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(full):
            raise RuntimeError("Invalid license notice byte range")
        full = full[start:end]
    return full


def _linked_terms(asset, entry, data, recorded, root, repo, commit):
    option = asset.get("license_option")
    authority = _AUTHORITY_TERMS.get(option)
    identity = (entry["name"], entry["version"], commit)
    if (authority is None or repo != authority["repository"] or identity not in authority["packages"]
            or option not in entry["declared_license"].split(" OR ")
            or entry["complete_license_options_present"] != [option]
            or asset["content_kind"] != "full_license_text"):
        raise RuntimeError("Unreviewed linked license authority or package")
    notices = [row for row in entry["assets"] if row["path"] == asset.get("notice_asset")]
    if (len(notices) != 1 or notices[0].get("source_kind", "same_commit") != "same_commit"
            or notices[0]["content_kind"] != "notice_only"):
        raise RuntimeError("Linked license requires its exact-version notice")
    notice = recorded(notices[0])
    if notice != _same_commit_source(notices[0], data, root, repo, commit):
        raise RuntimeError("Linked license notice differs from upstream source bytes")
    if authority["link"] not in notice:
        raise RuntimeError("Exact-version notice does not link this license authority")
    source_path = str(_relative(asset["source_path"]))
    raw = data.get(source_path)
    if (asset["source_url"] != authority["url"] or asset["source_sha256"] != authority["sha256"]
            or raw is None or hashlib.sha256(raw).hexdigest() != authority["sha256"]):
        raise RuntimeError("Linked authority URL or reviewed text hash differs")
    return raw


def _historical_terms(asset, entry, data, recorded, repo, commit):
    if (repo != "madsmtm/objc2" or _MIT_TARGETS.get((entry["name"], entry["version"])) != commit
            or entry["declared_license"] != "MIT" or entry["content_kind"] != "full_license_text"
            or entry["complete_license_options_present"] != ["MIT"]
            or asset.get("source_kind") != "historical_notice"
            or asset.get("provenance_verified") is not True
            or asset.get("legal_review") != "not_performed"
            or asset.get("content_kind") != "full_license_text"):
        raise RuntimeError("Historical license terms require their reviewed scope and provenance")
    # The current pinned release still declares MIT. Keeping its whole original
    # notice also preserves the upstream Apple SDK discussion without resolving it.
    root = f"upstream/{repo}/{commit}/"
    notices = [row for row in entry["assets"]
               if row.get("source_kind", "same_commit") == "same_commit"
               and row.get("content_kind") == "notice_only"]
    if len(notices) != 1:
        raise RuntimeError("Historical terms require their exact-version MIT notice")
    notice = recorded(notices[0])
    if (notice != _same_commit_source(notices[0], data, root, repo, commit)
            or hashlib.sha256(notice).hexdigest() != "7f976f7e9cb2d87df7230606feb932c3f21ac0e664045a775b600046ff850c54"):
        raise RuntimeError("Historical terms current-version MIT notice differs")
    declared = data.get(root + entry["path_in_vcs"] + "/Cargo.toml", b"")
    if re.search(rb'^license[ \t]*=[ \t]*"MIT"[ \t]*(?:#[^\r\n]*)?$', declared, re.MULTILINE) is None:
        raise RuntimeError("Historical terms require the current package MIT declaration")
    source_path = str(_relative(asset["source_path"]))
    raw = data.get(source_path)
    if (asset["source_url"] != _MIT_ORIGINAL_URL or asset["source_sha256"] != _MIT_ORIGINAL_SHA
            or raw is None or hashlib.sha256(raw).hexdigest() != _MIT_ORIGINAL_SHA):
        raise RuntimeError("Historical original URL or reviewed text hash differs")
    expected = {
        "https://github.com/madsmtm/objc2/commits/" + commit + "/LICENSE.txt",
        "https://github.com/madsmtm/objc2/commit/cfb199226661ec6e4939fcd7cd7531b3c5453076.patch",
        "https://github.com/madsmtm/objc2/commit/9961247c1a82027d6edbe6c516011b1363b9354c.patch",
    }
    proofs = asset.get("proofs", [])
    if type(proofs) is not list or len(proofs) != 3 or {p.get("source_url") for p in proofs} != expected:
        raise RuntimeError("Historical supplement requires its target-anchored provenance")
    for proof in proofs:
        body = recorded(proof)
        if hashlib.sha256(body).hexdigest() != _MIT_PROOFS[proof["source_url"]]:
            raise RuntimeError("Historical provenance differs from reviewed upstream bytes")
    return raw


def _source_archives(manifest, data, recorded):
    records = manifest.get("source_archives", [])
    if type(records) is not list or len(records) > len(_SOURCE_PACKAGES):
        raise RuntimeError("Unreviewed source archive inventory")
    expected = {(p["name"], p["version"], p["commit"]) for p in manifest["packages"]} & _SOURCE_PACKAGES.keys()
    result, sources = [], {}
    for row in records:
        raw = recorded(row)
        identity = (row.get("package"), row.get("version"), row.get("commit"))
        reviewed = _SOURCE_PACKAGES.get(identity)
        label = f"{row.get('package')}-{row.get('version')}"
        if (reviewed is None or identity not in expected or identity in sources
                or row.get("path") != f"source-archives/{label}.crate"
                or row.get("source_url") != f"https://static.crates.io/crates/{row.get('package')}/{label}.crate"
                or hashlib.sha256(raw).hexdigest() != reviewed["sha256"]):
            raise RuntimeError("Source archive is not the reviewed exact package")
        # Fixed compressed digest bounds all archive members. Never extract paths.
        members, total = {}, 0
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            for member in archive:
                name = str(_relative(member.name))
                if (not member.isreg() or not name.startswith(label + "/")
                        or name in members or len(members) >= reviewed["members"]
                        or not 0 <= member.size <= MAX_FILE):
                    raise RuntimeError("Source archive has unreviewed members")
                total += member.size
                if total > MAX_TOTAL:
                    raise RuntimeError("Source archive exceeds its bound")
                body = archive.extractfile(member).read(MAX_FILE + 1)
                if len(body) != member.size:
                    raise RuntimeError("Source archive member size differs")
                members[name] = body
        original = f"local-evidence/{label}/"
        if (len(members) != reviewed["members"]
                or any(members.get(label + "/" + name) != data.get(original + name)
                       for name in [".cargo_vcs_info.json", "Cargo.toml.orig"])):
            raise RuntimeError("Source archive version differs from license evidence")
        vcs = _json(members[label + "/.cargo_vcs_info.json"])
        if vcs.get("git", {}).get("sha1") != row["commit"] or vcs.get("path_in_vcs") != reviewed["path_in_vcs"]:
            raise RuntimeError("Source archive VCS differs from reviewed package")
        for name, digest in reviewed.get("rust", {}).items():
            body = members.get(label + "/" + name)
            upstream = f"upstream/{reviewed['repository']}/{row['commit']}/{reviewed['path_in_vcs']}/{name}"
            if body is None or hashlib.sha256(body).hexdigest() != digest or body != data.get(upstream):
                raise RuntimeError("Reviewed published Rust source differs from upstream bytes")
        sources[identity] = members
        result.append(row)
    if set(sources) != expected:
        raise RuntimeError("Reviewed source archive inventory is incomplete")
    return result, sources


def _manifest_provenance(entry, repo, vcs, upstream, published, sources):
    identity = (entry["name"], entry["version"], entry["commit"])
    if upstream == published and identity != _TRACT_IDENTITY:
        if (entry.get("publication_version_stamp") is not None
                or entry.get("package_manifest_matches_published_original") is not True):
            raise RuntimeError("Raw manifest match has inconsistent provenance")
        return "raw_exact_match"
    if (identity != _TRACT_IDENTITY or repo != "sonos/tract" or entry["path_in_vcs"] != "extra"
            or entry["id"] != "registry+https://github.com/rust-lang/crates.io-index#tract-extra@0.23.8"
            or entry["declared_license"] != "MIT OR Apache-2.0"
            or entry.get("package_manifest_matches_published_original") is not False
            or entry.get("publication_version_stamp") != _TRACT_STAMP
            or vcs.get("git", {}).get("dirty") is not True or identity not in sources
            or hashlib.sha256(upstream).hexdigest() != "c3b1bd149734cb869157c50391a213a5435debecd88062e9dfe8dd44d724c990"
            or hashlib.sha256(published).hexdigest() != "b6027b969552e2f7bffa346c0dfcd908c37659595ca22e70279b23a7ea7cc04c"
            or upstream.count(b'\nversion = "0.23.8-pre"\n') != 1
            or upstream.replace(b'\nversion = "0.23.8-pre"\n', b'\nversion = "0.23.8"\n') != published):
        raise RuntimeError("Upstream package manifest differs from published original outside reviewed stamp")
    return "reviewed_publication_version_stamp"


def stage_overlay(source, destination, packages):
    """Return matching package text records; copy the complete verified evidence.

    Missing optional overlays leave the original source-only collector intact.
    A present but corrupt overlay fails closed before any evidence is copied.
    """
    source, destination = Path(source), Path(destination)
    if not source.exists() and not source.is_symlink():
        return {"packages": {}, "record": {"status": "absent"}}
    raw_inventory = _read(source, "inventory.json")
    inventory = _json(raw_inventory)
    if inventory.get("schema") != 1 or type(inventory.get("files")) is not list or len(inventory["files"]) > MAX_FILES:
        raise RuntimeError("Invalid license evidence inventory")
    data, identities, total = {}, {}, 0
    for entry in inventory["files"]:
        name = str(_relative(entry["path"]))
        if name in data or name == "inventory.json":
            raise RuntimeError("Duplicate or recursive license evidence inventory")
        data[name] = _read(source, name, entry)
        identities[name] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        total += len(data[name])
        if total > MAX_TOTAL:
            raise RuntimeError("License evidence total exceeds its bound")
    observed = set()
    for index, path in enumerate(source.rglob("*")):
        if index > MAX_FILES * 4 or path.is_symlink():
            raise RuntimeError("Unexpected or excessive license evidence entries")
        if path.is_file():
            observed.add(path.relative_to(source).as_posix())
        elif not path.is_dir():
            raise RuntimeError("License evidence contains a special file")
    if observed != set(data) | {"inventory.json"}:
        raise RuntimeError("License evidence inventory is not exact")

    def recorded(entry):
        name = str(_relative(entry["path"]))
        if name not in data or identities[name] != {key: entry[key] for key in ["bytes", "sha256"]}:
            raise RuntimeError("License manifest identity differs from its inventory")
        return data[name]

    manifest = _json(data["manifest.json"])
    if manifest.get("schema") != 1 or type(manifest.get("packages")) is not list or len(manifest["packages"]) > MAX_FILES:
        raise RuntimeError("Invalid license overlay manifest")
    recorded(manifest["input_manifest"])
    if "license_gap_collection" in manifest:
        recorded(manifest["license_gap_collection"])
    if "napi_license_collection" in manifest:
        recorded(manifest["napi_license_collection"])
    source_archives, published_sources = _source_archives(manifest, data, recorded)
    available = {package["id"]: package for package in packages}
    matched, all_ids, all_assets = {}, set(), set()
    exact_matches, publication_stamps = 0, 0
    for entry in manifest["packages"]:
        key = entry["id"]
        if key in all_ids:
            raise RuntimeError("Duplicate license overlay package")
        all_ids.add(key)
        repo, commit = _repo(entry["repository"]), entry["commit"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError("License overlay requires an exact upstream commit")
        path = entry["path_in_vcs"]
        if path:
            _relative(path)
        root = f"upstream/{repo}/{commit}/"
        original = f"local-evidence/{entry['name']}-{entry['version']}/"
        upstream_manifest = root + (path + "/" if path else "") + "Cargo.toml"
        for proof in entry["evidence"]:
            recorded(proof)
        if any(name not in data for name in [original + ".cargo_vcs_info.json",
                                              original + "Cargo.toml.orig", upstream_manifest]):
            raise RuntimeError("Missing exact-version package evidence")
        vcs = _json(data[original + ".cargo_vcs_info.json"])
        if vcs.get("git", {}).get("sha1") != commit or vcs.get("path_in_vcs") != path:
            raise RuntimeError("License overlay VCS attribution differs")
        provenance = _manifest_provenance(entry, repo, vcs, data[upstream_manifest],
                                          data[original + "Cargo.toml.orig"], published_sources)
        exact_matches += provenance == "raw_exact_match"
        publication_stamps += provenance == "reviewed_publication_version_stamp"
        reviewed = _SOURCE_PACKAGES.get((entry["name"], entry["version"], commit), {})
        pinned = [(root + "Cargo.toml", "parent_manifest_sha256"),
                  (upstream_manifest, "package_manifest_sha256"),
                  (original + ".cargo_vcs_info.json", "vcs_sha256")]
        if any(field in reviewed and hashlib.sha256(data.get(name, b"")).hexdigest() != reviewed[field]
               for name, field in pinned):
            raise RuntimeError("Reviewed parent/package/VCS byte identity differs")
        if "terms" in reviewed:
            if (repo != reviewed["repository"] or path != reviewed["path_in_vcs"]
                    or key != f"registry+https://github.com/rust-lang/crates.io-index#{entry['name']}@{entry['version']}"
                    or entry["declared_license"] != reviewed["declared_license"]
                    or entry.get("license_inherited_from_workspace")
                    is not reviewed.get("license_inherited_from_workspace", False)
                    or entry.get("version_inherited_from_workspace") is not False
                    or entry["complete_license_options_present"] != reviewed["declared_license"].split(" OR ")
                    or len(entry["assets"]) != len(reviewed["terms"])
                    or {asset.get("source_path") for asset in entry["assets"]}
                    != {root + name for name in reviewed["terms"]}):
                raise RuntimeError("Reviewed parent license package scope differs")
            for asset in entry["assets"]:
                name = asset["source_path"][len(root):]
                if (asset.get("source_kind", "same_commit") != "same_commit"
                        or asset.get("source_byte_range") is not None
                        or asset.get("source_sha256") != reviewed["terms"][name]
                        or hashlib.sha256(data.get(root + name, b"")).hexdigest() != reviewed["terms"][name]
                        or asset.get("content_kind") != ("notice_only" if name == "LICENSE" and entry["name"] == "tract-extra" else "full_license_text")):
                    raise RuntimeError("Reviewed parent original terms hash or classification differs")
        kind = entry["content_kind"]
        options = entry["complete_license_options_present"]
        if kind not in {"full_license_text", "notice_only"} or type(options) is not list or any(not isinstance(value, str) or not value for value in options):
            raise RuntimeError("Invalid license text classification")
        if (kind == "full_license_text") != bool(options):
            raise RuntimeError("License notice cannot claim a complete license option")
        texts = []
        has_full_text = False
        for asset in entry["assets"]:
            raw = recorded(asset)
            asset_kind = asset.get("content_kind")
            if asset["path"] in all_assets or asset_kind not in {"full_license_text", "notice_only"}:
                raise RuntimeError("Duplicate or inconsistent upstream license asset")
            all_assets.add(asset["path"])
            source_kind = asset.get("source_kind", "same_commit")
            if source_kind == "same_commit":
                full = _same_commit_source(asset, data, root, repo, commit)
            elif source_kind == "linked_authority":
                full = _linked_terms(asset, entry, data, recorded, root, repo, commit)
            else:
                raise RuntimeError("Unreviewed license source kind")
            if full != raw:
                raise RuntimeError("License asset differs from its upstream source bytes")
            has_full_text |= asset_kind == "full_license_text"
            texts.append({"source": str(source / asset["path"]),
                          "path": (destination / asset["path"]).relative_to(destination.parent.parent).as_posix(),
                          **identities[asset["path"]], "origin": "verified_upstream_overlay",
                          "source_url": asset["source_url"], "source_kind": source_kind,
                          "content_kind": asset_kind})
        if not texts:
            raise RuntimeError("License classification differs from complete source texts")
        supplements = entry.get("supplemental", [])
        if type(supplements) is not list or len(supplements) > 1:
            raise RuntimeError("Unreviewed historical supplement inventory")
        for asset in supplements:
            raw = recorded(asset)
            if asset["path"] in all_assets:
                raise RuntimeError("Duplicate historical license supplement")
            all_assets.add(asset["path"])
            if _historical_terms(asset, entry, data, recorded, repo, commit) != raw:
                raise RuntimeError("Historical supplement differs from original source bytes")
            texts.append({"source": str(source / asset["path"]),
                          "path": (destination / asset["path"]).relative_to(destination.parent.parent).as_posix(),
                          **identities[asset["path"]], "origin": "verified_historical_license_text",
                          "source_url": asset["source_url"], "source_kind": "historical_notice",
                          "content_kind": "full_license_text", "provenance_verified": True,
                          "legal_review": "not_performed"})
            has_full_text = True
        if has_full_text != (kind == "full_license_text"):
            raise RuntimeError("License classification differs from complete source texts")
        package = available.get(key)
        if package is None:
            continue
        if any(package.get(field) != entry[field] for field in ["name", "version"]) or package.get("license") != entry["declared_license"] or _repo(package.get("repository")) != repo:
            raise RuntimeError("License overlay package metadata differs")
        directory = Path(package["manifest_path"]).parent
        for name in [".cargo_vcs_info.json", "Cargo.toml.orig"]:
            if _read(directory, name) != data[original + name]:
                raise RuntimeError("Current package source differs from exact-commit evidence")
        for name in reviewed.get("rust", {}):
            if _read(directory, name) != published_sources[(entry["name"], entry["version"], commit)][f"{entry['name']}-{entry['version']}/{name}"]:
                raise RuntimeError("Current package Rust source differs from reviewed publication")
        matched[key] = {"texts": texts, "complete_text": kind == "full_license_text",
                        "content_kind": kind, "commit": commit,
                        "full_text_gap": entry["unresolved_full_text_reason"],
                        "historical_provenance": bool(supplements),
                        "manifest_provenance": provenance}
    entries = manifest["packages"]
    current_unresolved = [entry["id"] for entry in entries
                          if entry["content_kind"] != "full_license_text"
                          or entry["unresolved_full_text_reason"] is not None]
    if manifest.get("unresolved_full_license_text") != current_unresolved:
        raise RuntimeError("Current license unresolved summary differs from its packages")
    expected_summary = {
        "requested_packages": len(entries),
        "exact_commit_manifest_matches": exact_matches,
        "full_license_text_packages": sum(entry["content_kind"] == "full_license_text" for entry in entries),
        "notice_only_packages": sum(entry["content_kind"] == "notice_only" for entry in entries),
        "overlay_files": len(all_assets),
        "source_archives": len(source_archives),
        "historical_text_packages": sum(bool(entry.get("supplemental")) for entry in entries),
    }
    summary = manifest.get("summary")
    # The retained pre-R49 overlay legitimately has zero stamps and no new
    # summary field. A stamp can never disappear into the raw-exact count.
    if publication_stamps or (type(summary) is dict and "reviewed_publication_version_stamps" in summary):
        expected_summary["reviewed_publication_version_stamps"] = publication_stamps
    if (type(summary) is not dict
            or any(type(summary.get(key)) is not int or summary[key] != value
                   for key, value in expected_summary.items())):
        raise RuntimeError("Current license count summary differs from its packages")
    # All provenance and classification checks precede copying. The caller's
    # clean-source checks also bind the inventory itself to the build revision.
    data["inventory.json"] = raw_inventory
    for name, raw in sorted(data.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    return {"packages": matched, "record": {
        "status": "verified", "manifest": {"path": (destination / "manifest.json").relative_to(destination.parent.parent).as_posix(), **identities["manifest.json"]},
        "matched_packages": len(matched),
        "complete_text_packages": sum(value["complete_text"] for value in matched.values()),
        "notice_only_packages": sum(not value["complete_text"] for value in matched.values()),
        "historical_text_packages": sum(value["historical_provenance"] for value in matched.values()),
        "exact_commit_manifest_matches": exact_matches,
        "reviewed_publication_version_stamps": publication_stamps,
        "legal_review": "not_performed",
        "source_archives": [{**row, "path": (destination / row["path"]).relative_to(destination.parent.parent).as_posix()} for row in source_archives],
        "archived_evidence_files": len(data),
        "scope": "Reviewed exact-version declarations and full terms, including explicitly identified historical originals; byte provenance and delivery checks only, no legal review"}}


def upstream_files(root):
    """Return validated offline evidence for CLI, wheel and npm archive payloads.

    The source snapshot supplies version provenance here. Static-Go packaging
    additionally passes the resolved Cargo packages to stage_overlay.
    """
    source = Path(root) / "licenses/upstream"
    if not source.exists():
        raise RuntimeError("Required upstream attribution directory is unavailable")
    with tempfile.TemporaryDirectory(prefix="markitai-upstream-notices-") as temporary:
        destination = Path(temporary) / "package/licenses/upstream"
        result = stage_overlay(source, destination, [])
        if result["record"]["status"] != "verified":
            raise RuntimeError("Upstream attribution did not validate")
        return {"licenses/upstream/" + path.relative_to(destination).as_posix(): path.read_bytes()
                for path in sorted(destination.rglob("*")) if path.is_file()}
