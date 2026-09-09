"""Proxy resolution: NO_PROXY bypass and trustworthy proxy detection.

Covers two regressions:

- ``get_proxy_for_url()`` was dead code, so NO_PROXY never reached any
  connection: every backend called ``_detect_proxy()`` directly.
- ``detect_proxy()`` probed common localhost proxy ports with a bare TCP
  ``connect_ex()``. Under a TUN-mode proxy every port answers, so detection
  reported a proxy that does not exist.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from markitai.fetch_session import FetchSession, get_default_session


class _StubResponse:
    """Minimal response object shaped like httpx/curl-cffi responses."""

    def __init__(self, url: str) -> None:
        self.content = b"<html>ok</html>"
        self.status_code = 200
        self.headers = {"Content-Type": "text/html"}
        self.url = url


class _StubClient:
    """Stand-in for the pooled httpx client / curl-cffi session."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    async def get(self, url: str, **kwargs: Any) -> _StubResponse:
        self.requested.append(url)
        return _StubResponse(url)


@pytest.fixture
def clean_session() -> Iterator[None]:
    """Reset the process-wide session's cached proxy state around a test."""
    session = get_default_session()
    saved = (session.detected_proxy, session.detected_proxy_bypass)
    session.detected_proxy = None
    session.detected_proxy_bypass = None
    try:
        yield
    finally:
        session.detected_proxy, session.detected_proxy_bypass = saved


class TestDetectProxyDoesNotProbePorts:
    """Auto-detection must not rely on raw TCP reachability."""

    def test_detect_proxy_does_not_open_sockets(self) -> None:
        """No proxy configured means no proxy — not a port scan.

        A TUN-mode proxy accepts a TCP connection on every local port, so a
        ``connect_ex()`` probe reports a proxy that is not there.
        """
        session = FetchSession()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("markitai.fetch_session._get_system_proxy", return_value=("", "")),
            patch("socket.socket") as mock_socket,
        ):
            result = session.detect_proxy(force_recheck=True)

        assert result == ""
        mock_socket.assert_not_called()

    def test_env_proxy_still_wins(self) -> None:
        """Explicit environment configuration remains the primary source."""
        session = FetchSession()
        with (
            patch.dict(
                os.environ,
                {"HTTPS_PROXY": "http://127.0.0.1:7890", "NO_PROXY": "internal.corp"},
                clear=True,
            ),
            patch("markitai.fetch_session._get_system_proxy", return_value=("", "")),
        ):
            assert session.detect_proxy(force_recheck=True) == "http://127.0.0.1:7890"
        assert session.detected_proxy_bypass == "internal.corp"

    def test_system_proxy_still_used(self) -> None:
        """OS-level proxy settings remain the secondary source."""
        session = FetchSession()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "markitai.fetch_session._get_system_proxy",
                return_value=("http://10.0.0.1:8080", "*.local"),
            ),
        ):
            assert session.detect_proxy(force_recheck=True) == "http://10.0.0.1:8080"
        assert session.detected_proxy_bypass == "*.local"


@pytest.fixture
def linux_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep desktop tests independent of the host OS and installed tools."""
    for key in list(os.environ):
        if key.lower().endswith("_proxy") or key == "XDG_CURRENT_DESKTOP":
            monkeypatch.delenv(key)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", lambda tool: f"/usr/bin/{tool}")


def _gnome_output(**overrides: str) -> str:
    values = {
        "mode": "'manual'",
        "https.host": "'secure.proxy'",
        "https.port": "8443",
        "http.host": "'web.proxy'",
        "http.port": "8080",
        "http.use-authentication": "false",
        "use-same-proxy": "false",
        "ignore-hosts": "['localhost', '*.corp', '10.0.0.0/8']",
    }
    values.update(overrides)
    return "\n".join(
        f"org.gnome.system.proxy{'.' + key.rsplit('.', 1)[0] if '.' in key else ''} "
        f"{key.rsplit('.', 1)[-1]} {value}"
        for key, value in values.items()
    )


class TestLinuxDesktopProxy:
    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, "http://secure.proxy:8443"),
            ({"https.host": "''"}, "http://web.proxy:8080"),
            ({"https.port": "0"}, "http://web.proxy:8080"),
            ({"https.port": "70000"}, "http://web.proxy:8080"),
            ({"https.host": "'::1'"}, "http://[::1]:8443"),
            ({"use-same-proxy": "true"}, "http://web.proxy:8080"),
            ({"ignore-hosts": "@as []"}, "http://secure.proxy:8443"),
            ({"mode": "'none'"}, ""),
            ({"mode": "'auto'"}, ""),
            ({"http.use-authentication": "true"}, ""),
            ({"https.host": "''", "http.host": "''"}, ""),
            ({"ignore-hosts": "'not a list'"}, ""),
            ({"mode": "not valid"}, ""),
            ({"https.host": "'user@proxy'", "http.host": "''"}, ""),
        ],
    )
    def test_gnome(
        self,
        linux_desktop: None,
        monkeypatch: pytest.MonkeyPatch,
        overrides: dict[str, str],
        expected: str,
    ) -> None:
        import subprocess

        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
        before = dict(os.environ)
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, _gnome_output(**overrides)),
        ) as run:
            session = FetchSession()
            assert session.detect_proxy() == expected
            assert session.detect_proxy() == expected
            run.assert_called_once()
            assert run.call_args.args[0] == [
                "/usr/bin/gsettings",
                "list-recursively",
                "org.gnome.system.proxy",
            ]
            assert 0 < run.call_args.kwargs["timeout"] <= 1
            assert run.call_args.kwargs["stdin"] == subprocess.DEVNULL
            if expected and "ignore-hosts" not in overrides:
                assert session.is_proxy_bypassed("http://service.corp")
                assert session.is_proxy_bypassed("http://10.1.2.3")
            session.detect_proxy(force_recheck=True)
            assert run.call_count == 2
        assert dict(os.environ) == before

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, "http://secure.proxy:8443"),
            (
                {"httpsProxy": "", "httpProxy": "web.proxy:8080"},
                "http://web.proxy:8080",
            ),
            ({"httpsProxy": "http://[::1] 8080"}, "http://[::1]:8080"),
            ({"ProxyType": "0"}, ""),
            ({"ProxyType": "2"}, ""),
            ({"ProxyType": "3"}, ""),
            ({"ProxyType": "4"}, ""),
            ({"ReversedException": "true"}, ""),
            ({"httpsProxy": "socks://proxy:1080", "httpProxy": ""}, ""),
            ({"httpsProxy": "http://proxy:99999", "httpProxy": ""}, ""),
            ({"httpsProxy": "http://proxy/path", "httpProxy": ""}, ""),
        ],
    )
    def test_kde(
        self,
        linux_desktop: None,
        monkeypatch: pytest.MonkeyPatch,
        overrides: dict[str, str],
        expected: str,
    ) -> None:
        import subprocess

        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE:GNOME")
        values = {
            "ProxyType": "1",
            "ReversedException": "false",
            "httpsProxy": "http://secure.proxy 8443",
            "httpProxy": "http://web.proxy 8080",
            "NoProxyFor": "localhost,.corp;10.0.0.0/8",
        }
        values.update(overrides)

        def run(args: list[str], **kwargs: Any) -> Any:
            assert args[:6] == [
                "/usr/bin/kreadconfig6",
                "--file",
                "kioslaverc",
                "--group",
                "Proxy Settings",
                "--key",
            ]
            assert 0 < kwargs["timeout"] <= 1
            return subprocess.CompletedProcess(args, 0, values[args[-1]])

        with patch("subprocess.run", side_effect=run):
            session = FetchSession()
            assert session.detect_proxy() == expected
            if expected:
                assert session.is_proxy_bypassed("https://service.corp")
                assert session.is_proxy_bypassed("http://10.2.3.4")

    @pytest.mark.parametrize("desktop", ["", "sway", "XFCE", "Cinnamon"])
    def test_unknown_desktop_does_not_probe(
        self,
        linux_desktop: None,
        monkeypatch: pytest.MonkeyPatch,
        desktop: str,
    ) -> None:
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
        with patch("subprocess.run") as run:
            assert FetchSession().detect_proxy() == ""
            run.assert_not_called()

    @pytest.mark.parametrize("desktop", ["GNOME", "KDE"])
    def test_missing_tool(
        self,
        linux_desktop: None,
        monkeypatch: pytest.MonkeyPatch,
        desktop: str,
    ) -> None:
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
        monkeypatch.setattr("shutil.which", lambda _: None)
        with patch("subprocess.run") as run:
            assert FetchSession().detect_proxy() == ""
            run.assert_not_called()

    @pytest.mark.parametrize(
        "variable",
        [
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "https_proxy",
            "http_proxy",
            "all_proxy",
        ],
    )
    def test_env_skips_desktop_tools(
        self,
        linux_desktop: None,
        monkeypatch: pytest.MonkeyPatch,
        variable: str,
    ) -> None:
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
        monkeypatch.setenv(variable, "http://env.proxy:1234")
        with patch("subprocess.run") as run:
            assert FetchSession().detect_proxy() == "http://env.proxy:1234"
            run.assert_not_called()

    @pytest.mark.parametrize("desktop", ["GNOME", "KDE"])
    def test_read_failures_are_silent(
        self,
        linux_desktop: None,
        monkeypatch: pytest.MonkeyPatch,
        desktop: str,
    ) -> None:
        import subprocess

        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
        for failure in [
            FileNotFoundError(),
            PermissionError(),
            subprocess.TimeoutExpired("tool", 1),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
        ]:
            with patch("subprocess.run", side_effect=failure):
                assert FetchSession().detect_proxy() == ""
        with patch(
            "subprocess.run", return_value=subprocess.CompletedProcess([], 1, "")
        ):
            assert FetchSession().detect_proxy() == ""

    def test_kde5_fallback_and_total_budget(
        self,
        linux_desktop: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess

        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
        monkeypatch.setattr(
            "shutil.which",
            lambda tool: "/usr/bin/kreadconfig5" if tool == "kreadconfig5" else None,
        )
        with (
            patch("markitai.fetch_session.time.monotonic", side_effect=[0, 0.2, 1.1]),
            patch(
                "subprocess.run", return_value=subprocess.CompletedProcess([], 0, "1")
            ) as run,
        ):
            assert FetchSession().detect_proxy() == ""
            run.assert_called_once()
            assert run.call_args.args[0][0] == "/usr/bin/kreadconfig5"
            assert run.call_args.kwargs["timeout"] == pytest.approx(0.8)


class TestSessionProxyBypass:
    """FetchSession owns the single "is this host exempt?" predicate."""

    def test_bypasses_host_listed_in_no_proxy_env(self) -> None:
        session = FetchSession()
        with patch.dict(os.environ, {"NO_PROXY": "internal.corp"}, clear=True):
            assert session.is_proxy_bypassed("https://internal.corp/page") is True
            assert session.is_proxy_bypassed("https://example.com/page") is False

    def test_bypasses_suffix_pattern(self) -> None:
        session = FetchSession()
        with patch.dict(os.environ, {"no_proxy": ".internal.corp"}, clear=True):
            assert session.is_proxy_bypassed("https://api.internal.corp/x") is True
            # NO_PROXY suffix syntax matches subdomains only
            assert session.is_proxy_bypassed("https://internal.corp/x") is False

    def test_bypasses_system_exception_list(self) -> None:
        """The OS proxy exception list recorded by detect_proxy also applies."""
        session = FetchSession()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "markitai.fetch_session._get_system_proxy",
                return_value=("http://10.0.0.1:8080", "localhost,*.corp"),
            ),
        ):
            session.detect_proxy(force_recheck=True)
            assert session.is_proxy_bypassed("http://app.corp/health") is True
            assert session.is_proxy_bypassed("https://example.com") is False

    def test_no_patterns_means_no_bypass(self) -> None:
        session = FetchSession()
        with patch.dict(os.environ, {}, clear=True):
            assert session.is_proxy_bypassed("https://example.com") is False


class TestGetProxyForUrlHonoursEnvNoProxy:
    """The public entry point picks up NO_PROXY even without a cached bypass."""

    def test_no_proxy_env_alone_bypasses(self, clean_session: None) -> None:
        from markitai.fetch import get_proxy_for_url

        session = get_default_session()
        session.detected_proxy = "http://127.0.0.1:7890"
        session.detected_proxy_bypass = None  # never populated by a system probe

        with patch.dict(os.environ, {"NO_PROXY": "example.com"}, clear=True):
            assert get_proxy_for_url("https://example.com/page") == ""
            assert (
                get_proxy_for_url("https://other.com/page") == "http://127.0.0.1:7890"
            )


class TestResolveProxyForUrlFallback:
    """The bypass still works when fetch_session never registered a provider."""

    def test_env_no_proxy_without_registered_provider(self) -> None:
        from markitai import fetch_http

        saved = fetch_http._proxy_bypass_provider
        fetch_http._proxy_bypass_provider = None
        try:
            with patch.dict(os.environ, {"NO_PROXY": "internal.corp"}, clear=True):
                assert (
                    fetch_http.resolve_proxy_for_url(
                        "https://internal.corp/x", "http://127.0.0.1:7890"
                    )
                    is None
                )
                assert (
                    fetch_http.resolve_proxy_for_url(
                        "https://example.com/x", "http://127.0.0.1:7890"
                    )
                    == "http://127.0.0.1:7890"
                )
        finally:
            fetch_http._proxy_bypass_provider = saved

    def test_no_candidate_proxy_stays_none(self) -> None:
        from markitai.fetch_http import resolve_proxy_for_url

        assert resolve_proxy_for_url("https://example.com", None) is None
        assert resolve_proxy_for_url("https://example.com", "") is None


class TestStaticHttpClientsHonourNoProxy:
    """The HTTP backends must not tunnel bypassed hosts through the proxy."""

    @pytest.mark.asyncio
    async def test_httpx_client_drops_proxy_for_bypassed_host(
        self, clean_session: None
    ) -> None:
        from markitai.fetch_http import HttpxClient

        session = get_default_session()
        session.detected_proxy = "http://127.0.0.1:7890"
        session.detected_proxy_bypass = "internal.corp"

        client = HttpxClient()
        seen: list[str | None] = []
        stub = _StubClient()

        def _spy(timeout_s: float, proxy: str | None) -> Any:
            seen.append(proxy)
            return stub

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(client, "_get_or_create_client", _spy),
        ):
            await client.get(
                "https://internal.corp/page",
                headers={},
                timeout_s=5.0,
                proxy="http://127.0.0.1:7890",
            )

        assert seen == [None]

    @pytest.mark.asyncio
    async def test_httpx_client_keeps_proxy_for_other_hosts(
        self, clean_session: None
    ) -> None:
        from markitai.fetch_http import HttpxClient

        session = get_default_session()
        session.detected_proxy = "http://127.0.0.1:7890"
        session.detected_proxy_bypass = "internal.corp"

        client = HttpxClient()
        seen: list[str | None] = []
        stub = _StubClient()

        def _spy(timeout_s: float, proxy: str | None) -> Any:
            seen.append(proxy)
            return stub

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(client, "_get_or_create_client", _spy),
        ):
            await client.get(
                "https://example.com/page",
                headers={},
                timeout_s=5.0,
                proxy="http://127.0.0.1:7890",
            )

        assert seen == ["http://127.0.0.1:7890"]

    @pytest.mark.asyncio
    async def test_curl_cffi_client_drops_proxy_for_bypassed_host(
        self, clean_session: None
    ) -> None:
        from markitai.fetch_http import CurlCffiClient

        session = get_default_session()
        session.detected_proxy = "http://127.0.0.1:7890"
        session.detected_proxy_bypass = None

        client = CurlCffiClient()
        seen: list[str | None] = []
        stub = _StubClient()

        def _spy(proxy: str | None) -> Any:
            seen.append(proxy)
            return stub

        with (
            patch.dict(os.environ, {"NO_PROXY": "10.0.0.0/8"}, clear=True),
            patch.object(client, "_get_or_create_session", _spy),
        ):
            await client.get(
                "http://10.1.2.3:9000/page",
                headers={},
                timeout_s=5.0,
                proxy="http://127.0.0.1:7890",
            )

        assert seen == [None]


@pytest.fixture
def proxy_detected(clean_session: None) -> Iterator[None]:
    """Pin a detected proxy so only the NO_PROXY decision is under test."""
    session = get_default_session()
    session.detected_proxy = "http://127.0.0.1:7890"
    session.detected_proxy_bypass = None
    yield


class TestPlaywrightKwargsHonourNoProxy:
    """``_get_playwright_fetch_kwargs`` builds the browser's proxy option."""

    @staticmethod
    def _kwargs(url: str) -> dict[str, Any]:
        from markitai.config import FetchConfig
        from markitai.fetch_support import _get_playwright_fetch_kwargs

        return _get_playwright_fetch_kwargs(url, FetchConfig())

    def test_bypassed_url_launches_browser_without_proxy(
        self, proxy_detected: None
    ) -> None:
        with patch.dict(os.environ, {"NO_PROXY": "internal.corp"}, clear=True):
            assert self._kwargs("https://internal.corp/page")["proxy"] is None

    def test_other_url_keeps_proxy(self, proxy_detected: None) -> None:
        with patch.dict(os.environ, {"NO_PROXY": "internal.corp"}, clear=True):
            assert (
                self._kwargs("https://example.com/page")["proxy"]
                == "http://127.0.0.1:7890"
            )


class _ProxySpyClient:
    """Records the ``proxy=`` httpx.AsyncClient was constructed with."""

    def __init__(self, seen: list[str | None], response: Any) -> None:
        self._seen = seen
        self._response = response

    def __call__(self, *args: Any, **kwargs: Any) -> _ProxySpyClient:
        self._seen.append(kwargs.get("proxy"))
        return self

    async def __aenter__(self) -> _ProxySpyClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        return self._response


class _CFResponse:
    """Minimal successful CF Browser Rendering / toMarkdown response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class TestCloudflareStrategyHonoursNoProxy:
    """CF Browser Rendering builds its own httpx client: it must bypass too."""

    @staticmethod
    async def _proxy_used(no_proxy: str) -> str | None:
        import httpx

        from markitai.fetch_strategies.cloudflare import fetch_with_cloudflare

        seen: list[str | None] = []
        response = _CFResponse({"success": True, "result": "<html><p>x</p></html>"})

        with (
            patch.dict(os.environ, {"NO_PROXY": no_proxy}, clear=True),
            patch.object(httpx, "AsyncClient", _ProxySpyClient(seen, response)),
        ):
            await fetch_with_cloudflare(
                "https://example.com/page",
                api_token="tok",
                account_id="acct",
            )

        assert len(seen) == 1
        return seen[0]

    @pytest.mark.asyncio
    async def test_bypassed_api_host_drops_proxy(self, proxy_detected: None) -> None:
        assert await self._proxy_used("api.cloudflare.com") is None

    @pytest.mark.asyncio
    async def test_unrelated_bypass_keeps_proxy(self, proxy_detected: None) -> None:
        assert await self._proxy_used("internal.corp") == "http://127.0.0.1:7890"


class TestCloudflareConverterHonoursNoProxy:
    """CF toMarkdown conversion uses the same api.cloudflare.com endpoint."""

    @staticmethod
    async def _proxy_used(tmp_file: Any, no_proxy: str) -> str | None:
        import httpx

        from markitai.converter.cloudflare import CloudflareConverter

        seen: list[str | None] = []
        response = _CFResponse(
            {"success": True, "result": [{"data": "# Converted", "format": "markdown"}]}
        )

        with (
            patch.dict(os.environ, {"NO_PROXY": no_proxy}, clear=True),
            patch.object(httpx, "AsyncClient", _ProxySpyClient(seen, response)),
        ):
            converter = CloudflareConverter(api_token="tok", account_id="acct")
            await converter.convert_async(tmp_file)

        assert len(seen) == 1
        return seen[0]

    @pytest.mark.asyncio
    async def test_bypassed_api_host_drops_proxy(
        self, proxy_detected: None, tmp_path: Any
    ) -> None:
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 fake")

        assert await self._proxy_used(src, "api.cloudflare.com") is None

    @pytest.mark.asyncio
    async def test_unrelated_bypass_keeps_proxy(
        self, proxy_detected: None, tmp_path: Any
    ) -> None:
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 fake")

        assert await self._proxy_used(src, "internal.corp") == "http://127.0.0.1:7890"


class TestBatchSharedRendererProxy:
    """One browser is shared by a whole batch; its proxy is a batch decision."""

    @staticmethod
    def _resolve(urls: list[str]) -> str | None:
        from markitai.cli.processors.batch import _shared_renderer_proxy

        return _shared_renderer_proxy(urls)

    def test_all_urls_bypassed_means_no_proxy(self, proxy_detected: None) -> None:
        with patch.dict(os.environ, {"NO_PROXY": "internal.corp"}, clear=True):
            assert (
                self._resolve(
                    ["https://internal.corp/a", "https://internal.corp/b"],
                )
                is None
            )

    def test_no_url_bypassed_keeps_proxy(self, proxy_detected: None) -> None:
        with patch.dict(os.environ, {"NO_PROXY": "internal.corp"}, clear=True):
            assert (
                self._resolve(["https://example.com/a", "https://other.com/b"])
                == "http://127.0.0.1:7890"
            )

    def test_mixed_batch_keeps_proxy_for_the_shared_browser(
        self, proxy_detected: None
    ) -> None:
        """A single browser cannot be proxied per URL; the majority case wins."""
        with patch.dict(os.environ, {"NO_PROXY": "internal.corp"}, clear=True):
            assert (
                self._resolve(["https://internal.corp/a", "https://example.com/b"])
                == "http://127.0.0.1:7890"
            )

    def test_no_detected_proxy_stays_none(self, clean_session: None) -> None:
        session = get_default_session()
        session.detected_proxy = ""
        session.detected_proxy_bypass = ""

        with patch.dict(os.environ, {}, clear=True):
            assert self._resolve(["https://example.com/a"]) is None
