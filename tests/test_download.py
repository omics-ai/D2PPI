"""The full download keeps a partial file when the connection drops, and the next run resumes it."""

import http.server
import os
import threading

import pytest

from d2ppi.download import fetch

PAYLOAD = bytes(range(256)) * 40


class Handler(http.server.BaseHTTPRequestHandler):
    cut = None  # send only this many bytes of the body, then close: a dropped connection
    ranges = []

    def do_GET(self):
        rng = self.headers.get("Range")
        Handler.ranges.append(rng)
        body = PAYLOAD[int(rng[len("bytes=") :].split("-")[0]) :] if rng else PAYLOAD
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[: self.cut])

    def log_message(self, *args):
        pass


def test_fetch_resumes_a_dropped_download(tmp_path, monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(Handler, "ranges", [])
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/benchmark.zip"
    dest = str(tmp_path / "benchmark.zip")
    try:
        monkeypatch.setattr(Handler, "cut", 3000)
        with pytest.raises(SystemExit, match="run the command again to resume"):
            fetch(url, dest, len(PAYLOAD))
        assert not os.path.exists(dest) and os.path.getsize(dest + ".part") == 3000

        monkeypatch.setattr(Handler, "cut", None)
        fetch(url, dest, len(PAYLOAD))
    finally:
        server.shutdown()
        server.server_close()
    assert Handler.ranges == [None, "bytes=3000-"]
    with open(dest, "rb") as f:
        assert f.read() == PAYLOAD
    assert not os.path.exists(dest + ".part")
