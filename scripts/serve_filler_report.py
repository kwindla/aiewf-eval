#!/usr/bin/env python3
"""Serve the generated filler-token report without exposing the repository."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "filler-token-latent-scratchpad-study.html"
)


class ReportHandler(BaseHTTPRequestHandler):
    report_path: Path

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.partition("?")[0]
        if path == "/healthz":
            self._send(
                HTTPStatus.OK,
                b"ok\n",
                "text/plain; charset=utf-8",
                include_body=False,
            )
            return
        if path in {"/", "/index.html"}:
            try:
                body = self.report_path.read_bytes()
            except OSError as exc:
                self.log_error("could not read report: %s", exc)
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    b"report unavailable\n",
                    "text/plain; charset=utf-8",
                    include_body=False,
                )
                return
            self._send(
                HTTPStatus.OK,
                body,
                "text/html; charset=utf-8",
                include_body=False,
            )
            return
        self._send(
            HTTPStatus.NOT_FOUND,
            b"not found\n",
            "text/plain; charset=utf-8",
            include_body=False,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.partition("?")[0]
        if path == "/healthz":
            self._send(HTTPStatus.OK, b"ok\n", "text/plain; charset=utf-8")
            return
        if path not in {"/", "/index.html"}:
            self._send(
                HTTPStatus.NOT_FOUND,
                b"not found\n",
                "text/plain; charset=utf-8",
            )
            return

        try:
            body = self.report_path.read_bytes()
        except OSError as exc:
            self.log_error("could not read report: %s", exc)
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                b"report unavailable\n",
                "text/plain; charset=utf-8",
            )
            return

        self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        include_body: bool = True,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.end_headers()
        if include_body:
            self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report.expanduser().resolve()
    if not report_path.is_file():
        raise SystemExit(f"report does not exist: {report_path}")

    handler = type("ConfiguredReportHandler", (ReportHandler,), {})
    handler.report_path = report_path
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {report_path} at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
