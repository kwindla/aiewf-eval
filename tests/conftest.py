"""Keep maintainer evidence audits separate from tests of a fresh checkout."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--local-evidence", action="store_true",
        help="Also audit the restored campaign archive and original local runs.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local_evidence: requires archived campaign evidence and original runs"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--local-evidence"):
        return
    skip = pytest.mark.skip(reason="Local archive audit; enable with --local-evidence")
    for item in items:
        if "local_evidence" in item.keywords:
            item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (item.config.getoption("--local-evidence")
            and "local_evidence" in item.keywords and report.skipped):
        report.outcome = "failed"
        report.longrepr = "Explicit local-evidence audit skipped: restore all required evidence."
