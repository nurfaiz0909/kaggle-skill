#!/usr/bin/env python3
"""Test suite for the Kaggle MCP Server.

Tests every MCP endpoint with both KGAT and legacy API tokens, verifying
which endpoints work with which auth type and comparing results against
documented behavior.

Usage:
    python tests/test_mcp.py                   # run all tests
    python tests/test_mcp.py --kgat-only       # only test KGAT token
    python tests/test_mcp.py --legacy-only     # only test legacy key
    python tests/test_mcp.py --quick            # test a subset (fast)
    python tests/test_mcp.py --destructive     # include write operations

Environment:
    KAGGLE_API_TOKEN   — KGAT-prefixed token (preferred for MCP)
    KAGGLE_KEY         — Legacy 32-char hex key
    KAGGLE_USERNAME    — Kaggle handle (needed for destructive tests)
    KAGGLE_MCP_TOKEN   — Override token for MCP calls (if set, used as KGAT)

    Credentials are loaded from:
    1. Environment variables
    2. .env file in repo root
    3. ~/.kaggle/kaggle.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Setup paths ───────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_ENDPOINT = "https://www.kaggle.com/mcp"

# ── Load credentials ─────────────────────────────────────────────────────────

def _load_dotenv():
    """Load .env file if present. Checks repo root, then home directory."""
    for env_path in [REPO_ROOT / ".env", Path.home() / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()


def get_kgat_token() -> str:
    """Get KGAT token from env."""
    token = os.getenv("KAGGLE_MCP_TOKEN") or os.getenv("KAGGLE_API_TOKEN", "")
    if token.startswith("KGAT_"):
        return token
    return ""


def get_legacy_key() -> str:
    """Get legacy API key from env or kaggle.json."""
    key = os.getenv("KAGGLE_KEY", "")
    if key and not key.startswith("KGAT_"):
        return key
    # Try kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text())
            k = data.get("key", "")
            if k and not k.startswith("KGAT_"):
                return k
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


def get_username() -> str:
    """Get Kaggle username."""
    u = os.getenv("KAGGLE_USERNAME", "")
    if u:
        return u
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            return json.loads(kaggle_json.read_text()).get("username", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


# ── Test infrastructure ──────────────────────────────────────────────────────

TOTAL = 0
PASSED = 0
FAILED = 0
SKIPPED = 0
KNOWN_FAILED = 0
RESULTS: list[dict] = []


def record(group: str, name: str, status: str, details: str = "") -> None:
    global TOTAL, PASSED, FAILED, SKIPPED, KNOWN_FAILED
    TOTAL += 1
    if status == "PASS":
        PASSED += 1
        icon = "\033[32m[PASS]\033[0m"
    elif status == "FAIL":
        FAILED += 1
        icon = "\033[31m[FAIL]\033[0m"
    elif status == "SKIP":
        SKIPPED += 1
        icon = "\033[33m[SKIP]\033[0m"
    elif status == "KNOWN_FAIL":
        KNOWN_FAILED += 1
        icon = "\033[33m[KNOWN_FAIL]\033[0m"
    else:
        icon = f"[{status}]"
    print(f"  {icon} {group}: {name} — {details}")
    RESULTS.append({"group": group, "name": name, "status": status, "details": details})


# ── MCP helpers ──────────────────────────────────────────────────────────────

def mcp_call(tool: str, arguments: dict, token: str, timeout: int = 30) -> dict:
    """Call an MCP tool and return parsed JSON response."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
        "id": 1,
    })
    try:
        result = subprocess.run(
            [
                "curl", "-s", "-m", str(timeout),
                "-X", "POST", MCP_ENDPOINT,
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {token}",
                "-d", payload,
            ],
            capture_output=True, text=True, timeout=timeout + 5,
        )
    except subprocess.TimeoutExpired:
        return {"error": {"message": "timeout"}}

    raw = result.stdout
    # Parse SSE format (data: ...)
    for line in raw.split("\n"):
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except json.JSONDecodeError:
                pass
    # Try raw JSON
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"raw": raw[:300]}


def classify_result(resp: dict) -> str:
    """Classify MCP response: ok, empty, unauthenticated, error:<msg>, parse_fail."""
    if "raw" in resp:
        return "parse_fail"
    error = resp.get("error", {})
    if error:
        return f"error: {error.get('message', 'unknown')[:80]}"
    result = resp.get("result", {})
    content = result.get("content", [])
    # authorize returns content as a string
    if isinstance(content, str):
        if "unauthenticated" in content.lower():
            return "unauthenticated"
        return "ok"
    if isinstance(content, list):
        for c in content:
            if not isinstance(c, dict):
                continue
            text = c.get("text", "")
            tl = text.lower()
            if "unauthenticated" in tl:
                return "unauthenticated"
            if tl.startswith("error") or '"error"' in tl or "server error" in tl:
                return f"error: {text[:100]}"
    if result and result != {}:
        return "ok"
    return "empty"


# ── Endpoint Definitions ─────────────────────────────────────────────────────

# Endpoints documented as requiring KGAT token (legacy key returns Unauthenticated)
KGAT_ONLY_ENDPOINTS = {
    "search_competitions",
    "get_competition_leaderboard",
    "list_competition_data_files",
    "download_competition_data_files",
    "download_competition_data_file",
    "download_competition_leaderboard",
    "search_competition_submissions",
    "get_competition_submission",
    "start_competition_submission_upload",
    "get_dataset_status",
    "search_notebooks",
    "list_notebook_files",
    "get_notebook_session_status",
}

# Read-only test cases: (tool_name, arguments)
READ_ONLY_TESTS = [
    # ── Auth ──
    ("authorize", {}),

    # ── Competition (read-only) ──
    ("search_competitions", {
        "request": {"search": "titanic", "hasSearch": True, "pageSize": 2, "hasPageSize": True},
    }),
    ("get_competition", {
        "request": {"competitionName": "titanic"},
    }),
    ("get_competition_leaderboard", {
        "request": {"competitionName": "titanic", "pageSize": 2, "hasPageSize": True},
    }),
    ("get_competition_data_files_summary", {
        "request": {"competitionName": "titanic"},
    }),
    ("list_competition_data_files", {
        "request": {"competitionName": "titanic", "pageSize": 3, "hasPageSize": True},
    }),
    ("list_competition_data_tree_files", {
        "request": {"competitionName": "titanic", "pageSize": 3, "hasPageSize": True},
    }),
    ("download_competition_data_file", {
        "request": {"competitionName": "titanic", "fileName": "train.csv"},
    }),
    ("download_competition_data_files", {
        "request": {"competitionName": "titanic"},
    }),
    ("download_competition_leaderboard", {
        "request": {"competitionName": "titanic"},
    }),
    ("search_competition_submissions", {
        "request": {"competitionName": "titanic", "sortBy": "Date", "group": "All",
                     "pageSize": 2, "hasPageSize": True},
    }),
    ("get_competition_submission", {
        "request": {"ref": 0},
    }),
    ("start_competition_submission_upload", {
        "request": {"competitionName": "titanic", "hasCompetitionName": True,
                     "contentLength": 100, "lastModifiedEpochSeconds": 1700000000,
                     "fileName": "test.csv"},
    }),

    # ── Dataset (read-only) ──
    ("search_datasets", {
        "request": {"search": "titanic", "hasSearch": True, "pageSize": 2,
                     "hasPageSize": True, "sortBy": "Hottest"},
    }),
    ("get_dataset_info", {
        "request": {"ownerSlug": "heptapod", "datasetSlug": "titanic"},
    }),
    ("get_dataset_metadata", {
        "request": {"ownerSlug": "heptapod", "datasetSlug": "titanic"},
    }),
    ("get_dataset_files_summary", {
        "request": {"ownerSlug": "heptapod", "datasetSlug": "titanic"},
    }),
    ("get_dataset_status", {
        "request": {"ownerSlug": "heptapod", "datasetSlug": "titanic"},
    }),
    ("list_dataset_files", {
        "request": {"ownerSlug": "heptapod", "datasetSlug": "titanic",
                     "pageSize": 3, "hasPageSize": True},
    }),
    ("list_dataset_tree_files", {
        "request": {"ownerSlug": "heptapod", "datasetSlug": "titanic",
                     "pageSize": 3, "hasPageSize": True},
    }),
    ("download_dataset", {
        "request": {"ownerSlug": "heptapod", "datasetSlug": "titanic"},
    }),

    # ── Notebook (read-only) ──
    ("search_notebooks", {
        "request": {"search": "titanic", "hasSearch": True, "pageSize": 2,
                     "hasPageSize": True, "sortBy": "Hotness", "group": "Everyone"},
    }),
    ("get_notebook_info", {
        "request": {"userName": "alexisbcook", "kernelSlug": "titanic-tutorial"},
    }),
    ("list_notebook_files", {
        "request": {"userName": "alexisbcook", "kernelSlug": "titanic-tutorial",
                     "pageSize": 3, "hasPageSize": True},
    }),
    ("download_notebook_output", {
        "request": {"ownerSlug": "alexisbcook", "kernelSlug": "titanic-tutorial"},
    }),
    ("download_notebook_output_zip", {
        "request": {"kernelSessionId": 0},
    }),
    ("get_notebook_session_status", {
        "request": {"userName": "alexisbcook", "kernelSlug": "titanic-tutorial"},
    }),
    ("list_notebook_session_output", {
        "request": {"userName": "alexisbcook", "kernelSlug": "titanic-tutorial",
                     "pageSize": 3, "hasPageSize": True},
    }),

    # ── Model (read-only) ──
    ("list_models", {
        "request": {"search": "gemma", "hasSearch": True, "pageSize": 2,
                     "hasPageSize": True, "sortBy": "Hotness", "hasSortBy": True},
    }),
    ("get_model", {
        "request": {"ownerSlug": "google", "modelSlug": "gemma"},
    }),
    ("list_model_variations", {
        "request": {"ownerSlug": "google", "modelSlug": "gemma",
                     "pageSize": 2, "hasPageSize": True},
    }),
    ("get_model_variation", {
        "request": {"ownerSlug": "google", "modelSlug": "gemma",
                     "framework": "Transformers", "instanceSlug": "2b"},
    }),
    ("list_model_variation_versions", {
        "request": {"ownerSlug": "google", "modelSlug": "gemma",
                     "framework": "Transformers", "instanceSlug": "2b",
                     "pageSize": 2, "hasPageSize": True},
    }),
    ("list_model_variation_version_files", {
        "request": {"ownerSlug": "google", "modelSlug": "gemma",
                     "framework": "Transformers", "instanceSlug": "2b",
                     "versionNumber": 2, "hasVersionNumber": True,
                     "pageSize": 3, "hasPageSize": True},
    }),
    ("download_model_variation_version", {
        "request": {"ownerSlug": "google", "modelSlug": "gemma",
                     "framework": "Transformers", "instanceSlug": "2b",
                     "versionNumber": 2, "path": "config.json", "hasPath": True},
    }),

    # ── Benchmark ──
    ("get_benchmark_leaderboard", {
        "request": {"ownerSlug": "kaggle", "benchmarkSlug": "test"},
    }),
]

# Quick subset for --quick mode
QUICK_TESTS = [
    "authorize", "search_competitions", "search_datasets", "get_dataset_info",
    "search_notebooks", "get_notebook_info", "list_models", "get_model",
    "get_competition_leaderboard", "get_dataset_status", "list_notebook_files",
]


# ── Test Groups ──────────────────────────────────────────────────────────────

def test_network_reachability():
    """Test basic connectivity to MCP endpoint."""
    group = "Network"
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "10",
             "-X", "POST", MCP_ENDPOINT,
             "-H", "Content-Type: application/json",
             "-d", '{"jsonrpc":"2.0","method":"initialize","id":0}'],
            capture_output=True, text=True, timeout=15,
        )
        code = result.stdout.strip()
        if code != "000":
            record(group, "MCP endpoint reachable", "PASS", f"HTTP {code}")
            return True
        else:
            record(group, "MCP endpoint reachable", "FAIL", "Connection failed (HTTP 000)")
            return False
    except Exception as e:
        record(group, "MCP endpoint reachable", "FAIL", str(e))
        return False


def test_tools_list(token: str, token_type: str):
    """Test tools/list and verify 47 tools are returned."""
    group = f"ToolsList ({token_type})"
    try:
        payload = json.dumps({
            "jsonrpc": "2.0", "method": "tools/list", "id": 1,
        })
        result = subprocess.run(
            ["curl", "-s", "-m", "30", "-X", "POST", MCP_ENDPOINT,
             "-H", "Content-Type: application/json",
             "-H", f"Authorization: Bearer {token}",
             "-d", payload],
            capture_output=True, text=True, timeout=35,
        )
        raw = result.stdout
        tools = []
        for line in raw.split("\n"):
            if line.startswith("data:"):
                d = json.loads(line[5:].strip())
                tools = d.get("result", {}).get("tools", [])
                break

        if len(tools) >= 47:
            record(group, "tools/list count", "PASS", f"{len(tools)} tools")
        elif len(tools) > 0:
            record(group, "tools/list count", "FAIL", f"Only {len(tools)} tools (expected 47)")
        else:
            record(group, "tools/list count", "FAIL", "No tools returned")

        # Verify expected tools exist
        tool_names = {t["name"] for t in tools}
        expected = {"authorize", "search_competitions", "search_datasets",
                    "search_notebooks", "list_models", "get_model",
                    "save_notebook", "create_model", "download_dataset"}
        missing = expected - tool_names
        if not missing:
            record(group, "expected tools present", "PASS", f"All {len(expected)} found")
        else:
            record(group, "expected tools present", "FAIL", f"Missing: {missing}")

        return tool_names
    except Exception as e:
        record(group, "tools/list", "FAIL", str(e))
        return set()


def test_endpoint(tool: str, args: dict, token: str, token_type: str, group: str):
    """Test a single MCP endpoint."""
    try:
        resp = mcp_call(tool, args, token)
        status = classify_result(resp)
    except Exception as e:
        status = f"exception: {e}"

    label = f"{tool} ({token_type})"
    is_kgat_only = tool in KGAT_ONLY_ENDPOINTS

    if status == "ok" or status == "empty":
        record(group, label, "PASS", status)
    elif status == "unauthenticated":
        if token_type == "legacy" and is_kgat_only:
            record(group, label, "KNOWN_FAIL", "Unauthenticated (expected — needs KGAT)")
        elif token_type == "KGAT":
            record(group, label, "FAIL", "Unauthenticated with KGAT token")
        else:
            record(group, label, "FAIL", "Unauthenticated")
    elif status.startswith("error:"):
        record(group, label, "FAIL", status)
    else:
        record(group, label, "FAIL", f"Unexpected: {status}")


def test_all_endpoints(token: str, token_type: str, test_list: list[tuple]):
    """Test all endpoints with a given token."""
    group = f"MCP-{token_type}"
    for tool, args in test_list:
        test_endpoint(tool, args, token, token_type, group)
        time.sleep(0.3)  # rate limiting


def test_auth_comparison(kgat_token: str, legacy_key: str, test_list: list[tuple]):
    """Compare KGAT vs legacy behavior and verify docs are accurate."""
    group = "Auth-Compare"
    documented_kgat_only = KGAT_ONLY_ENDPOINTS.copy()
    actual_kgat_only = set()
    actual_both_work = set()

    for tool, args in test_list:
        try:
            resp_kgat = mcp_call(tool, args, kgat_token)
            s_kgat = classify_result(resp_kgat)
        except Exception:
            s_kgat = "exception"
        time.sleep(0.2)
        try:
            resp_legacy = mcp_call(tool, args, legacy_key)
            s_legacy = classify_result(resp_legacy)
        except Exception:
            s_legacy = "exception"
        time.sleep(0.2)

        if s_kgat in ("ok", "empty") and s_legacy == "unauthenticated":
            actual_kgat_only.add(tool)
        elif s_kgat in ("ok", "empty") and s_legacy in ("ok", "empty"):
            actual_both_work.add(tool)

    # Verify documented KGAT-only matches actual
    tested_kgat_only_tools = {t for t, _ in test_list} & documented_kgat_only
    correct_kgat_only = tested_kgat_only_tools & actual_kgat_only
    false_negatives = tested_kgat_only_tools - actual_kgat_only  # doc says KGAT-only but works with both
    undocumented = actual_kgat_only - documented_kgat_only  # KGAT-only but not documented

    if correct_kgat_only == tested_kgat_only_tools and not undocumented:
        record(group, "KGAT-only docs accurate", "PASS",
               f"{len(correct_kgat_only)} of {len(tested_kgat_only_tools)} documented endpoints confirmed")
    else:
        details = []
        if false_negatives:
            details.append(f"docs say KGAT-only but both work: {false_negatives}")
        if undocumented:
            details.append(f"KGAT-only but not documented: {undocumented}")
        record(group, "KGAT-only docs accurate", "FAIL", "; ".join(details))

    record(group, "endpoints tested", "PASS",
           f"KGAT-only={len(actual_kgat_only)}, both={len(actual_both_work)}, "
           f"total={len(test_list)}")


def test_tools_list_count():
    """Verify that the documented tool count (47) is still accurate."""
    group = "ToolCount"
    kgat = get_kgat_token()
    legacy = get_legacy_key()
    token = kgat or legacy
    if not token:
        record(group, "tool count", "SKIP", "No credentials")
        return

    try:
        payload = json.dumps({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        result = subprocess.run(
            ["curl", "-s", "-m", "30", "-X", "POST", MCP_ENDPOINT,
             "-H", "Content-Type: application/json",
             "-H", f"Authorization: Bearer {token}",
             "-d", payload],
            capture_output=True, text=True, timeout=35,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("data:"):
                d = json.loads(line[5:].strip())
                count = len(d.get("result", {}).get("tools", []))
                if count == 47:
                    record(group, "documented count matches", "PASS", f"{count} tools")
                else:
                    record(group, "documented count matches", "FAIL",
                           f"Got {count}, documented 47 — update mcp-reference.md")
                return
        record(group, "documented count matches", "FAIL", "Could not parse response")
    except Exception as e:
        record(group, "documented count matches", "FAIL", str(e))


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report() -> Path:
    """Generate markdown test report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = REPO_ROOT / "tests" / f"test-mcp-report-{date_str}.md"

    lines = [
        "# Kaggle MCP Server Test Report",
        "",
        f"**Date:** {timestamp}",
        f"**User:** {get_username() or 'N/A'}",
        f"**KGAT Token:** {'Available' if get_kgat_token() else 'Not available'}",
        f"**Legacy Key:** {'Available' if get_legacy_key() else 'Not available'}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total | {TOTAL} |",
        f"| Passed | {PASSED} |",
        f"| Failed | {FAILED} |",
        f"| Known Fail | {KNOWN_FAILED} |",
        f"| Skipped | {SKIPPED} |",
    ]

    run = TOTAL - SKIPPED - KNOWN_FAILED
    pct_run = (PASSED * 100 // run) if run > 0 else 0
    lines.extend([
        f"| Pass Rate (excl. skip/known) | {pct_run}% |",
        "",
        "## Results",
        "",
        "| Group | Test | Status | Details |",
        "|-------|------|--------|---------|",
    ])
    for r in RESULTS:
        lines.append(f"| {r['group']} | {r['name']} | {r['status']} | {r['details']} |")

    lines.extend([
        "",
        "## KGAT-Only Endpoints (Legacy Key Returns Unauthenticated)",
        "",
        "These 13 endpoints require a KGAT token:",
        "",
    ])
    for ep in sorted(KGAT_ONLY_ENDPOINTS):
        lines.append(f"- `{ep}`")

    lines.extend([
        "",
        "---",
        f"*Generated by test_mcp.py on {timestamp}*",
        "",
    ])

    report_path.write_text("\n".join(lines))
    print(f"\n  Report saved to: {report_path}")
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kaggle MCP Server Test Suite")
    parser.add_argument("--kgat-only", action="store_true", help="Only test KGAT token")
    parser.add_argument("--legacy-only", action="store_true", help="Only test legacy key")
    parser.add_argument("--quick", action="store_true", help="Test a subset of endpoints")
    parser.add_argument("--compare", action="store_true",
                        help="Compare both tokens and verify docs (default if both available)")
    parser.add_argument("--destructive", action="store_true", help="Include write operations")
    args = parser.parse_args()

    kgat = get_kgat_token()
    legacy = get_legacy_key()

    print(f"\n{'=' * 60}")
    print(f"  Kaggle MCP Server Test Suite")
    print(f"{'=' * 60}")
    print(f"  KGAT token:  {'available' if kgat else 'NOT FOUND'}")
    print(f"  Legacy key:  {'available' if legacy else 'NOT FOUND'}")
    print(f"  Username:    {get_username() or 'N/A'}")
    print()

    if not kgat and not legacy:
        print("  ERROR: No credentials found. Set KAGGLE_API_TOKEN or KAGGLE_KEY.")
        sys.exit(1)

    # Filter test list
    test_list = READ_ONLY_TESTS
    if args.quick:
        test_list = [(t, a) for t, a in READ_ONLY_TESTS if t in QUICK_TESTS]

    # ── Network ──
    print("  --- Network ---\n")
    if not test_network_reachability():
        print("\n  FATAL: Cannot reach MCP endpoint. Aborting.")
        sys.exit(1)

    # ── Tool count verification ──
    print("\n  --- Tool Count ---\n")
    test_tools_list_count()

    # ── KGAT tests ──
    if kgat and not args.legacy_only:
        print("\n  --- KGAT Token Tests ---\n")
        test_tools_list(kgat, "KGAT")
        test_all_endpoints(kgat, "KGAT", test_list)

    # ── Legacy tests ──
    if legacy and not args.kgat_only:
        print("\n  --- Legacy Key Tests ---\n")
        test_tools_list(legacy, "legacy")
        test_all_endpoints(legacy, "legacy", test_list)

    # ── Auth comparison ──
    if kgat and legacy and not args.kgat_only and not args.legacy_only:
        print("\n  --- Auth Comparison ---\n")
        test_auth_comparison(kgat, legacy, test_list)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed, "
          f"{KNOWN_FAILED} known_fail, {SKIPPED} skipped (total: {TOTAL})")
    print(f"{'=' * 60}\n")

    generate_report()

    sys.exit(1 if FAILED > 0 else 0)


if __name__ == "__main__":
    main()
