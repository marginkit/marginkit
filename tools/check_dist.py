"""Check the built sdist and wheel in ``dist/`` before they are published.

Run after ``python -m build``: ``python tools/check_dist.py [--expect-version X.Y.Z]``. Both the
``packaging shape`` CI job and the release workflow call this file, so there is one copy of the
check (CLAUDE.md, on second copies of a rule).

A missing ``py.typed`` fails silently and disables type checking for every consumer, and
``load_schema(version)`` reads the packaged schemas at runtime, so both are asserted inside the
built wheel rather than merely on disk.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import tarfile
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-version",
        help="fail unless both artifacts carry exactly this version (the release tag, minus 'v')",
    )
    args = parser.parse_args()

    wheels = glob.glob("dist/*.whl")
    sdists = glob.glob("dist/*.tar.gz")
    assert len(wheels) == 1, f"expected exactly one wheel in dist/, found {wheels}"
    assert len(sdists) == 1, f"expected exactly one sdist in dist/, found {sdists}"
    wheel, sdist = wheels[0], sdists[0]

    names = zipfile.ZipFile(wheel).namelist()
    assert "robustmargin/py.typed" in names, f"py.typed missing from {wheel}"
    # Derived from the source tree rather than hardcoded, so a schema_version bump cannot leave
    # this check silently behind.
    expected_schemas = sorted(
        f"robustmargin/schema/{p.name}"
        for p in pathlib.Path("src/robustmargin/schema").glob("scorecard-v*.json")
    )
    assert expected_schemas, "no scorecard-v*.json found in src/robustmargin/schema"
    for schema in expected_schemas:
        assert schema in names, f"{schema} missing from {wheel}"

    members = tarfile.open(sdist).getnames()
    assert any(n.endswith("/LICENSE") for n in members), f"LICENSE missing from {sdist}"
    # The planning documents are deliberately private (.gitignore). A build from a clean
    # checkout never contains them; this catches a build from a working tree that does.
    private = [
        n
        for n in members
        if "/.claude/" in n or n.endswith("/CLAUDE.md") or ("/docs/" in n and "PROVENANCE" not in n)
    ]
    assert not private, f"private files in {sdist}: {private}"

    if args.expect_version is not None:
        for artifact in (wheel, sdist):
            name = pathlib.Path(artifact).name
            assert f"robustmargin-{args.expect_version}" in name, (
                f"{name} does not carry version {args.expect_version}"
            )

    print(
        f"py.typed and {len(expected_schemas)} schema file(s) present in wheel "
        f"({', '.join(expected_schemas)}); LICENSE present and no private files in sdist"
        + (f"; version {args.expect_version} confirmed" if args.expect_version else "")
    )


if __name__ == "__main__":
    main()
