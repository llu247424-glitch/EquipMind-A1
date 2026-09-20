from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = {
    "Manual files": list((ROOT / "data" / "manuals").glob("*.md")),
    "Approved case files": list((ROOT / "data" / "cases" / "approved").glob("*.md")),
    "Workflow files": list((ROOT / "data" / "workflows").glob("*.json")),
    "Image files": list((ROOT / "data" / "images").glob("*.png")),
}
for name, files in checks.items():
    print(f"{name}: {len(files)}")
    for f in files[:12]:
        print("  -", f.relative_to(ROOT))

assert len(checks["Manual files"]) >= 5
assert len(checks["Approved case files"]) >= 14
assert len(checks["Workflow files"]) >= 5
assert len(checks["Image files"]) >= 5
print("data check passed")
