#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import sys

def bump_version_text(text: str, part: str = "patch"):
    m = re.search(r'^(version\s*=\s*")(\d+)\.(\d+)\.(\d+)(")', text, flags=re.MULTILINE)
    if not m:
        return None
    major, minor, patch = int(m.group(2)), int(m.group(3)), int(m.group(4))
    if part == "major":
        major += 1; minor = 0; patch = 0
    elif part == "minor":
        minor += 1; patch = 0
    else:
        patch += 1
    newver = f"{major}.{minor}.{patch}"
    new_text = text[:m.start()] + m.group(1) + newver + m.group(5) + text[m.end():]
    return new_text, newver

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--part", choices=["major","minor","patch"], default="patch")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"{path} not found", file=sys.stderr); return 1
    text = path.read_text(encoding="utf-8")
    res = bump_version_text(text, args.part)
    if res is None:
        print("version field not found", file=sys.stderr); return 1
    new_text, newver = res
    path.write_text(new_text, encoding="utf-8")
    print(f"Bumped {path} -> {newver}")
    return 0

if __name__ == '__main__':
    sys.exit(main())