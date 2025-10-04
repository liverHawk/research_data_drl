#!/bin/sh
set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1
git config core.hooksPath .githooks
chmod -R +x .githooks scripts
echo "Installed hooks: core.hooksPath -> .githooks"