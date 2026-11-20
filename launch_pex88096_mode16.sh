#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PEX88096_MODE16=1
exec "$root/launch_pex88096.sh" "$@"
