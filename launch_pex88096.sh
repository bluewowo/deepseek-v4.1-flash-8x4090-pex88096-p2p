#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ENV_FILE="${ENV_FILE:-$root/.env.pex88096}"
if [[ ! -r "$ENV_FILE" ]]; then
  echo 'Copy profiles/pex88096.env.example to .env.pex88096 and configure the host first.' >&2
  exit 1
fi
exec "$root/launch.sh" "$@"
