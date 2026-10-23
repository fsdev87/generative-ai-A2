#!/usr/bin/env sh
# One-click start of the screening application (macOS/Linux). Requires Docker.
set -e
cd "$(dirname "$0")"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker and run this again." >&2
  exit 1
fi

docker compose up -d --build

printf "Waiting for the server on http://localhost:8000 "
until curl -s -o /dev/null http://localhost:8000/api/health; do
  printf "."
  sleep 2
done
echo

URL=http://localhost:8000
(command -v xdg-open >/dev/null && xdg-open "$URL") || (command -v open >/dev/null && open "$URL") || true
echo "The app is running at $URL"
echo "On the first start the model is downloaded and loaded in the background (status shown top-right)."
echo "Stop it with: docker compose down"
