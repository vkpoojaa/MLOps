#!/usr/bin/env sh
set -eu

# Build from a real Git checkout so the image is cryptographically traceable to source.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: build-image.sh must run inside the project's Git checkout." >&2
  echo "A source ZIP has no authoritative commit; do not invent one." >&2
  exit 2
fi

COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
VERSION="${APP_VERSION:-0.1.0}"
IMAGE="${IMAGE_NAME:-mlops-house-price-api}"
TAG="${IMAGE}:${VERSION}-${SHORT_COMMIT}"

docker build \
  --build-arg APP_VERSION="$VERSION" \
  --build-arg GIT_COMMIT="$COMMIT" \
  --build-arg IMAGE_NAME="$IMAGE" \
  -t "$TAG" \
  .

printf '\nBuilt image: %s\nGit commit: %s\n' "$TAG" "$COMMIT"
printf 'Inspect labels with: docker image inspect %s --format "{{json .Config.Labels}}"\n' "$TAG"
