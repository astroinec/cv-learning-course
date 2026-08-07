#!/bin/zsh
set -e
COURSE_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$COURSE_DIR/start_course.sh"
