#!/usr/bin/env bash
set -euo pipefail

REFINE="${1:-false}"   # primer argumento: valor del checkbox (true/false)
shift                  # el resto de los argumentos forman el comando base

EXTRA_FLAG=""
case "$REFINE" in
  1|true|True|TRUE) EXTRA_FLAG="--refine" ;;
esac

# Ejecuta el comando que sigue, agregando --refine si corresponde
"$@" $EXTRA_FLAG
