#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v python3 >/dev/null 2>&1; then
    printf 'Python 3 is required.\n' >&2
    exit 1
fi
if [[ "${1:-}" == "--cli" ]]; then
    shift
    exec python3 -I capture_menu.py "$@"
fi
if python3 -I -c 'import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy()' >/dev/null 2>&1; then
    exec python3 -I capture_gui.py "$@"
fi
printf 'No usable Tk display; starting the terminal menu.\n'
exec python3 -I capture_menu.py "$@"
