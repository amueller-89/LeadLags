#!/bin/bash
# Quick script to fix all ruff warnings in the project
# Usage: ./ruff-fix.sh

echo "Fixing all ruff linting issues..."
python3 -m ruff check --fix .

echo ""
echo "Formatting code..."
python3 -m ruff format .

echo ""
echo "✅ Done! All warnings should be fixed."

