set -e

echo "==> Installing dependencies..."
pip install -r backend/requirements.txt --quiet

echo "==> Installing test dependencies..."
pip install pytest pytest-cov httpx reportlab --quiet

echo "==> Running tests..."
python -m pytest "$@" --cov=backend --cov-report=term-missing
