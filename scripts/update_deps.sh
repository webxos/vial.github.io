```bash
#!/bin/bash

set -e

echo "Updating Node.js dependencies..."
npm install -g npm@latest
npm update
npm audit fix --force || echo "Audit fix failed, continuing..."
npm ci
echo "Node.js dependencies updated. Generating package-lock.json..."
npm install --package-lock-only

echo "Updating Python dependencies..."
pip install --upgrade pip
pip install pip-audit
pip-audit -r requirements.txt || echo "Vulnerability check failed, review manually"
pip install -r requirements.txt --upgrade
echo "Python dependencies updated."

echo "Running lint checks..."
npm run lint
flake8 server/ --max-line-length=88 --extend-ignore=E203,W503

echo "Running tests..."
pytest server/tests/ -v --cov=server --cov-report=xml

echo "Dependency update complete."
```
