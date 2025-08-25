# WebXOS 2025 Vial MCP SDK: Frontend Emergency Backup - Part 4 (CI/CD Workflows)

**Objective**: Set up CI/CD workflows for building, testing, and deploying the frontend.

**Instructions for LLM**:
1. Create `.github/workflows/frontend-ci.yml` for frontend CI/CD.
2. Include linting, testing, and Docker build steps.
3. Ensure compatibility with `package.json` and `frontend.Dockerfile`.

## Step 1: Create CI/CD Workflow

### .github/workflows/frontend-ci.yml
```yaml
name: Frontend CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm install --legacy-peer-deps
      - name: Run lint
        run: npm run lint
      - name: Run tests
        run: npm run test
      - name: Build Docker image
        run: docker build -f build/dockerfiles/frontend.Dockerfile -t webxos-frontend .
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: frontend-build
          path: .next/
```

## Step 2: Validation
Push to GitHub and verify the CI pipeline:
```bash
# Check GitHub Actions logs for successful build and test
```

**Next**: Proceed to `frontend-part5.md` for testing and validation.