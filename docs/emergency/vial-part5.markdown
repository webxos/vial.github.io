# WebXOS 2025 Vial MCP SDK: Vial Wallet Beta Emergency Backup - Part 5 (CI/CD and Testing)

**Objective**: Set up CI/CD workflows and testing for the Vial MCP Wallet Beta frontend.

**Instructions for LLM**:
1. Create `.github/workflows/vial-ci.yml` for CI/CD.
2. Create `vialfolder/js/__tests__/wallet.test.js` for Jest tests.
3. Create `scripts/validate-vial.sh` for validation.
4. Ensure tests cover wallet functionality and OAuth2 integration.

## Step 1: Create CI/CD Workflow

### .github/workflows/vial-ci.yml
```yaml
name: Vial Frontend CI
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
      - name: Build
        run: npm run build
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: vial-build
          path: dist/
```

## Step 2: Create Test File

### vialfolder/js/__tests__/wallet.test.js
```javascript
describe('Wallet Functions', () => {
  beforeEach(() => {
    global.indexedDB = { open: jest.fn().mockReturnValue({
      onupgradeneeded: null,
      onsuccess: null,
      onerror: null
    }) };
  });

  it('initializes IndexedDB', async () => {
    const mockDB = { transaction: jest.fn().mockReturnValue({ objectStore: jest.fn() }) };
    indexedDB.open.mockReturnValue({
      onupgradeneeded: (cb) => cb({ target: { result: { createObjectStore: jest.fn() } } }),
      onsuccess: (cb) => cb({ target: { result: mockDB } }),
      onerror: null
    });
    const db = await import('../wallet.js').then(m => m.initDB());
    expect(db).toBeDefined();
  });
});
```

## Step 3: Create Validation Script

### scripts/validate-vial.sh
```bash
#!/bin/bash
set -e
echo "Starting Vial frontend validation..."
npm run lint
npm run build
echo "Starting Vial frontend server..."
npx serve dist -p 3000 &
sleep 5
curl http://localhost:3000/vial.html -o /dev/null -s -w "%{http_code}\n" | grep 200 || { echo "Vial frontend failed to start"; exit 1; }
echo "Vial frontend validation successful!"
```

## Step 4: Update package.json
Add to `package.json` scripts:
```json
{
  "scripts": {
    "validate": "bash scripts/validate-vial.sh"
  }
}
```

## Step 5: Validation
```bash
chmod +x scripts/validate-vial.sh
npm run validate
```

**Completion**: Vial MCP Wallet Beta frontend rebuild complete.