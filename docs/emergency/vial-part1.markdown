# WebXOS 2025 Vial MCP SDK: Vial Wallet Beta Emergency Backup - Part 1 (Structure and Dependencies)

**Objective**: Set up the directory structure and dependencies for the Vial MCP Wallet Beta frontend (`vial.html` and `vialfolder`).

**Instructions for LLM**:
1. Create the `vialfolder/` directory structure under `webxos-vial-mcp/`.
2. Set up Node.js 18+ environment for build tools.
3. Use CDN-based dependencies to minimize local dependencies.
4. Configure environment variables for backend API integration.
5. Ensure compatibility with OAuth2 authentication.

## Step 1: Directory Structure
Create the following structure:
```
webxos-vial-mcp/
├── .github/workflows/
├── build/dockerfiles/
├── docs/emergency/backup/
├── vialfolder/
│   ├── js/
│   ├── css/
│   └── assets/
├── vial.html
├── .eslintrc.json
├── package.json
```

## Step 2: Dependency File

### package.json
```json
{
  "name": "vial-mcp-wallet-beta",
  "version": "1.0.0-beta",
  "scripts": {
    "lint": "eslint vialfolder/js/ --ext .js,.jsx --fix",
    "build": "cp vial.html dist/vial.html && cp -r vialfolder dist/vialfolder",
    "serve": "npx serve dist -p 3000"
  },
  "devDependencies": {
    "eslint": "^8.57.0",
    "@typescript-eslint/parser": "^6.21.0",
    "@typescript-eslint/eslint-plugin": "^6.21.0",
    "serve": "^14.2.1"
  }
}
```

## Step 3: Install Dependencies
```bash
npm install --legacy-peer-deps
```

## Step 4: Environment Configuration
Create `.env` in the root:
```text
VIAL_API_URL=http://localhost:8000
VIAL_OAUTH_REDIRECT_URI=http://localhost:3000/callback
```

## Step 5: ESLint Configuration

### .eslintrc.json
```json
{
  "env": {
    "browser": true,
    "es2021": true
  },
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended"
  ],
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "ecmaVersion": 12,
    "sourceType": "module"
  },
  "plugins": ["@typescript-eslint"],
  "rules": {
    "no-unused-vars": "warn"
  }
}
```

## Step 6: Validation
```bash
npm run lint
mkdir -p dist
npm run build
npm run serve
open http://localhost:3000/vial.html
```

**Next**: Proceed to `vial-part2.md` for core `vial.html` implementation.