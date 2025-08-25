# WebXOS 2025 Vial MCP SDK: Frontend Emergency Backup - Part 1 (Dependencies and Structure)

**Objective**: Set up frontend dependencies and directory structure for the WebXOS frontend using React, Next.js, and Tailwind CSS.

**Instructions for LLM**:
1. Create the frontend directory structure under `webxos-vial-mcp/`.
2. Set up Node.js 18+ environment.
3. Install dependencies via `package.json`.
4. Configure environment variables for backend API integration.
5. Ensure compatibility with OAuth2 authentication.

## Step 1: Directory Structure
Create the following structure:
```
webxos-vial-mcp/
├── .github/workflows/
├── build/dockerfiles/
├── docs/emergency/backup/
├── public/
│   ├── js/
│   └── css/
├── pages/
│   ├── api/
│   └── _app.js
├── .eslintrc.json
├── next.config.js
├── package.json
```

## Step 2: Dependency File

### package.json
```json
{
  "name": "webxos-vial-mcp-frontend",
  "version": "1.2.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint public/js/ pages/ --ext .js,.jsx,.ts,.tsx --fix",
    "test": "jest"
  },
  "dependencies": {
    "next": "^14.2.5",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "next-auth": "^4.24.7",
    "jose": "^5.6.3",
    "axios": "^1.7.2",
    "tailwindcss": "^3.4.10",
    "postcss": "^8.4.40",
    "autoprefixer": "^10.4.20"
  },
  "devDependencies": {
    "eslint": "^8.57.0",
    "@typescript-eslint/parser": "^6.21.0",
    "@typescript-eslint/eslint-plugin": "^6.21.0",
    "jest": "^29.7.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.4.8"
  }
}
```

## Step 3: Install Dependencies
```bash
npm install --legacy-peer-deps
```

## Step 4: Environment Configuration
Create `.env.local` with:
```text
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_OAUTH_REDIRECT_URI=http://localhost:3000/api/auth/callback/google
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret
```

## Step 5: ESLint Configuration

### .eslintrc.json
```json
{
  "env": {
    "browser": true,
    "es2021": true,
    "node": true
  },
  "extends": [
    "eslint:recommended",
    "plugin:react/recommended",
    "plugin:@typescript-eslint/recommended"
  ],
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "ecmaVersion": 12,
    "sourceType": "module"
  },
  "plugins": ["react", "@typescript-eslint"],
  "rules": {
    "react/prop-types": "off"
  }
}
```

## Step 6: Next.js Configuration

### next.config.js
```javascript
module.exports = {
  reactStrictMode: true,
  env: {
    API_URL: process.env.NEXT_PUBLIC_API_URL
  }
};
```

## Validation
```bash
npm run lint
npm run dev
open http://localhost:3000
```

**Next**: Proceed to `frontend-part2.md` for core frontend application.