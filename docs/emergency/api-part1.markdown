# WebXOS 2025 Vial MCP SDK: API Emergency Backup - Part 1 (Dependency CDNs and Forking)

**Objective**: Set up API dependency CDNs and guide forking the WebXOS repository for API development.

**Instructions for LLM**:
1. Fork the repository from `https://github.com/webxos/webxos-vial-mcp`.
2. Set up CDN-based dependencies for frontend API clients and backend libraries.
3. Configure environment for API development with Node.js and Python.
4. Ensure compatibility with OAuth2 and existing backend APIs.

## Step 1: Fork the Repository
```bash
git clone https://github.com/webxos/webxos-vial-mcp.git
cd webxos-vial-mcp
git remote add origin https://github.com/<your-username>/webxos-vial-mcp.git
git push -u origin main
```

## Step 2: Dependency CDNs
Create `public/js/lib/cdn.js` to load API client dependencies via CDN:
```javascript
const loadCDN = () => {
  const scripts = [
    'https://cdn.jsdelivr.net/npm/axios@1.7.2/dist/axios.min.js',
    'https://cdn.jsdelivr.net/npm/jose@5.6.3/dist/browser/index.js',
  ];
  scripts.forEach(src => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    document.head.appendChild(script);
  });
};
loadCDN();
```

## Step 3: Update package.json
```json
{
  "name": "webxos-vial-mcp-api",
  "version": "1.2.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint public/js/ pages/ --ext .js,.jsx,.ts,.tsx --fix"
  },
  "dependencies": {
    "next": "^14.2.5",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "next-auth": "^4.24.7",
    "axios": "^1.7.2",
    "tailwindcss": "^3.4.10",
    "postcss": "^8.4.40",
    "autoprefixer": "^10.4.20"
  },
  "devDependencies": {
    "eslint": "^8.57.0",
    "@typescript-eslint/parser": "^6.21.0",
    "@typescript-eslint/eslint-plugin": "^6.21.0"
  }
}
```

## Step 4: Environment Configuration
Update `.env.local`:
```text
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_OAUTH_REDIRECT_URI=http://localhost:3000/api/auth/callback/google
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret
```

## Step 5: Validation
```bash
npm install --legacy-peer-deps
npm run dev
curl http://localhost:3000
```

**Next**: Proceed to `api-part2.md` for FastAPI REST endpoints.