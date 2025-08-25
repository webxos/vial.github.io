# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 10 (Frontend Integration)

**Objective**: Implement a React and Next.js frontend for the WebXOS backend.

**Instructions for LLM**:
1. Create `index.html` and `public/js/app.jsx` for the frontend.
2. Use Next.js with Tailwind CSS for styling.
3. Integrate with backend APIs using OAuth2.
4. Ensure compatibility with `package.json`.

## Step 1: Create Frontend Files

### index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WebXOS 2025 Vial MCP SDK</title>
  <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/next@14.2.5/dist/next.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/jose@5.6.3/dist/browser/index.js"></script>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/js/app.jsx"></script>
</body>
</html>
```

### public/js/app.jsx
```jsx
import React, { useState, useEffect } from 'react';
import { render } from 'react-dom';
import { jwtDecode } from 'jose';

const App = () => {
  const [token, setToken] = useState(null);
  const [launches, setLaunches] = useState([]);

  const login = async () => {
    const response = await fetch('http://localhost:8000/mcp/auth/login');
    const { auth_url, code_verifier } = await response.json();
    window.location.href = `${auth_url}&state=${code_verifier}`;
  };

  const fetchLaunches = async () => {
    const response = await fetch('http://localhost:8000/mcp/spacex/launches?limit=5', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    const data = await response.json();
    setLaunches(data);
  };

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    const state = urlParams.get('state');
    if (code && state) {
      fetch('http://localhost:8000/mcp/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, code_verifier: state })
      })
        .then(res => res.json())
        .then(data => setToken(data.access_token));
    }
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold">WebXOS 2025 Vial MCP SDK</h1>
      {!token ? (
        <button className="bg-blue-500 text-white p-2 rounded" onClick={login}>Login with Google</button>
      ) : (
        <div>
          <button className="bg-green-500 text-white p-2 rounded" onClick={fetchLaunches}>Fetch SpaceX Launches</button>
          <ul className="mt-4">
            {launches.map((launch, idx) => (
              <li key={idx} className="p-2 border-b">{launch.name} - {launch.date_utc}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

render(<App />, document.getElementById('root'));
```

## Step 2: Validation
```bash
npm run dev
open http://localhost:3000
```

**Completion**: Advanced backend and frontend rebuild complete.