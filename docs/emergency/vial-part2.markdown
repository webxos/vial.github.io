# WebXOS 2025 Vial MCP SDK: Vial Wallet Beta Emergency Backup - Part 2 (Core vial.html)

**Objective**: Implement `vial.html` as a standalone HTML file with React for the Vial MCP Wallet Beta frontend.

**Instructions for LLM**:
1. Create `vial.html` with embedded React and Tailwind CSS via CDNs.
2. Implement basic wallet UI with login and wallet creation.
3. Ensure minimal dependencies and compatibility with backend APIs.
4. Use a black and neon green cyberpunk aesthetic.

## Step 1: Create Core HTML File

### vial.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vial MCP Wallet Beta</title>
  <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/axios@1.7.2/dist/axios.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/jose@5.6.3/dist/browser/index.js"></script>
  <style>
    body { background-color: #000; color: #00ff00; font-family: 'Courier New', monospace; }
    .error { color: #ff0000; }
  </style>
</head>
<body>
  <div id="root" class="container mx-auto p-4"></div>
  <script type="module">
    const { useState, useEffect } = React;
    const { render } = ReactDOM;

    const App = () => {
      const [token, setToken] = useState(null);
      const [wallet, setWallet] = useState(null);
      const [error, setError] = useState('');

      const login = () => {
        axios.get(`${process.env.VIAL_API_URL}/mcp/auth/login`)
          .then(({ data }) => {
            window.location.href = `${data.auth_url}&state=${data.code_verifier}`;
          })
          .catch(err => setError('Login failed: ' + err.message));
      };

      const handleCallback = () => {
        const urlParams = new URLSearchParams(window.location.search);
        const code = urlParams.get('code');
        const state = urlParams.get('state');
        if (code && state) {
          axios.post(`${process.env.VIAL_API_URL}/mcp/auth/token`, { code, code_verifier: state })
            .then(({ data }) => setToken(data.access_token))
            .catch(err => setError('Token exchange failed: ' + err.message));
        }
      };

      const createWallet = () => {
        const address = '0x' + Array(40).fill().map(() => Math.floor(Math.random() * 16).toString(16)).join('');
        const privateKey = Array(64).fill().map(() => Math.floor(Math.random() * 16).toString(16)).join('');
        axios.post(`${process.env.VIAL_API_URL}/mcp/wallet/create`, { address, private_key: privateKey, balance: 0.0 }, {
          headers: { Authorization: `Bearer ${token}` }
        })
          .then(({ data }) => setWallet(data))
          .catch(err => setError('Wallet creation failed: ' + err.message));
      };

      useEffect(() => {
        handleCallback();
      }, []);

      return (
        <div className="text-green-500">
          <h1 className="text-3xl font-bold mb-4">Vial MCP Wallet Beta</h1>
          {error && <p className="error">{error}</p>}
          {!token ? (
            <button className="bg-green-500 text-black p-2 rounded" onClick={login}>Login with Google</button>
          ) : (
            <div>
              <button className="bg-red-500 text-black p-2 rounded mr-2" onClick={() => setToken(null)}>Logout</button>
              <button className="bg-green-500 text-black p-2 rounded" onClick={createWallet}>Create Wallet</button>
              {wallet && (
                <div className="mt-4">
                  <p>Address: {wallet.address}</p>
                  <p>Balance: {wallet.balance}</p>
                </div>
              )}
            </div>
          )}
        </div>
      );
    };

    render(<App />, document.getElementById('root'));
  </script>
</body>
</html>
```

## Step 2: Validation
```bash
cp vial.html dist/vial.html
npx serve dist -p 3000
open http://localhost:3000/vial.html
```

**Next**: Proceed to `vial-part3.md` for wallet API integration and offline support.