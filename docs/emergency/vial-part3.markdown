# WebXOS 2025 Vial MCP SDK: Vial Wallet Beta Emergency Backup - Part 3 (Wallet API and Offline Support)

**Objective**: Enhance `vial.html` with wallet API integration and offline support using IndexedDB.

**Instructions for LLM**:
1. Create `vialfolder/js/wallet.js` for wallet API interactions and offline storage.
2. Update `vial.html` to include offline wallet caching.
3. Ensure secure integration with backend wallet APIs via OAuth2.
4. Handle errors with red text in the UI.

## Step 1: Create Wallet JavaScript File

### vialfolder/js/wallet.js
```javascript
const initDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('VialWalletDB', 1);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      db.createObjectStore('wallets', { keyPath: 'address' });
    };
    request.onsuccess = (event) => resolve(event.target.result);
    request.onerror = (event) => reject(event.target.error);
  });
};

const saveWalletOffline = async (wallet) => {
  const db = await initDB();
  const tx = db.transaction('wallets', 'readwrite');
  const store = tx.objectStore('wallets');
  store.put(wallet);
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

const getWalletOffline = async (address) => {
  const db = await initDB();
  const tx = db.transaction('wallets', 'readonly');
  const store = tx.objectStore('wallets');
  return new Promise((resolve, reject) => {
    const request = store.get(address);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const fetchWallet = async (token, address) => {
  try {
    const response = await axios.get(`${process.env.VIAL_API_URL}/mcp/wallet/${address}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    await saveWalletOffline(response.data);
    return response.data;
  } catch (error) {
    const offlineWallet = await getWalletOffline(address);
    if (offlineWallet) return offlineWallet;
    throw error;
  }
};
```

## Step 2: Update vial.html
Update `<script>` section in `vial.html`:
```html
<script type="module">
  import { fetchWallet, saveWalletOffline } from './vialfolder/js/wallet.js';
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
        .then(async ({ data }) => {
          await saveWalletOffline(data);
          setWallet(data);
        })
        .catch(err => setError('Wallet creation failed: ' + err.message));
    };

    const loadWallet = (address) => {
      fetchWallet(token, address)
        .then(data => setWallet(data))
        .catch(err => setError('Wallet fetch failed: ' + err.message));
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
            <button className="bg-green-500 text-black p-2 rounded mr-2" onClick={createWallet}>Create Wallet</button>
            <input
              type="text"
              placeholder="Enter wallet address"
              className="p-2 bg-black border border-green-500 text-green-500"
              onChange={(e) => loadWallet(e.target.value)}
            />
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
```

## Step 3: Validation
```bash
cp vial.html dist/vial.html
cp -r vialfolder dist/vialfolder
npx serve dist -p 3000
open http://localhost:3000/vial.html
# Test wallet creation and offline retrieval
```

**Next**: Proceed to `vial-part4.md` for Docker deployment.