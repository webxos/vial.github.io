const fs = require('fs');
const fetch = require('node-fetch');

class MDWalletSync {
  async syncWallet(walletId, token) {
    const data = this.importWallet(walletId);
    if (data) {
      const response = await fetch('https://api.githubcopilot.com/mcp/wallets/sync', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ walletId, data }),
      });
      return response.json();
    }
    return { error: 'Wallet not found' };
  }

  importWallet(walletId) {
    const file = `.md_wallets/${walletId}.md`;
    return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
  }
}

module.exports = new MDWalletSync();
