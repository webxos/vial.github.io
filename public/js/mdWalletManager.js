const fs = require('fs');

class MDWalletManager {
  exportWallet(walletId, data) {
    fs.writeFileSync(`.md_wallets/${walletId}.md`, JSON.stringify(data), 'utf8');
  }

  importWallet(walletId) {
    const file = `.md_wallets/${walletId}.md`;
    return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
  }
}

module.exports = new MDWalletManager();
