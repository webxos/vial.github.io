```typescript
import * as vscode from 'vscode';
import axios from 'axios';
import * as crypto from 'crypto-js';
import * as fs from 'fs';
import * as path from 'path';
import Database from 'better-sqlite3';
import { Kyber512 } from 'crypto-kyber';

export function activate(context: vscode.ExtensionContext) {
  const config = vscode.workspace.getConfiguration('webxos-2025');
  const apiEndpoint = config.get('apiEndpoint', 'http://localhost:3000');
  const walletDir = config.get('walletDir', path.join(process.env.HOME || '', '.webxos/wallets'));
  const db = new Database(path.join(walletDir, 'webxos.db'));
  db.exec('CREATE TABLE IF NOT EXISTS wallets (id TEXT PRIMARY KEY, token TEXT, exported BOOLEAN)');

  if (!fs.existsSync(walletDir)) {
    fs.mkdirSync(walletDir, { recursive: true });
  }

  let token: string | null = null;

  // Initialize Wallet
  context.subscriptions.push(vscode.commands.registerCommand('webxos-2025.initWallet', async () => {
    try {
      const walletId = crypto.lib.WordArray.random(16).toString();
      const kyber = new Kyber512();
      const [publicKey, privateKey] = kyber.keypair();
      const response = await axios.post(`${apiEndpoint}/mcp/auth`, { walletId, publicKey: publicKey.toString('hex') });
      token = response.data.access_token;
      db.prepare('INSERT INTO wallets (id, token, exported) VALUES (?, ?, ?)').run(walletId, token, false);
      const walletContent = `WebXOS Wallet\nID: ${walletId}\nToken: ${token}`;
      await vscode.workspace.fs.writeFile(
        vscode.Uri.file(path.join(walletDir, `${walletId}.mdwallet`)),
        Buffer.from(walletContent)
      );
      vscode.window.showInformationMessage(`Wallet ${walletId} initialized. Export before closing terminal.`);
    } catch (error) {
      vscode.window.showErrorMessage(`Wallet initialization failed: ${error.message}`);
    }
  }));

  // Run Quantum RAG
  context.subscriptions.push(vscode.commands.registerCommand('webxos-2025.runQuantumRAG', async () => {
    if (!token) return vscode.window.showErrorMessage('Initialize wallet first.');
    const query = await vscode.window.showInputBox({ prompt: 'Enter RAG query' });
    const circuit = await vscode.window.showInputBox({ prompt: 'Enter QASM circuit' });
    if (query && circuit) {
      try {
        const response = await axios.post(`${apiEndpoint}/mcp/quantum_rag`, { query, quantum_circuit: circuit, max_results: 5 }, {
          headers: { Authorization: `Bearer ${token}` }
        });
        vscode.window.showInformationMessage(`Quantum RAG Results: ${JSON.stringify(response.data.results)}`);
      } catch (error) {
        vscode.window.showErrorMessage(`Quantum RAG failed: ${error.response?.data?.detail || error.message}`);
      }
    }
  }));

  // Create ServiceNow Ticket
  context.subscriptions.push(vscode.commands.registerCommand('webxos-2025.createServiceNowTicket', async () => {
    if (!token) return vscode.window.showErrorMessage('Initialize wallet first.');
    const shortDescription = await vscode.window.showInputBox({ prompt: 'Enter ticket short description' });
    const description = await vscode.window.showInputBox({ prompt: 'Enter ticket description' });
    if (shortDescription && description) {
      try {
        const response = await axios.post(`${apiEndpoint}/mcp/servicenow/ticket`, {
          short_description: shortDescription,
          description,
          urgency: 'low'
        }, { headers: { Authorization: `Bearer ${token}` } });
        vscode.window.showInformationMessage(`ServiceNow Ticket Created: ${response.data.result.number}`);
      } catch (error) {
        vscode.window.showErrorMessage(`ServiceNow ticket failed: ${error.response?.data?.detail || error.message}`);
      }
    }
  }));

  // Initialize OBS Scene
  context.subscriptions.push(vscode.commands.registerCommand('webxos-2025.initOBSScene', async () => {
    if (!token) return vscode.window.showErrorMessage('Initialize wallet first.');
    const sceneName = await vscode.window.showInputBox({ prompt: 'Enter OBS scene name' });
    if (sceneName) {
      try {
        const response = await axios.post(`${apiEndpoint}/mcp/tools/obs.init`, { scene_name: sceneName }, {
          headers: { Authorization: `Bearer ${token}` }
        });
        vscode.window.showInformationMessage(`OBS Scene Initialized: ${response.data.scene}`);
      } catch (error) {
        vscode.window.showErrorMessage(`OBS scene initialization failed: ${error.response?.data?.detail || error.message}`);
      }
    }
  }));

  // Check System Health
  context.subscriptions.push(vscode.commands.registerCommand('webxos-2025.checkHealth', async () => {
    if (!token) return vscode.window.showErrorMessage('Initialize wallet first.');
    try {
      const response = await axios.get(`${apiEndpoint}/mcp/monitoring/health`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      vscode.window.showInformationMessage(`System Health: ${JSON.stringify(response.data)}`);
    } catch (error) {
      vscode.window.showErrorMessage(`Health check failed: ${error.response?.data?.detail || error.message}`);
    }
  }));

  // Erase unexported wallets on terminal close
  context.subscriptions.push(vscode.window.onDidCloseTerminal(() => {
    const wallets = db.prepare('SELECT id, exported FROM wallets').all();
    for (const wallet of wallets) {
      if (!wallet.exported) {
        fs.unlinkSync(path.join(walletDir, `${wallet.id}.mdwallet`));
        db.prepare('DELETE FROM wallets WHERE id = ?').run(wallet.id);
      }
    }
    db.close();
  }));

  // Export wallet
  context.subscriptions.push(vscode.commands.registerCommand('webxos-2025.exportWallet', async () => {
    const walletId = await vscode.window.showInputBox({ prompt: 'Enter wallet ID to export' });
    if (walletId) {
      db.prepare('UPDATE wallets SET exported = ? WHERE id = ?').run(true, walletId);
      vscode.window.showInformationMessage(`Wallet ${walletId} exported.`);
    }
  }));
}

export function deactivate() {
  // Clean up resources if needed
}
```
