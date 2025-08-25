const fetch = require('node-fetch');

class TranslatorAgent {
  async handleTranslation(message, target) {
    const response = await fetch('https://api.githubcopilot.com/mcp/translate', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${process.env.GITHUB_TOKEN}` },
      body: JSON.stringify({ message, target }),
    });
    return response.json();
  }
}

module.exports = new TranslatorAgent();
