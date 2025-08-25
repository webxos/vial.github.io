const fetch = require('node-fetch');

class OAuthSync {
  constructor(token) {
    this.token = token;
    this.url = 'https://api.githubcopilot.com/mcp/oauth/sync';
  }

  async sync() {
    const response = await fetch(this.url, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.token}` },
    });
    return response.json();
  }
}

module.exports = OAuthSync;
