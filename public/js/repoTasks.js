const fetch = require('node-fetch');

class RepoTasks {
  async scan(token) {
    const response = await fetch('https://api.githubcopilot.com/mcp/repos/scan', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
    });
    return response.json();
  }
}

module.exports = new RepoTasks();
