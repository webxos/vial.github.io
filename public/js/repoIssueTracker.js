const fetch = require('node-fetch');

class RepoIssueTracker {
  async trackIssues(token) {
    const response = await fetch('https://api.githubcopilot.com/mcp/repos/webxos/webxos-vial-mcp/issues', {
      headers: { 'Authorization': `Bearer ${token}` },
    });
    const issues = await response.json();
    return issues.filter(issue => issue.state === 'open');
  }
}

module.exports = new RepoIssueTracker();
