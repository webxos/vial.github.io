```javascript
async function updateModeStatus() {
    const response = await fetch('/api/mode', { method: 'GET' });
    const data = await response.json();
    document.getElementById('modeStatus').textContent = `Mode: ${data.status || 'Unknown'} at ${new Date().toLocaleString('en-US', { timeZone: 'America/New_York' })}`;
}

async function updateUserStatus() {
    const response = await fetch('/api/mcp/status');
    const data = await response.json();
    document.getElementById('userStatus').textContent = `User Status: ${data.message || 'Unknown'} at ${new Date().toLocaleString('en-US', { timeZone: 'America/New_York' })}`;
}

document.addEventListener('DOMContentLoaded', () => {
    updateModeStatus();
    updateUserStatus();
    setInterval(updateModeStatus, 5000);
    setInterval(updateUserStatus, 5000);
});
```
