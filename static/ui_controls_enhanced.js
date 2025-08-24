```javascript
async function updateModeStatus() {
    const response = await fetch('/api/mode', { method: 'GET' });
    const data = await response.json();
    document.getElementById('modeStatus').textContent = `Mode: ${data.status || 'Unknown'} at ${new Date().toLocaleString('en-US', { timeZone: 'America/New_York' })}`;
}

async function updateAgoraFeedback() {
    const response = await fetch('/api/agora/send', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: document.getElementById('mode').value})
    });
    const agoraData = await response.json();
    document.getElementById('agoraFeedback').textContent = `Agora: ${agoraData.status || 'No data'} at ${new Date().toLocaleString('en-US', { timeZone: 'America/New_York' })}`;
}

document.addEventListener('DOMContentLoaded', () => {
    updateModeStatus();
    updateAgoraFeedback();
    setInterval(updateModeStatus, 5000);
    setInterval(updateAgoraFeedback, 5000);
    document.getElementById('switchMode').addEventListener('click', () => {
        updateModeStatus();
        updateAgoraFeedback();
    });
});
```
