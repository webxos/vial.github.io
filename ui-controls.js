```javascript
function validateInput(input) {
  const sanitized = input.replace(/[<>&]/g, '');
  if (!sanitized) throw new Error('Invalid input');
  return sanitized;
}

async function executeMode(mode, walletId, inputData) {
  try {
    const sanitizedData = validateInput(inputData);
    const response = await fetch('/api/mcp/status', {
      method: 'POST',
      body: JSON.stringify({ mode, walletId, inputData: sanitizedData })
    });
    if (!response.ok) throw new Error('API error');
    const data = await response.json();
    document.getElementById('errorMessage').textContent = data.message || 'Status updated';
  } catch (error) {
    document.getElementById('errorMessage').textContent = `Error: ${error.message}`;
  }
}
```
