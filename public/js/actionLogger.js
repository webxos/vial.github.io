const fs = require('fs');

class ActionLogger {
  logAction(action, details) {
    const logEntry = `${new Date().toISOString()} - ${action}: ${JSON.stringify(details)}\n`;
    fs.appendFileSync('logs/action.log', logEntry, 'utf8');
  }
}

module.exports = new ActionLogger();
