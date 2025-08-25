const { PythonShell } = require('python-shell');

class QuantumCopilot {
  async executeQuery(query) {
    const options = { scriptPath: '../server/quantum/', args: [query] };
    return new Promise((resolve) => {
      PythonShell.run('copilot.py', options, (err, results) => {
        resolve(err ? { error: err.message } : { result: results });
      });
    });
  }
}

module.exports = new QuantumCopilot();
