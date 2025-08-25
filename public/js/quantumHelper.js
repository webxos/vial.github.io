const { PythonShell } = require('python-shell');

class QuantumHelper {
  async optimize() {
    const options = { scriptPath: '../server/quantum/' };
    return new Promise((resolve) => {
      PythonShell.run('optimizer.py', options, (err, results) => {
        resolve(err ? { error: err.message } : { result: results });
      });
    });
  }
}

module.exports = new QuantumHelper();
