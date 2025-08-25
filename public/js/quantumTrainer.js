const { PythonShell } = require('python-shell');

class QuantumTrainer {
  async trainModel(data) {
    const options = { scriptPath: '../server/quantum/', args: [JSON.stringify(data)] };
    return new Promise((resolve) => {
      PythonShell.run('trainer.py', options, (err, results) => {
        resolve(err ? { error: err.message } : { result: results });
      });
    });
  }
}

module.exports = new QuantumTrainer();
