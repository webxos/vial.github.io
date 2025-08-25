const cron = require('node-cron');

class RepoScheduler {
  scheduleTask(task, time) {
    cron.schedule(time, () => {
      console.log(`Running task: ${task}`);
      require('./repoTasks').scan(process.env.GITHUB_TOKEN);
    });
  }
}

module.exports = new RepoScheduler();
