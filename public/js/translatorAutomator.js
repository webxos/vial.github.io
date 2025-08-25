const cron = require('node-cron');
const TranslatorAgent = require('./translatorAgent');

class TranslatorAutomator {
  automateTranslation() {
    cron.schedule('*/30 * * * *', async () => {
      const result = await TranslatorAgent.handleTranslation('Update status', 'en');
      console.log('Automated translation:', result);
    });
  }
}

module.exports = new TranslatorAutomator();
