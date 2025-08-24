```javascript
import axios from 'axios';

class MCPClient {
  constructor(baseUrl = '/api/mcp') {
    this.api = axios.create({ baseURL: baseUrl });
    this.tools = [];
  }

  async init() {
    try {
      const response = await this.api.get('/tools');
      this.tools = response.data;
      console.log('MCP Tools:', this.tools);
    } catch (error) {
      console.error('MCP Init Error:', error.message);
    }
  }

  async listTools() {
    return this.tools;
  }

  async runQuantumCircuit(qasmCode) {
    try {
      const response = await this.api.post('/tools/quantum_sync', { qasm: qasmCode });
      return response.data.result;
    } catch (error) {
      console.error('Quantum Circuit Error:', error.message);
      throw error;
    }
  }

  async fetchNASADataset(query) {
    try {
      const response = await this.api.post('/resources/nasa_data', { query });
      return response.data.results;
    } catch (error) {
      console.error('NASA Data Error:', error.message);
      throw error;
    }
  }
}

export default new MCPClient();
