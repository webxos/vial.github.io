import React, { useState, useEffect } from 'react';
import { render } from 'react-dom';
import * as d3 from 'd3';

const App = () => {
  const [diagram, setDiagram] = useState(null);
  const [task, setTask] = useState('');
  const [ws, setWs] = useState(null);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8000/mcp/ws');
    socket.onopen = () => {
      console.log('WebSocket connected');
      socket.send(JSON.stringify({
        token: localStorage.getItem('token')
      }));
    };
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.result && data.result.circuit) {
        setDiagram(data.result.circuit);
      }
      console.log('WebSocket message:', data);
    };
    socket.onerror = (error) => console.error('WebSocket error:', error);
    setWs(socket);

    const fetchDiagram = async () => {
      try {
        const response = await fetch('/visual/diagram/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            components: [{ id: '1', type: 'api_endpoint', title: 'API', position: { x: 50, y: 50, z: 0 } }],
            connections: []
          })
        });
        const data = await response.json();
        setDiagram(data.svg_content);
      } catch (error) {
        console.error('Error fetching diagram:', error);
      }
    };
    fetchDiagram();

    return () => socket.close();
  }, []);

  const handleTaskSubmit = async () => {
    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          tool: task.includes('quantum') ? 'quantum.circuit.build' : 'vial.status.get',
          params: task.includes('quantum') ? { qubits: 2, gates: ['h', 'cx'] } : { vial_id: 'vial_123' }
        }));
      }
      const response = await fetch('/swarm/task', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ task, context: { user: 'anonymous' } })
      });
      const data = await response.json();
      console.log('Swarm task result:', data.result);
    } catch (error) {
      console.error('Error submitting task:', error);
    }
  };

  return (
    <div>
      <h1>Vial MCP Controller</h1>
      <div>
        <input
          type="text"
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Enter task (e.g., Generate quantum circuit)"
        />
        <button onClick={handleTaskSubmit}>Submit Task</button>
      </div>
      {diagram && (
        <div
          style={{ width: '100%', height: '500px' }}
          dangerouslySetInnerHTML={{ __html: diagram }}
        />
      )}
    </div>
  );
};

render(<App />, document.getElementById('root'));
