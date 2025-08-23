import { useState, useEffect } from 'react';
import { useSession, signIn, signOut } from 'next-auth/react';
import * as THREE from 'https://unpkg.com/three@0.153.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.153.0/examples/jsm/controls/OrbitControls.js';
import { create3DComponent, createConnection, setupScene } from '../public/js/threejs_integrations.js';
import styles from '../styles/Home.module.css';

export default function Home() {
  const { data: session } = useSession();
  const [walletStatus, setWalletStatus] = useState('Not authenticated');
  const [task, setTask] = useState('');
  const [components, setComponents] = useState([]);
  let scene, camera, renderer;

  useEffect(() => {
    const canvas = document.getElementById('canvas');
    const { scene: s, camera: c, renderer: r } = setupScene(canvas);
    scene = s;
    camera = c;
    renderer = r;

    const ws = new WebSocket('ws://localhost:8000/mcp/ws');
    ws.onopen = () => {
      if (session?.accessToken) {
        ws.send(JSON.stringify({ token: session.accessToken }));
      }
    };
    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (data.result?.circuit) {
        const component = create3DComponent(scene, {
          id: data.request_id,
          type: 'quantum',
          title: 'Quantum Circuit',
          position: { x: Math.random() * 4 - 2, y: Math.random() * 4 - 2, z: 0 },
          svg: data.result.circuit
        });
        setComponents((prev) => [...prev, component]);
      }
      if (data.result?.status) {
        setWalletStatus(`Balance: ${data.result.balance} WebXOS, Active: ${data.result.active}`);
      }
      if (data.result?.yaml_result) {
        console.log('YAML Workflow Result:', data.result.yaml_result);
      }
    };
    ws.onerror = (error) => console.error('WebSocket error:', error);
    return () => ws.close();
  }, [session]);

  const handleTaskSubmit = async () => {
    if (!session) {
      alert('Please sign in');
      return;
    }
    const ws = new WebSocket('ws://localhost:8000/mcp/ws');
    ws.onopen = () => {
      ws.send(JSON.stringify({
        token: session.accessToken,
        tool: task.includes('quantum') ? 'quantum.circuit.build' : task.includes('yaml') ? 'yaml.workflow.execute' : 'vial.status.get',
        params: task.includes('quantum') ? { qubits: 2, gates: ['h', 'cx'] } :
                task.includes('yaml') ? { yaml_content: 'steps:\n  - task: vial_status_get\n    params:\n      vial_id: vial_123' } :
                { vial_id: 'vial_123' }
      }));
    };
  };

  const handleDragDrop = (event, componentId) => {
    const { clientX, clientY } = event;
    const component = components.find(c => c.userData.id === componentId);
    if (component) {
      component.position.set(clientX / 100 - 2, -clientY / 100 + 2, 0);
    }
  };

  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <h1 className={styles.title}>Vial MCP Controller - SVG Editor</h1>
        <div className={styles.controls}>
          {session ? (
            <>
              <p>Signed in as {session.user.name}</p>
              <button onClick={() => signOut()}>Sign out</button>
            </>
          ) : (
            <button onClick={() => signIn()}>Sign in</button>
          )}
          <input
            type="text"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Enter task (e.g., Generate quantum circuit, Run YAML)"
            className={styles.input}
          />
          <button onClick={handleTaskSubmit} className={styles.button}>Submit Task</button>
          <p>Wallet: {walletStatus}</p>
        </div>
        <canvas id="canvas" className={styles.canvas} onMouseMove={(e) => handleDragDrop(e, components[0]?.userData.id)}></canvas>
      </main>
    </div>
  );
}
