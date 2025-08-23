import { useEffect, useRef, useState } from 'react';
import { setupScene, create3DComponent, createConnection } from '../public/js/threejs_integrations';
import styles from '../styles/Home.module.css';

export default function Home() {
  const canvasRef = useRef(null);
  const [walletData, setWalletData] = useState(null);
  const [agents, setAgents] = useState([]);
  const [menuInfo, setMenuInfo] = useState({ selectedTool: 'agent', guideStep: 1 });

  useEffect(() => {
    const saveSession = async () => {
      await fetch('/v1/save_session', {
        method: 'POST',
        headers: { 'Authorization': 'Bearer test_token', 'Content-Type': 'application/json' },
        body: JSON.stringify({ menu_info: menuInfo, build_progress: agents })
      });
    };
    saveSession();
  }, [menuInfo, agents]);

  useEffect(() => {
    if (canvasRef.current) {
      const { scene, camera, renderer } = setupScene(canvasRef.current);
      const fetchWallet = async () => {
        const response = await fetch('/v1/wallet/export', {
          method: 'POST',
          headers: { 'Authorization': 'Bearer test_token', 'Content-Type': 'application/json' },
          body: JSON.stringify({ network_id: '54965687-3871-4f3d-a803-ac9840af87c4' })
        });
        const data = await response.json();
        setWalletData(data);
        if (data.markdown) {
          const vials = ['vial1', 'vial2', 'vial3', 'vial4'];
          const agentNodes = vials.map((vial, i) => {
            return create3DComponent(scene, {
              id: vial,
              type: 'vial_node',
              title: vial,
              position: { x: i * 2 - 3, y: 0, z: 0 }
            });
          });
          createConnection(scene, agentNodes[0].position, agentNodes[1].position);
          createConnection(scene, agentNodes[1].position, agentNodes[2].position);
          createConnection(scene, agentNodes[2].position, agentNodes[3].position);
          setAgents(vials);
        }
        renderer.render(scene, camera);
      };
      fetchWallet();
    }
  }, []);

  const handleAddAgent = async () => {
    const response = await fetch('/v1/execute_svg_task', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer test_token', 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_name: 'create_agent',
        params: { vial_id: `vial${agents.length + 1}`, x_position: 0, y_position: 0 }
      })
    });
    const data = await response.json();
    if (data.status === 'success') {
      setAgents([...agents, `vial${agents.length + 1}`]);
      setMenuInfo({ ...menuInfo, guideStep: menuInfo.guideStep + 1 });
      alert('Agent created via SVG UI');
    }
  };

  const handleTroubleshoot = async () => {
    const response = await fetch('/v1/troubleshoot/status', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer test_token', 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: 'test_token' })
    });
    const data = await response.json();
    if (data.status === 'reset') {
      setMenuInfo({ selectedTool: 'agent', guideStep: 1 });
      setAgents([]);
      alert('Session reset via troubleshoot');
    }
  };

  return (
    <div className={styles.container}>
      <h1 className="text-3xl font-bold mb-4">Vial MCP Controller</h1>
      <div className={styles.console}>
        <p>Guide: {menuInfo.guideStep === 1 ? 'Select a tool to add agents or endpoints' : 'Drag and drop to position'}</p>
        <p className={styles.balance}>
          $WEBXOS Balance: {walletData ? walletData.balance || '0.0000' : '0.0000'} | Reputation: 0
        </p>
      </div>
      <div className={styles['button-group']}>
        <button onClick={() => setMenuInfo({ ...menuInfo, selectedTool: 'agent' })}>Agent Tool</button>
        <button onClick={() => setMenuInfo({ ...menuInfo, selectedTool: 'endpoint' })}>Endpoint Tool</button>
        <button onClick={handleAddAgent}>Add Agent</button>
        <button onClick={handleTroubleshoot}>Troubleshoot</button>
      </div>
      <canvas ref={canvasRef} className={styles.canvas} />
      {walletData && (
        <div className="mt-4">
          <h2>Wallet: {walletData.network_id}</h2>
          <pre>{walletData.markdown}</pre>
          <h3>Agents: {agents.join(', ')}</h3>
        </div>
      )}
    </div>
  );
}
