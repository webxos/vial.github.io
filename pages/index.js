import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import Head from 'next/head';
import styles from '../styles/Home.module.css';

export default function Home() {
  const canvasRef = useRef(null);
  const [gitCommand, setGitCommand] = useState('');
  const [troubleshootResult, setTroubleshootResult] = useState(null);
  const [svgStyle, setSvgStyle] = useState('default');
  const [metrics, setMetrics] = useState(null);
  const [svgExport, setSvgExport] = useState(null);

  useEffect(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    canvasRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    camera.position.z = 5;

    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const ws = new WebSocket('ws://localhost:8000/alchemist/ws');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(`WebSocket message: ${JSON.stringify(data)}`);
      if (data.result && data.result.steps) {
        setTroubleshootResult(data.result);
      }
    };
    ws.onopen = () => {
      ws.send(JSON.stringify({ tool: "vial.status.get", params: { vial_id: "vial1" } }));
    };

    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/alchemist/metrics');
        const data = await response.json();
        setMetrics(data);
      } catch (error) {
        console.error('Metrics fetch error:', error);
      }
    };
    fetchMetrics();

    return () => {
      renderer.dispose();
      ws.close();
    };
  }, []);

  const handleTroubleshoot = async () => {
    try {
      const response = await fetch('/api/alchemist/troubleshoot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ error: 'User reported issue' })
      });
      const result = await response.json();
      setTroubleshootResult(result);
    } catch (error) {
      console.error('Troubleshoot error:', error);
    }
  };

  const handleGitCommand = async () => {
    try {
      const response = await fetch('/api/alchemist/git', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: gitCommand })
      });
      const result = await response.json();
      console.log('Git command result:', result);
      setGitCommand('');
    } catch (error) {
      console.error('Git command error:', error);
    }
  };

  const handleExportWallet = async () => {
    try {
      const response = await fetch('/api/alchemist/wallet/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'test', svg_style: svgStyle })
      });
      const result = await response.json();
      console.log('Wallet export result:', result);
      setSvgExport(result.resource_path);
    } catch (error) {
      console.error('Wallet export error:', error);
    }
  };

  const handleSvgDownload = () => {
    if (svgExport) {
      const link = document.createElement('a');
      link.href = `/api/file/${svgExport.split('/').pop()}`;
      link.download = svgExport.split('/').pop();
      link.click();
    }
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>Vial MCP Controller</title>
      </Head>
      <div className={styles.sidebar}>
        <h2>Components</h2>
        <div className={styles.component} draggable>API Endpoint</div>
        <div className={styles.component} draggable>LLM Model</div>
        <div className={styles.component} draggable>Database</div>
        <div className={styles.component} draggable>Tool</div>
        <div className={styles.component} draggable>Agent</div>
      </div>
      <div className={styles.main}>
        <div className={styles.toolbar}>
          <button onClick={handleTroubleshoot}>Troubleshoot</button>
          <button onClick={handleExportWallet}>Export Wallet</button>
          <button onClick={handleSvgDownload} disabled={!svgExport}>Download SVG</button>
          <select value={svgStyle} onChange={(e) => setSvgStyle(e.target.value)}>
            <option value="default">Default</option>
            <option value="alert">Alert</option>
            <option value="success">Success</option>
          </select>
        </div>
        <div className={styles.canvas} ref={canvasRef} data-testid="canvas"></div>
        <textarea
          className={styles.console}
          value={gitCommand}
          onChange={(e) => setGitCommand(e.target.value)}
          placeholder="Enter Git commands (e.g., git commit -m 'Update')"
        />
        {troubleshootResult && (
          <div className={styles.troubleshoot}>
            <h3>Troubleshooting Steps</h3>
            <p>{troubleshootResult.steps}</p>
            <ul>
              {troubleshootResult.options.map((option, index) => (
                <li key={index}>{option}</li>
              ))}
            </ul>
          </div>
        )}
        {metrics && (
          <div className={styles.metrics}>
            <h3>Metrics</h3>
            <p>Tool Calls: {JSON.stringify(metrics.tool_calls)}</p>
            <p>Average Duration: {JSON.stringify(metrics.avg_duration)}</p>
          </div>
        )}
      </div>
    </div>
  );
}
