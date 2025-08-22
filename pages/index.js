import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import Head from 'next/head';
import styles from '../styles/Home.module.css';

export default function Home() {
  const canvasRef = useRef(null);
  const [gitCommand, setGitCommand] = useState('');
  const [troubleshootResult, setTroubleshootResult] = useState(null);

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

    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onmessage = (event) => {
      console.log(`WebSocket message: ${event.data}`);
    };

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
          <button>Save Configuration</button>
          <button>Export SVG</button>
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
      </div>
    </div>
  );
}
