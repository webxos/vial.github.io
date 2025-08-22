import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import Head from 'next/head';
import styles from '../styles/Home.module.css';

export default function Home() {
  const canvasRef = useRef(null);

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
          <button>Save Configuration</button>
          <button>Export SVG</button>
        </div>
        <div className={styles.canvas} ref={canvasRef}></div>
      </div>
    </div>
  );
}
