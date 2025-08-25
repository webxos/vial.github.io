import React, { useEffect, useRef } from 'react';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.167.0/build/three.min.js';
import axios from 'axios';

const DropshipViz = ({ token }) => {
  const mountRef = useRef(null);
  const [error, setError] = React.useState('');

  useEffect(() => {
    if (!token) return;

    axios.get(`${process.env.VIAL_API_URL}/mcp/dropship/simulate?origin=Earth&destination=Mars&cargo=1000`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(({ data }) => {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        mountRef.current.appendChild(renderer.domElement);

        // Add planets and dropship
        const earth = new THREE.Mesh(
          new THREE.SphereGeometry(1, 32, 32),
          new THREE.MeshBasicMaterial({ color: 0x0000ff })
        );
        earth.position.set(-10, 0, 0);
        scene.add(earth);

        const mars = new THREE.Mesh(
          new THREE.SphereGeometry(0.7, 32, 32),
          new THREE.MeshBasicMaterial({ color: 0xff0000 })
        );
        mars.position.set(10, 0, 0);
        scene.add(mars);

        const dropship = new THREE.Mesh(
          new THREE.BoxGeometry(0.5, 0.5, 0.5),
          new THREE.MeshBasicMaterial({ color: 0x00ff00 })
        );
        dropship.position.set(0, 0, 0);
        scene.add(dropship);

        camera.position.z = 20;
        const animate = () => {
          requestAnimationFrame(animate);
          dropship.position.x += (mars.position.x - earth.position.x) / 1000;
          renderer.render(scene, camera);
        };
        animate();

        return () => {
          mountRef.current.removeChild(renderer.domElement);
        };
      })
      .catch(err => setError('Failed to fetch dropship data: ' + err.message));
  }, [token]);

  return (
    <div className="text-green-500">
      <h2 className="text-xl font-bold mb-4">Dropship Mission Visualization</h2>
      {error && <p className="text-red-500">{error}</p>}
      <div ref={mountRef} style={{ width: '100%', height: '400px' }}></div>
    </div>
  );
};

export default DropshipViz;
