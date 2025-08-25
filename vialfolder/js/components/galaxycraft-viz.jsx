import React, { useEffect, useRef } from 'react';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.167.0/build/three.min.js';
import axios from 'axios';

const GalaxyCraftViz = ({ token }) => {
  const mountRef = useRef(null);
  const [error, setError] = React.useState('');

  useEffect(() => {
    if (!token) return;

    // Fetch galaxy data
    axios.get(`${process.env.VIAL_API_URL}/mcp/galaxycraft/generate?stars=100`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(({ data }) => {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        mountRef.current.appendChild(renderer.domElement);

        // Add stars
        data.stars.forEach(star => {
          const geometry = new THREE.SphereGeometry(star.size, 32, 32);
          const material = new THREE.MeshBasicMaterial({ color: star.color });
          const sphere = new THREE.Mesh(geometry, material);
          sphere.position.set(...star.position);
          scene.add(sphere);
        });

        camera.position.z = 100;
        const animate = () => {
          requestAnimationFrame(animate);
          scene.rotation.y += 0.001;
          renderer.render(scene, camera);
        };
        animate();

        // Cleanup
        return () => {
          mountRef.current.removeChild(renderer.domElement);
        };
      })
      .catch(err => setError('Failed to fetch galaxy data: ' + err.message));
  }, [token]);

  return (
    <div className="text-green-500">
      <h2 className="text-xl font-bold mb-4">GalaxyCraft Visualization</h2>
      {error && <p className="text-red-500">{error}</p>}
      <div ref={mountRef} style={{ width: '100%', height: '400px' }}></div>
    </div>
  );
};

export default GalaxyCraftViz;
