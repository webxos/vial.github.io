import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.167.0/build/three.min.js';
import axios from 'axios';

export class NASAOrbitVisualizer {
  constructor(containerId, token) {
    this.container = document.getElementById(containerId);
    this.token = token;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer();
  }

  async init() {
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.container.appendChild(this.renderer.domElement);

    const { data } = await axios.get(`${process.env.VIAL_API_URL}/mcp/nasa_satellite/data`, {
      headers: { Authorization: `Bearer ${this.token}` }
    });
    if (data.trajectory) {
      const geometry = new THREE.SphereGeometry(0.1, 32, 32);
      const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
      data.trajectory[0].forEach((pos, index) => {
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(pos / 1000, index * 0.1, 0);
        this.scene.add(sphere);
      });
    }

    this.camera.position.z = 50;
    this.animate();
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.renderer.render(this.scene, this.camera);
  }
}n
