import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.167.0/build/three.min.js';
import axios from 'axios';

export class NASAThreeJSVisualizer {
  constructor(containerId, token) {
    this.container = document.getElementById(containerId);
    this.token = token;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ alpha: true });
    this.isOffline = !navigator.onLine;
  }

  async init() {
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.container.appendChild(this.renderer.domElement);

    if (this.isOffline) {
      const cachedData = await caches.match('/nasa-data-cache');
      if (cachedData) {
        const data = await cachedData.json();
        this.renderCachedData(data);
      }
    } else {
      const { data } = await axios.get(`${process.env.VIAL_API_URL}/mcp/nasa_tools/datasets`, {
        headers: { Authorization: `Bearer ${this.token}` }
      });
      this.renderData(data);
      caches.open('nasa-data-cache').then(cache => cache.put('/nasa-data-cache', new Response(JSON.stringify(data))));
    }

    this.camera.position.z = 50;
    this.animate();
  }

  renderData(data) {
    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
    data.forEach(item => {
      const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(item.x || 0, item.y || 0, item.z || 0);
      this.scene.add(sphere);
    });
  }

  renderCachedData(data) {
    this.renderData(data);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.renderer.render(this.scene, this.camera);
  }
}
