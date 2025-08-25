import * as THREE from 'three';
import axios from 'axios';

export class GalaxyCraftVisualizer {
  constructor(containerId, token) {
    this.container = document.getElementById(containerId);
    this.token = token;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer();
    this.stars = [];
  }

  async init() {
    try {
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.container.appendChild(this.renderer.domElement);

      // Fetch galaxy data from API
      const { data } = await axios.get(`${process.env.VIAL_API_URL}/mcp/galaxycraft/stars`, {
        headers: { Authorization: `Bearer ${this.token}` }
      });

      // Add stars
      const starGeometry = new THREE.SphereGeometry(0.1, 16, 16);
      const starMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
      data.stars.forEach(star => {
        const starMesh = new THREE.Mesh(starGeometry, starMaterial);
        starMesh.position.set(star.x, star.y, star.z);
        this.scene.add(starMesh);
        this.stars.push(starMesh);
      });

      this.camera.position.z = 50;
      this.animate();
    } catch (error) {
      console.error('Failed to initialize GalaxyCraft visualizer:', error);
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.stars.forEach(star => {
      star.rotation.y += 0.01;
    });
    this.renderer.render(this.scene, this.camera);
  }
}
