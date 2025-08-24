```javascript
import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';
import MCPClient from './mcp-client.js';

export default function AstronomyVisualizer({ containerId, canvasId, onError }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 400, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById(canvasId) });
  renderer.setSize(window.innerWidth, 400);
  camera.position.z = 5;

  const geometry = new THREE.SphereGeometry(1, 32, 32);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
  const sphere = new THREE.Mesh(geometry, material);
  scene.add(sphere);

  function animate() {
    requestAnimationFrame(animate);
    sphere.rotation.y += 0.01;
    renderer.render(scene, camera);
  }
  animate();

  async function visualizeAstronomy(query, walletId) {
    try {
      const container = document.getElementById(containerId);
      container.innerHTML = '';
      const [layer, time] = query.includes(',') ? query.split(',') : [query, '2023-01-01'];
      const response = await MCPClient.fetchAstronomyData({ layer, time, wallet_id: walletId });
      container.innerText = JSON.stringify(response, null, 2);
      if (response.gibs && response.gibs.url) {
        const texture = new THREE.TextureLoader().load(response.gibs.url);
        sphere.material = new THREE.MeshBasicMaterial({ map: texture });
      } else if (response.apod && response.apod.url) {
        const texture = new THREE.TextureLoader().load(response.apod.url);
        sphere.material = new THREE.MeshBasicMaterial({ map: texture });
      }
      return response;
    } catch (error) {
      onError(`Astronomy Visualization Error: ${error.message}`);
      return {};
    }
  }

  return { visualizeAstronomy };
}
```
