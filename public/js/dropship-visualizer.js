```javascript
import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';
import MCPClient from './mcp-client.js';

export default function DropshipVisualizer({ containerId, canvasId, onError }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 400, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById(canvasId) });
  renderer.setSize(window.innerWidth, 400);
  camera.position.z = 5;

  const geometry = new THREE.SphereGeometry(1, 32, 32);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
  const globe = new THREE.Mesh(geometry, material);
  scene.add(globe);

  function animate() {
    requestAnimationFrame(animate);
    globe.rotation.y += 0.01;
    renderer.render(scene, camera);
  }
  animate();

  async function visualizeDropship(config, walletId) {
    try {
      const [route, time] = config.includes(',') ? config.split(',') : ['moon-mars', '2023-01-01'];
      const response = await MCPClient.fetchDropshipData({ route, time, wallet_id: walletId });
      if (response.gibs && response.gibs.url) {
        const texture = new THREE.TextureLoader().load(response.gibs.url);
        globe.material = new THREE.MeshBasicMaterial({ map: texture });
      }
      if (response.obs && response.obs.url) {
        document.getElementById(containerId).innerHTML = `<iframe src="${response.obs.url}" width="100%" height="200px"></iframe>`;
      } else {
        document.getElementById(containerId).innerText = JSON.stringify(response, null, 2);
      }
      return response;
    } catch (error) {
      onError(`Dropship Visualization Error: ${error.message}`);
      return {};
    }
  }

  return { visualizeDropship };
}
```
