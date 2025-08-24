```javascript
import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';
import MCPClient from './mcp-client.js';

export default function TelescopeVisualizer({ containerId, canvasId, onError }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 400, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById(canvasId) });
  renderer.setSize(window.innerWidth, 400);
  camera.position.z = 5;

  const geometry = new THREE.PlaneGeometry(4, 2);
  const material = new THREE.MeshBasicMaterial({ color: 0x000000 });
  const plane = new THREE.Mesh(geometry, material);
  scene.add(plane);

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }
  animate();

  async function visualizeTelescope(config, walletId) {
    try {
      const [route, time] = config.includes(',') ? config.split(',') : ['moon-mars', '2023-01-01'];
      const response = await MCPClient.fetchDropshipData({ route, time, wallet_id: walletId });
      if (response.obs && response.obs.url) {
        document.getElementById(containerId).innerHTML = `<iframe src="${response.obs.url}" width="100%" height="200px"></iframe>`;
      } else if (response.gibs && response.gibs.url) {
        const texture = new THREE.TextureLoader().load(response.gibs.url);
        plane.material = new THREE.MeshBasicMaterial({ map: texture });
      }
      return response;
    } catch (error) {
      onError(`Telescope Visualization Error: ${error.message}`);
      return {};
    }
  }

  return { visualizeTelescope };
}
```
