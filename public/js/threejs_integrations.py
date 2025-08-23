import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export function setupThreeJsCanvas(canvasElement, components = []) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true, canvas: canvasElement });
  renderer.setSize(window.innerWidth, window.innerHeight);

  const controls = new OrbitControls(camera, renderer.domElement);
  camera.position.z = 5;

  components.forEach(comp => {
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial({ color: comp.color || 0x00ff00 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(comp.position?.x || 0, comp.position?.y || 0, comp.position?.z || 0);
    scene.add(mesh);
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return {
    addComponent: (comp) => {
      const geometry = new THREE.BoxGeometry();
      const material = new THREE.MeshBasicMaterial({ color: comp.color || 0x00ff00 });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(comp.position?.x || 0, comp.position?.y || 0, comp.position?.z || 0);
      scene.add(mesh);
    },
    dispose: () => renderer.dispose()
  };
}
