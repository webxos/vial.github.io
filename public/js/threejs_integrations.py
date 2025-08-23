import * as THREE from 'https://unpkg.com/three@0.153.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.153.0/examples/jsm/controls/OrbitControls.js';

export function setupScene(canvas) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  const controls = new OrbitControls(camera, renderer.domElement);
  camera.position.z = 5;

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
  return { scene, camera, renderer };
}

export function create3DComponent(scene, config) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(config.position.x, config.position.y, config.position.z);
  mesh.userData = { id: config.id, type: config.type, title: config.title };
  scene.add(mesh);
  return mesh;
}

export function createConnection(scene, start, end) {
  const material = new THREE.LineBasicMaterial({ color: 0x00ff00 });
  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(start.x, start.y, start.z),
    new THREE.Vector3(end.x, end.y, end.z)
  ]);
  const line = new THREE.Line(geometry, material);
  scene.add(line);
  return line;
}
