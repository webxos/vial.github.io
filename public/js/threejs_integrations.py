import * as THREE from 'three';

export function setupScene(canvas) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  camera.position.z = 5;
  return { scene, camera, renderer };
}

export function create3DComponent(scene, config) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const cube = new THREE.Mesh(geometry, material);
  cube.position.set(config.position.x, config.position.y, config.position.z);
  scene.add(cube);
  return cube;
}

export function createConnection(scene, startPos, endPos) {
  const material = new THREE.LineBasicMaterial({ color: 0xffffff });
  const points = [
    new THREE.Vector3(startPos.x, startPos.y, startPos.z),
    new THREE.Vector3(endPos.x, endPos.y, endPos.z)
  ];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geometry, material);
  scene.add(line);
}
