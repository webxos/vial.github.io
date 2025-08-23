import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGPURenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
camera.position.z = 5;

const nodes = [];
function addNode(id, type, position) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const node = new THREE.Mesh(geometry, material);
  node.position.set(...position);
  node.userData = { id, type };
  scene.add(node);
  nodes.push(node);
}

function exportSVG() {
  const svg = `<svg><rect x="0" y="0" width="100" height="100" fill="green"/></svg>`;
  const blob = new Blob([svg], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'network.svg';
  link.click();
}

const ws = new WebSocket('ws://localhost:8000/v1/mcp/ws?token=mock_token');
ws.onmessage = (event) => {
  const { channel, data } = JSON.parse(event.data);
  console.log(`Update on ${channel}: ${data}`);
};

addNode('vial1', 'agent', [0, 0, 0]);
renderer.render(scene, camera);
