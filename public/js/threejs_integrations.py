// public/js/threejs_integrations.js
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js';

// Initialize Three.js scene
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth - 250, window.innerHeight - 50);
document.getElementById('canvas').appendChild(renderer.domElement);

// Orbit controls for 3D interaction
const controls = new THREE.OrbitControls(camera, renderer.domElement);
camera.position.z = 5;

// Component storage
const components = [];
const connections = [];

// Component types and colors
const componentTypes = {
    'API Endpoint': 0x00ff00,
    'LLM Model': 0xff0000,
    'Database': 0x0000ff,
    'Tool': 0xffff00,
    'Agent': 0xff00ff,
    'Wallet': 0x00ffff // Added for WebXOS wallet visualization
};

// Add component to scene
function addComponent(type, position) {
    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const material = new THREE.MeshBasicMaterial({ color: componentTypes[type] });
    const cube = new THREE.Mesh(geometry, material);
    cube.position.set(position.x, position.y, position.z);
    cube.userData = { type, id: `comp_${components.length}` };
    scene.add(cube);
    components.push(cube);
}

// Add connection line between components
function addConnection(fromId, toId) {
    const fromComp = components.find(c => c.userData.id === fromId);
    const toComp = components.find(c => c.userData.id === toId);
    if (fromComp && toComp) {
        const material = new THREE.LineBasicMaterial({ color: 0xffffff });
        const geometry = new THREE.BufferGeometry().setFromPoints([
            fromComp.position,
            toComp.position
        ]);
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        connections.push(line);
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

// Drag-and-drop handling
document.querySelectorAll('.component').forEach(component => {
    component.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/plain', component.textContent);
    });
});

document.getElementById('canvas').addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.getElementById('canvas').addEventListener('drop', (e) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('text/plain');
    const rect = renderer.domElement.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    addComponent(type, { x, y, z: 0 });
});

// Save configuration to backend
document.getElementById('save').addEventListener('click', async () => {
    const config = {
        components: components.map(c => ({
            id: c.userData.id,
            type: c.userData.type,
            position: c.position
        })),
        connections: connections.map((_, i) => ({
            from: components[i % components.length].userData.id,
            to: components[(i + 1) % components.length].userData.id
        }))
    };
    await fetch('http://localhost:8000/save-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
});

// Export SVG diagram
document.getElementById('export-svg').addEventListener('click', async () => {
    const response = await fetch('http://localhost:8000/diagram/export');
    const svg = await response.text();
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'diagram.svg';
    a.click();
    URL.revokeObjectURL(url);
});

// WebXOS Wallet visualization
async function loadWalletData() {
    const response = await fetch('http://localhost:8000/wallet/balance');
    const walletData = await response.json();
    addComponent('Wallet', { x: 2, y: 2, z: 0 });
    console.log(`Wallet Balance: ${walletData.balance} $WEBXOS`);
}
loadWalletData();
