import * as THREE from 'https://unpkg.com/three@0.134.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.134.0/examples/jsm/controls/OrbitControls.js';

function initThreeJS() {
    const API_BASE_URL = window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://your-api-domain.com';
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('output').appendChild(renderer.domElement);

    // Responsive design
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Theme toggle
    let isDarkTheme = true;
    document.getElementById('theme-toggle').addEventListener('click', () => {
        isDarkTheme = !isDarkTheme;
        document.body.style.backgroundColor = isDarkTheme ? '#1a1a1a' : '#ffffff';
        scene.traverse(obj => {
            if (obj.isMesh) obj.material.color.set(isDarkTheme ? 0x00ff00 : 0xff0000);
        });
    });

    // Orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Component storage
    const components = [];
    const connections = [];

    // Component creation
    function addComponent(type, position) {
        const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
        const material = new THREE.MeshBasicMaterial({ color: isDarkTheme ? 0x00ff00 : 0xff0000 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(position);
        mesh.userData = { id: `comp_${components.length + 1}`, type, config: {} };
        scene.add(mesh);
        components.push(mesh);
        return mesh;
    }

    // Connection line rendering
    function addConnection(fromId, toId) {
        const fromComp = components.find(c => c.userData.id === fromId);
        const toComp = components.find(c => c.userData.id === toId);
        if (!fromComp || !toComp) return;
        const material = new THREE.LineBasicMaterial({ color: 0xffffff });
        const points = [
            fromComp.position.clone(),
            toComp.position.clone()
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        connections.push({ fromId, toId, line });
    }

    // Drag-and-drop
    let selected = null;
    renderer.domElement.addEventListener('mousedown', (event) => {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(components);
        if (intersects.length > 0) {
            selected = intersects[0].object;
            selected.material.color.set(0xffff00);
        }
    });

    renderer.domElement.addEventListener('mousemove', (event) => {
        if (selected) {
            const mouse = new THREE.Vector2();
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0));
            if (intersects.length > 0) {
                selected.position.copy(intersects[0].point);
                connections.forEach(conn => {
                    if (conn.fromId === selected.userData.id || conn.toId === selected.userData.id) {
                        scene.remove(conn.line);
                        addConnection(conn.fromId, conn.toId);
                    }
                });
            }
        }
    });

    renderer.domElement.addEventListener('mouseup', () => {
        if (selected) {
            selected.material.color.set(isDarkTheme ? 0x00ff00 : 0xff0000);
            selected = null;
        }
    });

    // SVG export
    async function exportSVG() {
        const svg = `<svg width="${window.innerWidth}" height="${window.innerHeight}" xmlns="http://www.w3.org/2000/svg">
            ${components.map(c => `<rect x="${c.position.x * 50 + window.innerWidth / 2}" y="${c.position.y * 50 + window.innerHeight / 2}" width="20" height="20" fill="${isDarkTheme ? 'green' : 'red'}" />`).join('')}
            ${connections.map(c => {
                const from = components.find(comp => comp.userData.id === c.fromId);
                const to = components.find(comp => comp.userData.id === c.toId);
                return `<line x1="${from.position.x * 50 + window.innerWidth / 2 + 10}" y1="${from.position.y * 50 + window.innerHeight / 2 + 10}" x2="${to.position.x * 50 + window.innerWidth / 2 + 10}" y2="${to.position.y * 50 + window.innerHeight / 2 + 10}" stroke="white" />`;
            }).join('')}
        </svg>`;
        const blob = new Blob([svg], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'diagram.svg';
        link.click();
        URL.revokeObjectURL(url);
    }

    document.getElementById('export-svg').addEventListener('click', exportSVG);

    // Save configuration
    document.getElementById('save').addEventListener('click', async () => {
        const config = {
            components: components.map(c => ({
                id: c.userData.id,
                type: c.userData.type,
                position: c.position.toArray()
            })),
            connections: connections.map(c => ({ fromId: c.fromId, toId: c.toId }))
        };
        await fetch(`${API_BASE_URL}/save-config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
    });

    // Component drop handling
    document.getElementById('output').addEventListener('dragover', (e) => e.preventDefault());
    document.getElementById('output').addEventListener('drop', (e) => {
        e.preventDefault();
        const type = e.dataTransfer.getData('text/plain');
        const mouse = new THREE.Vector2();
        mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0));
        if (intersects.length > 0) {
            addComponent(type, intersects[0].point);
        }
    });

    camera.position.z = 5;
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}
initThreeJS();
