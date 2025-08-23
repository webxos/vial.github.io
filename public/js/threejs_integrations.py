import * as THREE from 'https://unpkg.com/three@0.153.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.153.0/examples/jsm/controls/OrbitControls.js';

export function create3DComponent(scene, componentData) {
    const { id, type, title, position, svg } = componentData;
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(position.x || 0, position.y || 0, position.z || 0);
    mesh.userData = { id, type, title, svg };
    scene.add(mesh);
    return mesh;
}

export function updateComponentPosition(mesh, x, y, z) {
    mesh.position.set(x, y, z);
}

export function createConnection(scene, source, target, type) {
    const material = new THREE.LineBasicMaterial({ color: 0x00ff00 });
    const points = [
        new THREE.Vector3(source.position.x, source.position.y, source.position.z),
        new THREE.Vector3(target.position.x, target.position.y, target.position.z)
    ];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(geometry, material);
    line.userData = { type };
    scene.add(line);
    return line;
}

export function setupScene(canvas) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 600, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas });
    renderer.setSize(window.innerWidth, 600);
    const controls = new OrbitControls(camera, renderer.domElement);
    camera.position.z = 5;

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    animate();
    return {
        scene,
        camera,
        renderer,
        dispose: () => renderer.dispose()
    };
}
