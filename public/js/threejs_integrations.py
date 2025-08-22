import * as THREE from 'https://unpkg.com/three@0.134.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.134.0/examples/jsm/controls/OrbitControls.js';

function initThreeJS() {
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

    // Orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Component mesh rendering
    const components = [];
    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    cube.userData = { id: 'comp1', type: 'api', config: {} };
    scene.add(cube);
    components.push(cube);

    // Theme toggle
    let isDarkTheme = true;
    document.getElementById('theme-toggle').addEventListener('click', () => {
        isDarkTheme = !isDarkTheme;
        document.body.style.backgroundColor = isDarkTheme ? '#1a1a1a' : '#ffffff';
        material.color.set(isDarkTheme ? 0x00ff00 : 0xff0000);
    });

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
            selected.material.color.set(0xffff00); // Highlight
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
            }
        }
    });

    renderer.domElement.addEventListener('mouseup', () => {
        if (selected) {
            selected.material.color.set(isDarkTheme ? 0x00ff00 : 0xff0000);
            selected = null;
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
