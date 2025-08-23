import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.module.js';

export function setupScene(canvas) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  camera.position.z = 5;

  let isDragging = false;
  let selectedObject = null;
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  canvas.addEventListener('mousedown', (event) => {
    mouse.x = (event.clientX / canvas.clientWidth) * 2 - 1;
    mouse.y = -(event.clientY / canvas.clientHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children);
    if (intersects.length > 0) {
      isDragging = true;
      selectedObject = intersects[0].object;
    }
  });

  canvas.addEventListener('mousemove', (event) => {
    if (isDragging && selectedObject) {
      mouse.x = (event.clientX / canvas.clientWidth) * 2 - 1;
      mouse.y = -(event.clientY / canvas.clientHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
      const intersect = raycaster.ray.intersectPlane(plane, new THREE.Vector3());
      if (intersect) {
        selectedObject.position.set(intersect.x, intersect.y, 0);
      }
    }
  });

  canvas.addEventListener('mouseup', () => {
    if (isDragging && selectedObject) {
      isDragging = false;
      const task = {
        task_name: selectedObject.userData.type === 'endpoint' ? 'create_endpoint' : 'create_agent',
        params: {
          vial_id: selectedObject.userData.id,
          x_position: selectedObject.position.x,
          y_position: selectedObject.position.y,
          endpoint: selectedObject.userData.type === 'endpoint' ? `/v1/custom/${selectedObject.userData.id}` : null
        }
      };
      fetch('/v1/execute_svg_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test_token' },
        body: JSON.stringify(task)
      }).then(response => response.json()).then(data => {
        if (data.status === 'success') {
          alert(`${task.task_name === 'create_endpoint' ? 'Endpoint' : 'Agent'} ${selectedObject.userData.id} created`);
        }
      });
      selectedObject = null;
    }
  });

  return { scene, camera, renderer };
}

export function create3DComponent(scene, { id, type, title, position }) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: type === 'endpoint' ? 0xff0000 : 0x00ff00 });
  const cube = new THREE.Mesh(geometry, material);
  cube.position.set(position.x, position.y, position.z);
  cube.userData = { id, type, title };
  scene.add(cube);
  return cube;
}

export function createConnection(scene, startPos, endPos) {
  const material = new THREE.LineBasicMaterial({ color: 0x00ff00 });
  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(startPos.x, startPos.y, startPos.z),
    new THREE.Vector3(endPos.x, endPos.y, endPos.z)
  ]);
  const line = new THREE.Line(geometry, material);
  scene.add(line);
  return line;
}
