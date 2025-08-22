// public/js/threejs_integrations.js
import * as THREE from 'three';

export class WalletVisualizer {
    constructor(containerId) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById(containerId).appendChild(this.renderer.domElement);
        this.camera.position.z = 5;
    }

    async loadWalletData(walletAddress) {
        try {
            const response = await fetch(`/wallet/export/${walletAddress}`, {
                headers: { 'Authorization': `Bearer ${sessionStorage.getItem('access_token')}` }
            });
            const data = await response.json();
            
            const geometry = new THREE.SphereGeometry(data.reputation / 10, 32, 32);
            const material = new THREE.MeshBasicMaterial({ color: 0x0000ff });
            const sphere = new THREE.Mesh(geometry, material);
            this.scene.add(sphere);
            
            this.animate();
            return { status: 'success', reputation: data.reputation };
        } catch (error) {
            console.error('Error loading wallet data:', error);
            throw error;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.scene.children.forEach(child => {
            if (child instanceof THREE.Mesh) {
                child.rotation.y += 0.01;
            }
        });
        this.renderer.render(this.scene, this.camera);
    }
}
