import { useEffect, useRef, useState } from 'react';
import { setupScene, create3DComponent } from '../public/js/threejs_integrations';
import styles from '../styles/Home.module.css';

export default function Home() {
  const canvasRef = useRef(null);
  const [walletData, setWalletData] = useState(null);

  useEffect(() => {
    if (canvasRef.current) {
      const { scene, camera, renderer } = setupScene(canvasRef.current);
      const fetchWallet = async () => {
        const response = await fetch('/v1/wallet/export', {
          method: 'POST',
          headers: {
            'Authorization': 'Bearer test_token',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ network_id: '54965687-3871-4f3d-a803-ac9840af87c4' })
        });
        const data = await response.json();
        setWalletData(data);
        if (data.markdown) {
          const vials = ['vial1', 'vial2', 'vial3', 'vial4'];
          vials.forEach((vial, i) => {
            create3DComponent(scene, {
              id: vial,
              type: 'vial',
              title: vial,
              position: { x: i * 2 - 3, y: 0, z: 0 }
            });
          });
        }
        renderer.render(scene, camera);
      };
      fetchWallet();
    }
  }, []);

  return (
    <div className={styles.container}>
      <h1>Vial MCP Controller</h1>
      {walletData && (
        <div>
          <h2>Wallet: {walletData.network_id}</h2>
          <pre>{walletData.markdown}</pre>
        </div>
      )}
      <canvas ref={canvasRef} style={{ width: '100%', height: '400px' }} />
    </div>
  );
}
