```javascript
import MCPClient from './mcp-client.js';
import SVGVisualizer from './svg-visualizer.js';
import LaunchVisualizer from './launch-visualizer.js';
import SwarmVisualizer from './swarm-visualizer.js';
import GalaxyCraftVisualizer from './galaxycraft-visualizer.js';
import AstronomyVisualizer from './astronomy-visualizer.js';
import ReputationVisualizer from './reputation-visualizer.js';
import EightBimVisualizer from './8bim-visualizer.js';

export default function UIControls({ contentId, errorConsoleId, canvasId }) {
  const modes = {
    svg: SVGVisualizer({ containerId: contentId, onError: displayError }),
    launch: LaunchVisualizer({ containerId: contentId, onError: displayError }),
    swarm: SwarmVisualizer({ containerId: contentId, onError: displayError }),
    galaxycraft: GalaxyCraftVisualizer({ containerId: contentId, onError: displayError }),
    telescope: AstronomyVisualizer({ containerId: contentId, canvasId, onError: displayError })
  };
  let currentMode = 'svg';
  const reputation = ReputationVisualizer({ containerId: contentId, onError: displayError });
  const eightBim = EightBimVisualizer({ containerId: contentId, onError: displayError });

  function displayError(message) {
    document.getElementById(errorConsoleId).innerText += `${message}\n`;
  }

  async function switchMode(mode) {
    currentMode = mode;
    document.getElementById(contentId).innerHTML = mode === 'telescope' ? '<a href="/telescope.html">Go to Telescope Console</a>' : `<canvas id="${canvasId}"></canvas>`;
    const walletId = document.getElementById('wallet-id').value;
    if (walletId) {
      await reputation.visualizeReputation(walletId);
    }
  }

  async function executeMode(mode, walletId, inputData) {
    try {
      document.getElementById(errorConsoleId).innerText = '';
      if (mode === 'svg') {
        await modes.svg.visualizeSVG(inputData, walletId);
      } else if (mode === 'launch') {
        await modes.launch.visualizeLaunches(inputData || 10);
      } else if (mode === 'swarm') {
        await modes.swarm.visualizeSwarm(walletId);
      } else if (mode === 'galaxycraft') {
        await modes.galaxycraft.visualizeGalaxyCraft(inputData, walletId);
        await eightBim.visualize8bim(inputData, walletId);
      } else if (mode === 'telescope') {
        await modes.telescope.visualizeAstronomy({ layer: inputData.split(',')[0], time: inputData.split(',')[1] || '2023-01-01', walletId });
      }
      await reputation.visualizeReputation(walletId);
    } catch (error) {
      displayError(`Execution Error: ${error.message}`);
    }
  }

  return { switchMode, executeMode };
}
```
