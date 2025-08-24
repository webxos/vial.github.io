import React, { useEffect, useRef, useState } from 'react';
import { fetch } from 'whatwg-fetch';

interface Component {
  id: string;
  type: string;
  x: number;
  y: number;
  properties: any;
  connections: string[];
}

const CircuitDesigner: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [components, setComponents] = useState<Component[]>([]);

  useEffect(() => {
    if (svgRef.current) {
      const svg = svgRef.current;
      svg.innerHTML = `
        <defs>
          <style>
            .grid { fill: url(#grid); }
            .neon-green { fill: #00ff41; stroke: #00ff41; }
            .neon-blue { fill: #00d4ff; stroke: #00d4ff; }
            .circuit-line { stroke: #00ff41; stroke-width: 2; stroke-dasharray: 5,5; animation: flow 3s linear infinite; }
            @keyframes flow { 0% { stroke-dashoffset: 20; } 100% { stroke-dashoffset: 0; } }
          </style>
          <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" stroke="#333333" stroke-width="0.5"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" class="grid"/>
      `;
    }
  }, []);

  const handleDrop = async (event: React.DragEvent<SVGSVGElement>) => {
    event.preventDefault();
    const type = event.dataTransfer.getData('componentType');
    const rect = svgRef.current!.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (1920 / rect.width);
    const y = (event.clientY - rect.top) * (1080 / rect.height);

    const response = await fetch('http://localhost:8000/mcp/task/orchestrate', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer YOUR_TOKEN',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        tasks: [{ type: 'circuit_design', data: { components: [{ type, x, y, properties: {} }], connections: [] } }],
        wallet_id: 'wallet123'
      })
    });
    const result = await response.json();

    setComponents([...components, { id: Math.random().toString(), type, x, y, properties: {}, connections: [] }]);
  };

  return (
    <div style={{ width: '100%', height: '600px' }}>
      <svg ref={svgRef} width="100%" height="600" xmlns="http://www.w3.org/2000/svg" onDragOver={e => e.preventDefault()} onDrop={handleDrop}>
        {components.map(c => (
          <g key={c.id} transform={`translate(${c.x}, ${c.y})`}>
            <rect width="60" height="20" x="-30" y="-10" className="neon-green" rx="5"/>
            <text x="0" y="5" fill="#00d4ff" fontFamily="monospace" textAnchor="middle">{c.type}</text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default CircuitDesigner;
