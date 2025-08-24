```typescript
import React, { useEffect, useState } from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import { useSWRConfig } from 'swr';
import useSWR from 'swr';
import { fetcher } from '../utils/fetcher';

interface QuantumCircuitProps {
  circuit: string;
}

const QuantumCircuit: React.FC<QuantumCircuitProps> = ({ circuit }) => {
  const { mutate } = useSWRConfig();
  const [nodes, setNodes] = useState<any[]>([]);
  const [edges, setEdges] = useState<any[]>([]);

  const { data, error } = useSWR('/mcp/quantum_rag', fetcher, {
    onSuccess: (data) => {
      // Parse quantum circuit and generate nodes/edges (simplified)
      const parsedNodes = data.results.map((result: string, index: number) => ({
        id: `node-${index}`,
        type: 'default',
        data: { label: result },
        position: { x: index * 100, y: 50 }
      }));
      const parsedEdges = data.results.slice(1).map((_, index: number) => ({
        id: `edge-${index}`,
        source: `node-${index}`,
        target: `node-${index + 1}`
      }));
      setNodes(parsedNodes);
      setEdges(parsedEdges);
    }
  });

  useEffect(() => {
    if (circuit) {
      mutate('/mcp/quantum_rag', { query: 'visualize circuit', quantum_circuit: circuit, max_results: 5 });
    }
  }, [circuit, mutate]);

  if (error) return <div>Error loading circuit: {error.message}</div>;
  if (!data) return <div>Loading...</div>;

  return (
    <div style={{ height: '500px', width: '100%' }}>
      <ReactFlow nodes={nodes} edges={edges}>
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default QuantumCircuit;
```
