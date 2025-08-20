import React, { useEffect, useState } from 'react';

const Dashboard: React.FC = () => {
  const [status, setStatus] = useState<string>('Loading...');

  useEffect(() => {
    fetch('http://localhost:8000/mcp/tools/status')
      .then((res) => res.json())
      .then((data) => setStatus(data.status))
      .catch((err) => setStatus(`Error: ${err.message}`));
  }, []);

  return (
    <div className="p-4 bg-gray-800 text-white">
      <h2 className="text-xl">Dashboard</h2>
      <p>Server Status: {status}</p>
    </div>
  );
};

export default Dashboard;
