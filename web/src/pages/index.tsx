import React from 'react';
import Dashboard from '../components/Dashboard';

const Home: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-green-400">
      <header className="p-4 text-center">
        <h1 className="text-3xl">Vial MCP - vial.github.io/vial/</h1>
        <p className="mt-2">Scalable MCP Server Template</p>
      </header>
      <main className="p-4">
        <Dashboard />
      </main>
      <footer className="p-4 text-center">
        <p>© 2025 Vial MCP | v0.1.0</p>
      </footer>
    </div>
  );
};

export default Home;
