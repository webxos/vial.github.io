import React, { useState, useEffect } from 'react';
import axios from 'axios';

const WalletViz = ({ address, token }) => {
  const [transactions, setTransactions] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!address || !token) return;
    axios.get(`${process.env.VIAL_API_URL}/mcp/economic/forecast/${address}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(({ data }) => setTransactions(data.forecast))
      .catch(err => setError('Failed to fetch transactions: ' + err.message));
  }, [address, token]);

  return (
    <div className="mt-4">
      <h2 className="text-xl font-bold text-green-500">Wallet Transactions</h2>
      {error && <p className="text-red-500">{error}</p>}
      <canvas id="wallet-chart" width="400" height="200"></canvas>
      <script>
        const ctx = document.getElementById('wallet-chart').getContext('2d');
        new Chart(ctx, {
          type: 'line',
          data: {
            labels: {transactions.map(tx => tx.date)},
            datasets: [{
              label: 'Balance Forecast',
              data: {transactions.map(tx => tx.balance)},
              borderColor: '#00ff00',
              backgroundColor: 'rgba(0, 255, 0, 0.2)',
              fill: true
            }]
          },
          options: {
            scales: {
              y: { beginAtZero: true }
            }
          }
        });
      </script>
    </div>
  );
};

export default WalletViz;
