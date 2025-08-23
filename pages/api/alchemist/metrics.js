export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const response = await fetch('http://localhost:8000/alchemist/metrics', {
      headers: {
        Authorization: `Bearer ${req.headers.authorization}`
      }
    });
    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('Metrics fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch metrics' });
  }
}
