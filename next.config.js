const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true
});

module.exports = withPWA({
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/:path*'
      }
    ];
  },
  env: {
    NEXTAUTH_URL: process.env.VERCEL_URL || 'http://localhost:3000'
  }
});
