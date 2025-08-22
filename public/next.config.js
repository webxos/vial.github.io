/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
    MONGO_URI: process.env.MONGO_URI || 'mongodb://mongodb:27017/vial_mcp',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_BASE_URL}/alchemist/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
