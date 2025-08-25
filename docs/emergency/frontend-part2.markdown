# WebXOS 2025 Vial MCP SDK: Frontend Emergency Backup - Part 2 (Core Application)

**Objective**: Implement the core frontend application with React, Next.js, and Tailwind CSS, integrating with backend APIs via OAuth2.

**Instructions for LLM**:
1. Create core frontend files (`pages/index.js`, `pages/_app.js`, `pages/api/auth/[...nextauth].js`).
2. Implement OAuth2 login with Google and API integration (SpaceX, NASA, etc.).
3. Use Tailwind CSS for styling.
4. Ensure compatibility with `package.json`.

## Step 1: Create Core Frontend Files

### pages/_app.js
```javascript
import '../public/css/globals.css';
import { SessionProvider } from 'next-auth/react';

export default function App({ Component, pageProps: { session, ...pageProps } }) {
  return (
    <SessionProvider session={session}>
      <Component {...pageProps} />
    </SessionProvider>
  );
}
```

### pages/index.js
```javascript
import { useState } from 'react';
import { useSession, signIn, signOut } from 'next-auth/react';
import axios from 'axios';

export default function Home() {
  const { data: session } = useSession();
  const [launches, setLaunches] = useState([]);

  const fetchLaunches = async () => {
    try {
      const response = await axios.get(`${process.env.API_URL}/mcp/spacex/launches?limit=5`, {
        headers: { Authorization: `Bearer ${session.accessToken}` }
      });
      setLaunches(response.data);
    } catch (error) {
      console.error('Error fetching launches:', error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">WebXOS 2025 Vial MCP SDK</h1>
      {!session ? (
        <button className="bg-blue-500 text-white p-2 rounded" onClick={() => signIn('google')}>
          Login with Google
        </button>
      ) : (
        <div>
          <button className="bg-red-500 text-white p-2 rounded mr-2" onClick={() => signOut()}>
            Logout
          </button>
          <button className="bg-green-500 text-white p-2 rounded" onClick={fetchLaunches}>
            Fetch SpaceX Launches
          </button>
          <ul className="mt-4">
            {launches.map((launch, idx) => (
              <li key={idx} className="p-2 border-b">{launch.name} - {launch.date_utc}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

### pages/api/auth/[...nextauth].js
```javascript
import NextAuth from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';

export default NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.OAUTH_CLIENT_ID,
      clientSecret: process.env.OAUTH_CLIENT_SECRET,
      authorization: {
        params: {
          scope: 'openid email profile',
          response_type: 'code',
          redirect_uri: process.env.NEXT_PUBLIC_OAUTH_REDIRECT_URI,
          code_challenge_method: 'S256'
        }
      }
    })
  ],
  callbacks: {
    async jwt({ token, account }) {
      if (account) {
        token.accessToken = account.access_token;
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      return session;
    }
  },
  secret: process.env.NEXTAUTH_SECRET
});
```

### public/css/globals.css
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Step 2: Validation
```bash
npm run dev
open http://localhost:3000
# Login with Google and fetch SpaceX launches
```

**Next**: Proceed to `frontend-part3.md` for Docker configuration.