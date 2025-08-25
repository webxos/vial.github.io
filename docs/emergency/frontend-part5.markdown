# WebXOS 2025 Vial MCP SDK: Frontend Emergency Backup - Part 5 (Testing and Validation)

**Objective**: Implement frontend tests and validation scripts to ensure functionality.

**Instructions for LLM**:
1. Create `public/js/__tests__/app.test.js` for Jest tests.
2. Create `scripts/validate-frontend.sh` for validation.
3. Ensure tests cover OAuth2 login and API integration.
4. Integrate with `package.json` test script.

## Step 1: Create Test Files

### public/js/__tests__/app.test.js
```javascript
import { render, screen } from '@testing-library/react';
import Home from '../../../pages/index';
import { SessionProvider } from 'next-auth/react';

describe('Home Component', () => {
  it('renders login button when not authenticated', () => {
    render(
      <SessionProvider session={null}>
        <Home />
      </SessionProvider>
    );
    expect(screen.getByText('Login with Google')).toBeInTheDocument();
  });

  it('renders logout and fetch buttons when authenticated', () => {
    const session = { accessToken: 'mock-token' };
    render(
      <SessionProvider session={session}>
        <Home />
      </SessionProvider>
    );
    expect(screen.getByText('Logout')).toBeInTheDocument();
    expect(screen.getByText('Fetch SpaceX Launches')).toBeInTheDocument();
  });
});
```

### scripts/validate-frontend.sh
```bash
#!/bin/bash
set -e
echo "Starting frontend validation..."
npm run lint
npm run test
npm run build
echo "Starting frontend server..."
npm run start &
sleep 5
curl http://localhost:3000 -o /dev/null -s -w "%{http_code}\n" | grep 200 || { echo "Frontend failed to start"; exit 1; }
echo "Frontend validation successful!"
```

## Step 2: Update package.json
Add to `package.json` scripts:
```json
{
  "scripts": {
    "validate": "bash scripts/validate-frontend.sh"
  }
}
```

## Step 3: Validation
```bash
chmod +x scripts/validate-frontend.sh
npm run validate
```

**Completion**: Frontend rebuild complete.