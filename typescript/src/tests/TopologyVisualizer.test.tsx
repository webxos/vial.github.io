```typescript
import { render, screen } from '@testing-library/react';
import TopologyVisualizer from '../components/TopologyVisualizer';
import { SWRConfig } from 'swr';

jest.mock('swr');

describe('TopologyVisualizer', () => {
  it('renders topology with health data', async () => {
    require('swr').mockReturnValue({
      data: {
        status: 'healthy',
        services: { llm_router: 'healthy', obs: 'healthy', servicenow: 'healthy' }
      },
      error: null
    });
    render(
      <SWRConfig value={{ provider: () => new Map() }}>
        <TopologyVisualizer />
      </SWRConfig>
    );
    expect(await screen.findByText(/Server: healthy/)).toBeInTheDocument();
    expect(await screen.findByText(/LLM: healthy/)).toBeInTheDocument();
  });

  it('displays error message on fetch failure', async () => {
    require('swr').mockReturnValue({ data: null, error: new Error('Fetch error') });
    render(
      <SWRConfig value={{ provider: () => new Map() }}>
        <TopologyVisualizer />
      </SWRConfig>
    );
    expect(await screen.findByText(/Error loading topology: Fetch error/)).toBeInTheDocument();
  });
});
```
