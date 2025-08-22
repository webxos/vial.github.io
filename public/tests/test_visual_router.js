import { render, screen, fireEvent } from '@testing-library/react';
import Home from '../../pages/index';
import '@testing-library/jest-dom';

describe('Visual API Router', () => {
  test('Component drag and drop functionality', () => {
    render(<Home />);
    const component = screen.getByText('API Endpoint');
    expect(component).toBeInTheDocument();
    fireEvent.dragStart(component);
    const canvas = screen.getByTestId('canvas');
    fireEvent.dragOver(canvas);
    fireEvent.drop(canvas);
    expect(console.log).toHaveBeenCalledWith('Dropped: API Endpoint');
  });

  test('3D scene rendering performance', () => {
    render(<Home />);
    const canvas = screen.getByTestId('canvas');
    expect(canvas).toBeInTheDocument();
    // Simulate rendering performance test
    jest.spyOn(window, 'requestAnimationFrame').mockImplementation(cb => cb());
    jest.advanceTimersByTime(1000);
    expect(canvas.children.length).toBeGreaterThan(0);
  });

  test('Configuration export/import', async () => {
    render(<Home />);
    const saveButton = screen.getByText('Save Configuration');
    fireEvent.click(saveButton);
    expect(console.log).toHaveBeenCalledWith('Configuration saved');
  });
});
