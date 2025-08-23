import '../styles/globals.css';
import { useEffect, useState } from 'react';

export default function App({ Component, pageProps }) {
  const [svgEditor, setSvgEditor] = useState(null);

  useEffect(() => {
    const loadSvgEditor = async () => {
      const SVGEditor = require('next-svg-editor');
      setSvgEditor(() => SVGEditor);
    };
    loadSvgEditor();
  }, []);

  return (
    <div className="app-container">
      {svgEditor && <svgEditor.Component />}
      <Component {...pageProps} />
    </div>
  );
}
