```javascript
export default class SVGTranspiler {
  constructor({ containerId, onError }) {
    this.container = document.getElementById(containerId);
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("width", "100%");
    this.svg.setAttribute("height", "200");
    this.container.appendChild(this.svg);
  }

  transpileSVG(script, style = { fill: "neon-green", stroke: "black" }) {
    try {
      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(script, "image/svg+xml").documentElement;
      this.svg.innerHTML = svgDoc.innerHTML;
      this.applyWebXOSStyle(svgDoc, style);
      return this.svg.outerHTML;
    } catch (error) {
      onError(`SVG Transpile Error: ${error.message}`);
      return "";
    }
  }

  applyWebXOSStyle(svgElement, style) {
    svgElement.querySelectorAll("*").forEach(el => {
      el.setAttribute("fill", style.fill);
      el.setAttribute("stroke", style.stroke);
    });
  }

  exportSVG() {
    return this.svg.outerHTML;
  }
}
```
