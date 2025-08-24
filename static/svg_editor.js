```javascript
class SVGEditor {
    constructor() {
        this.canvas = document.getElementById('svgDiagram');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.init();
    }

    init() {
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());
    }

    startDrawing(e) {
        this.isDrawing = true;
        [this.lastX, this.lastY] = [e.offsetX, e.offsetY];
    }

    draw(e) {
        if (!this.isDrawing) return;
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(e.offsetX, e.offsetY);
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        [this.lastX, this.lastY] = [e.offsetX, e.offsetY];
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    saveSVG() {
        const svgData = this.canvas.toDataURL('image/svg+xml');
        fetch('/api/svg/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({data: svgData})
        }).then(res => res.json()).then(data => console.log(data));
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SVGEditor();
    document.getElementById('transpileSvg').addEventListener('click', () => {
        new SVGEditor().saveSVG();
    });
});
```
