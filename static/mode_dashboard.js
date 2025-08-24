```javascript
class ModeDashboard {
    constructor() {
        this.dashboard = document.createElement('div');
        this.dashboard.id = 'modeDashboard';
        this.dashboard.className = 'bg-gray-800 p-4 rounded';
        document.body.appendChild(this.dashboard);
    }

    async update() {
        const modes = ['SVG', 'LAUNCH', 'SWARM', 'GALAXYCRAFT'];
        const promises = modes.map(mode =>
            fetch('/api/mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode})
            }).then(res => res.json())
        );
        const results = await Promise.all(promises);
        this.dashboard.innerHTML = modes.map((mode, i) => `
            <div class="p-2">${mode}: ${results[i].status} at ${results[i].timestamp}</div>
        `).join('');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new ModeDashboard();
    dashboard.update();
    setInterval(() => dashboard.update(), 5000);
});
```
