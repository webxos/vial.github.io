```javascript
class SettingsMenu {
    constructor() {
        this.menu = document.createElement('div');
        this.menu.id = 'settingsMenu';
        this.menu.className = 'bg-gray-800 p-4 rounded hidden';
        document.body.appendChild(this.menu);
    }

    render() {
        const modes = ['SVG', 'LAUNCH', 'SWARM', 'GALAXYCRAFT'];
        this.menu.innerHTML = `
            <h3 class="text-lg">Settings</h3>
            ${modes.map(mode => `
                <div class="p-2">
                    <label>${mode}</label>
                    <input type="checkbox" data-mode="${mode}" ${localStorage.getItem(`mode_${mode}`) === 'true' ? 'checked' : ''}>
                </div>
            `).join('')}
            <button id="saveSettings" class="bg-blue-500 p-2 rounded">Save</button>
        `;
        document.getElementById('modeSelect').appendChild(this.button);
        this.button.addEventListener('click', () => this.toggle());
        document.getElementById('saveSettings').addEventListener('click', () => this.save());
    }

    toggle() {
        this.menu.classList.toggle('hidden');
    }

    save() {
        const checkboxes = this.menu.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => localStorage.setItem(`mode_${cb.dataset.mode}`, cb.checked));
        this.menu.classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const settings = new SettingsMenu();
    settings.render();
    const button = document.createElement('button');
    button.id = 'openSettings';
    button.className = 'bg-yellow-500 p-2 rounded';
    button.textContent = 'Settings';
    settings.button = button;
    document.getElementById('modeSelect').appendChild(button);
});
```settin
