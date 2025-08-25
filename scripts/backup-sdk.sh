#!/bin/bash
set -e

echo "Starting WebXOS 2025 Vial MCP SDK backup..."

# Configuration
BACKUP_DIR="backup-$(date +%Y%m%d-%H%M%S)"
DB_FILE="webxos.db"
LOG_FILE="security.log"
REPO_DIR="webxos-vial-mcp"
TAR_FILE="$BACKUP_DIR.tar.gz"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Copy critical files
cp -r server "$BACKUP_DIR/server"
cp -r vialfolder "$BACKUP_DIR/vialfolder"
cp -r examples "$BACKUP_DIR/examples"
cp -r docs "$BACKUP_DIR/docs"
cp -r helm "$BACKUP_DIR/helm"
cp vial.html "$BACKUP_DIR/vial.html"
cp requirements*.txt "$BACKUP_DIR/"
cp package.json "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/.env"
cp .env.local "$BACKUP_DIR/.env.local"
[ -f "$DB_FILE" ] && cp "$DB_FILE" "$BACKUP_DIR/$DB_FILE"
[ -f "$LOG_FILE" ] && cp "$LOG_FILE" "$BACKUP_DIR/$LOG_FILE"

# Archive backup
tar -czf "$TAR_FILE" "$BACKUP_DIR"

# Validate backup
if [ -f "$TAR_FILE" ]; then
    echo "Backup created: $TAR_FILE"
    tar -tzf "$TAR_FILE" | grep "server/models/wallet_models.py" || { echo "Backup validation failed: missing wallet_models.py"; exit 1; }
    tar -tzf "$TAR_FILE" | grep "vialfolder/js/components/galaxycraft-viz.jsx" || { echo "Backup validation failed: missing galaxycraft-viz.jsx"; exit 1; }
else
    echo "Backup creation failed!"
    exit 1
fi

# Clean up
rm -rf "$BACKUP_DIR"

echo "WebXOS 2025 Vial MCP SDK backup completed successfully!"
