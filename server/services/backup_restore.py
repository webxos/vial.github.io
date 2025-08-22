import shutil


def setup_backup_restore(app):
    def backup_data():
        shutil.copytree("data", "backup/data")
    app.state.backup = backup_data
