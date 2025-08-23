from typing import Dict
import uuid
import sqlite3
from pymongo import MongoClient
from web3 import Web3
from dotenv import load_dotenv
import os
import logging
from server.logging import logger

load_dotenv()

class WebXOSWallet:
    def __init__(self):
        self.conn = sqlite3.connect("wallet.db")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS balances "
            "(network_id TEXT PRIMARY KEY, balance REAL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS transactions "
            "(id TEXT, network_id TEXT, amount REAL, timestamp TEXT)"
        )
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]
        self.w3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER")))

    def update_balance(self, network_id: str, amount: float) -> float:
        """Update wallet balance and log transaction."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO balances (network_id, balance) "
                "VALUES (?, COALESCE((SELECT balance FROM balances WHERE network_id = ?), 0) + ?)",
                (network_id, network_id, amount)
            )
            tx_id = str(uuid.uuid4())
            timestamp = "2025-08-23T01:00:00Z"
            cursor.execute(
                "INSERT INTO transactions (id, network_id, amount, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (tx_id, network_id, amount, timestamp)
            )
            self.conn.commit()
            self.db.balances.update_one(
                {"network_id": network_id},
                {"$set": {"balance": self.get_balance(network_id), "timestamp": timestamp}},
                upsert=True
            )
            self.db.transactions.insert_one({
                "id": tx_id,
                "network_id": network_id,
                "amount": amount,
                "timestamp": timestamp
            })
            if self.w3.is_connected():
                logger.info(f"Web3 connected, tx for {amount} $WEBXOS", request_id=tx_id)
            balance = cursor.execute(
                "SELECT balance FROM balances WHERE network_id = ?",
                (network_id,)
            ).fetchone()[0]
            logger.info(f"Updated balance for {network_id}: {balance}", request_id=tx_id)
            return balance
        except Exception as e:
            logger.error(f"Wallet update error: {str(e)}", request_id=str(uuid.uuid4()))
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-23T01:00:00Z]** Wallet update error: {str(e)}\n")
            raise

    def get_balance(self, network_id: str) -> float:
        """Get balance for a network ID."""
        cursor = self.conn.cursor()
        result = cursor.execute(
            "SELECT balance FROM balances WHERE network_id = ?",
            (network_id,)
        ).fetchone()
        return result[0] if result else 0.0

    def export_wallet(self, network_id: str) -> str:
        """Export wallet data as markdown matching Vial format."""
        try:
            balance = self.get_balance(network_id)
            cursor = self.conn.cursor()
            transactions = cursor.execute(
                "SELECT id, amount, timestamp FROM transactions WHERE network_id = ?",
                (network_id,)
            ).fetchall()
            markdown = (
                f"# Vial Wallet Export\n"
                f"**Timestamp**: 2025-08-23T01:00:00Z\n"
                f"**User ID**: {network_id}\n"
                f"**Wallet Address**: {network_id}\n"
                f"**Balance**: {balance} $WEBXOS\n"
                f"**Vials**: vial1, vial2, vial3, vial4\n"
                f"\n## Transactions\n"
            )
            for tx_id, amount, timestamp in transactions:
                markdown += f"- {timestamp}: {amount} $WEBXOS (ID: {tx_id})\n"
            self.db.exports.insert_one({
                "network_id": network_id,
                "markdown": markdown,
                "timestamp": "2025-08-23T01:00:00Z"
            })
            logger.info(f"Exported wallet for {network_id}", request_id=str(uuid.uuid4()))
            return markdown
        except Exception as e:
            logger.error(f"Wallet export error: {str(e)}", request_id=str(uuid.uuid4()))
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-23T01:00:00Z]** Wallet export error: {str(e)}\n")
            raise

    def import_wallet(self, markdown: str) -> Dict:
        """Import wallet data from markdown."""
        try:
            lines = markdown.split("\n")
            network_id = next(line.split(": ")[1] for line in lines if line.startswith("**User ID**"))
            balance = float(next(line.split(": ")[1].split(" ")[0] for line in lines if line.startswith("**Balance**")))
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO balances (network_id, balance) VALUES (?, ?)",
                (network_id, balance)
            )
            self.conn.commit()
            self.db.balances.update_one(
                {"network_id": network_id},
                {"$set": {"balance": balance, "timestamp": "2025-08-23T01:00:00Z"}},
                upsert=True
            )
            logger.info(f"Imported wallet for {network_id}", request_id=str(uuid.uuid4()))
            return {"network_id": network_id, "balance": balance}
        except Exception as e:
            logger.error(f"Wallet import error: {str(e)}", request_id=str(uuid.uuid4()))
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-23T01:00:00Z]** Wallet import error: {str(e)}\n")
            raise
