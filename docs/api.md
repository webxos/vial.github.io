Vial MCP API Documentation
Authentication
POST /v1/auth/token
Generate JWT using OAuth 2.0+PKCE.
Request:
{
  "access_token": "google_oauth_token"
}

Response:
{
  "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

Wallet Operations
POST /v1/wallet/export
Export encrypted wallet as .md file.
Request:
{
  "address": "0x123",
  "amount": 100.0
}

Response:
{
  "wallet_md": "# Wallet\nCiphertext: abc123..."
}

Security Requirements

All endpoints require JWT authentication
Microsoft Prompt Shields scans all inputs
Rate limits: 10/min (wallet), 5/min (quantum)
