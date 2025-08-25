# WebXOS 2025 Vial MCP SDK: Vial Wallet Beta Emergency Backup - Part 4 (Docker Deployment)

**Objective**: Containerize the Vial MCP Wallet Beta frontend using Docker.

**Instructions for LLM**:
1. Create `build/dockerfiles/vial.Dockerfile` for the frontend.
2. Ensure `vial.html` and `vialfolder/` are included in the image.
3. Expose port 3000 for the frontend server.
4. Integrate with backend API URL via environment variables.

## Step 1: Create Dockerfile

### build/dockerfiles/vial.Dockerfile
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY vial.html .
COPY vialfolder ./vialfolder
COPY package.json .
RUN npm install --legacy-peer-deps
EXPOSE 3000
ENV VIAL_API_URL=http://backend:8000
CMD ["npx", "serve", ".", "-p", "3000"]
```

## Step 2: Create docker-compose.yml
```yaml
version: '3.8'
services:
  vial-frontend:
    build:
      context: .
      dockerfile: build/dockerfiles/vial.Dockerfile
    ports:
      - "3000:3000"
    environment:
      - VIAL_API_URL=${VIAL_API_URL}
      - VIAL_OAUTH_REDIRECT_URI=http://localhost:3000/callback
```

## Step 3: Validation
```bash
docker-compose up -d
open http://localhost:3000/vial.html
```

**Next**: Proceed to `vial-part5.md` for CI/CD and testing.