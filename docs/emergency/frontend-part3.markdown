# WebXOS 2025 Vial MCP SDK: Frontend Emergency Backup - Part 3 (Docker Configuration)

**Objective**: Containerize the frontend using Docker for deployment.

**Instructions for LLM**:
1. Create `build/dockerfiles/frontend.Dockerfile` for the frontend.
2. Ensure compatibility with Next.js and Tailwind CSS.
3. Expose port 3000 for the frontend server.
4. Integrate with backend API URL.

## Step 1: Create Dockerfile

### build/dockerfiles/frontend.Dockerfile
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json .
RUN npm install --legacy-peer-deps
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/next.config.js ./next.config.js
RUN npm install --production --legacy-peer-deps
EXPOSE 3000
ENV NODE_ENV=production
ENV NEXT_PUBLIC_API_URL=http://backend:8000
CMD ["npm", "start"]
```

## Step 2: Validation
```bash
docker build -f build/dockerfiles/frontend.Dockerfile -t webxos-frontend .
docker run -p 3000:3000 --env-file .env.local webxos-frontend
open http://localhost:3000
```

**Next**: Proceed to `frontend-part4.md` for CI/CD workflows.