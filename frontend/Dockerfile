# ==============================================================================
# BookRAG AI - Frontend Dockerfile (Phase 19 Production Multi-Stage Build)
# ==============================================================================

# Stage 1: Build static assets
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source code and config files
COPY frontend ./

# Build production bundle with TypeScript check
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy built static artifacts
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration for SPA routing
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
