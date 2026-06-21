FROM node:18-alpine AS builder

WORKDIR /app

# Copiar package files
COPY frontend/package.json frontend/package-lock.json ./

# Instalar dependencias
RUN npm install

# Copiar código
COPY frontend ./

# Build
RUN npm run build

# Stage 2: Servir con nginx
FROM nginx:alpine

# Copiar build a nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Copiar configuración nginx
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
