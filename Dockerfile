FROM node:22-alpine

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8080
ENV DATA_DIR=/data

COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund

COPY server.js sim.js match.js index.html ./

RUN mkdir -p /data && chown -R node:node /app /data
USER node

EXPOSE 8080

CMD ["node", "--max-old-space-size=192", "server.js"]
