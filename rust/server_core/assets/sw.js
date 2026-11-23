const CACHE_NAME = 'hyperstrike-pwa-v1';
const PRECACHE_URLS = [
  '/manifest.webmanifest',
  '/icon.svg',
  '/icon-192.png',
  '/icon-512.png',
  '/icon-maskable-512.png',
  '/apple-touch-icon.png',
];
const CDN_HOSTS = new Set([
  'cdn.jsdelivr.net',
  'unpkg.com',
  'fonts.googleapis.com',
  'fonts.gstatic.com',
]);

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  let url;
  try {
    url = new URL(req.url);
  } catch {
    return;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;

  if (url.origin === self.location.origin) {
    if (
      url.pathname.startsWith('/api/') ||
      url.pathname === '/sw.js' ||
      url.pathname === '/admin' ||
      url.pathname.startsWith('/admin/')
    ) {
      return;
    }

    if (
      req.mode === 'navigate' ||
      url.pathname === '/' ||
      url.pathname === '/index.html'
    ) {
      event.respondWith(
        fetch(req)
          .then((res) => {
            if (res && res.ok) {
              const copy = res.clone();
              caches.open(CACHE_NAME).then((cache) => cache.put('/', copy)).catch(() => {});
            }
            return res;
          })
          .catch(async () => {
            const cached = await caches.match('/');
            if (cached) return cached;
            return new Response(
              '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HYPERSTRIKE — Offline</title><style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#03050c;color:#eef4ff;font-family:system-ui,sans-serif;text-align:center;padding:24px}h1{letter-spacing:.12em;font-style:italic;margin:0 0 10px}p{color:#9fb2d6;margin:0 0 20px}button{background:#3ef2ff;color:#03141c;border:0;padding:12px 28px;font-weight:800;letter-spacing:.16em;cursor:pointer}</style></head><body><div><h1>HYPERSTRIKE</h1><p>You are currently offline. Reconnect to launch the game.</p><button onclick="location.reload()">RETRY</button></div></body></html>',
              {
                status: 503,
                headers: { 'Content-Type': 'text/html; charset=utf-8' },
              }
            );
          })
      );
      return;
    }

    if (PRECACHE_URLS.includes(url.pathname)) {
      event.respondWith(
        caches.match(req).then((cached) => {
          const networkFetch = fetch(req)
            .then((res) => {
              if (res && res.ok) {
                const copy = res.clone();
                caches.open(CACHE_NAME).then((cache) => cache.put(req, copy)).catch(() => {});
              }
              return res;
            })
            .catch(() => cached);
          return cached || networkFetch;
        })
      );
      return;
    }
    return;
  }

  if (CDN_HOSTS.has(url.hostname)) {
    event.respondWith(
      caches.match(req).then((cached) => {
        const networkFetch = fetch(req)
          .then((res) => {
            if (res && (res.ok || res.type === 'opaque')) {
              const copy = res.clone();
              caches.open(CACHE_NAME).then((cache) => cache.put(req, copy)).catch(() => {});
            }
            return res;
          })
          .catch(() => cached);
        return cached || networkFetch;
      })
    );
  }
});
