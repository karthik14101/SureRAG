import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Tailwind v4 is configured through the Vite plugin plus an `@import "tailwindcss"`
// in index.css. There is deliberately no tailwind.config.js or postcss.config.js:
// v4 dropped that setup, and leaving stale config files around breaks the build.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // The frontend calls /api/... and Vite forwards it to FastAPI, so there are
      // no CORS preflights in development and no hard-coded backend URL in the code.
      '/api': {
        // 127.0.0.1 rather than "localhost" on purpose: Node 18+ resolves
        // localhost to IPv6 ::1 first, and the backend binds IPv4 loopback, so
        // "localhost" here intermittently fails with ECONNREFUSED.
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Server-sent events must not be buffered by the proxy.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
            }
          })
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
