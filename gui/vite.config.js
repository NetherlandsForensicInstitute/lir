import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // stable entry file name, so the documentation page can reference it statically
        entryFileNames: 'assets/gui.js',
      },
    },
  },
})
