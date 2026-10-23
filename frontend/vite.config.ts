import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the FastAPI server runs on :8000; in production it serves this build.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
