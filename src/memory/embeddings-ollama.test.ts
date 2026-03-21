import { describe, it, expect, vi } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import {
  createOllamaEmbeddingProvider,
  formatEmbeddingGemmaDocumentPrompt,
  getEmbeddingGemmaDocumentTitle,
  getEmbeddingGemmaDocumentTitleFromPath,
  getOllamaEmbeddingStrategyVersion,
} from "./embeddings-ollama.js";

describe("embeddings-ollama", () => {
  it("calls /api/embeddings and returns normalized vectors", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [3, 4] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {} as OpenClawConfig,
      provider: "ollama",
      model: "nomic-embed-text",
      fallback: "none",
      remote: { baseUrl: "http://127.0.0.1:11434" },
    });

    const v = await provider.embedQuery("hi");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // normalized [3,4] => [0.6,0.8]
    expect(v[0]).toBeCloseTo(0.6, 5);
    expect(v[1]).toBeCloseTo(0.8, 5);
  });

  it("resolves baseUrl/apiKey/headers from models.providers.ollama and strips /v1", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [1, 0] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {
        models: {
          providers: {
            ollama: {
              baseUrl: "http://127.0.0.1:11434/v1",
              apiKey: "ollama-\nlocal\r\n", // pragma: allowlist secret
              headers: {
                "X-Provider-Header": "provider",
              },
            },
          },
        },
      } as unknown as OpenClawConfig,
      provider: "ollama",
      model: "",
      fallback: "none",
    });

    await provider.embedQuery("hello");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/embeddings",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          Authorization: "Bearer ollama-local",
          "X-Provider-Header": "provider",
        }),
      }),
    );
  });

  it("fails fast when memory-search remote apiKey is an unresolved SecretRef", async () => {
    await expect(
      createOllamaEmbeddingProvider({
        config: {} as OpenClawConfig,
        provider: "ollama",
        model: "nomic-embed-text",
        fallback: "none",
        remote: {
          baseUrl: "http://127.0.0.1:11434",
          apiKey: { source: "env", provider: "default", id: "OLLAMA_API_KEY" },
        },
      }),
    ).rejects.toThrow(/agents\.\*\.memorySearch\.remote\.apiKey: unresolved SecretRef/i);
  });

  it("falls back to env key when models.providers.ollama.apiKey is an unresolved SecretRef", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [1, 0] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    vi.stubEnv("OLLAMA_API_KEY", "ollama-env");

    const { provider } = await createOllamaEmbeddingProvider({
      config: {
        models: {
          providers: {
            ollama: {
              baseUrl: "http://127.0.0.1:11434/v1",
              apiKey: { source: "env", provider: "default", id: "OLLAMA_API_KEY" },
              models: [],
            },
          },
        },
      } as unknown as OpenClawConfig,
      provider: "ollama",
      model: "nomic-embed-text",
      fallback: "none",
    });

    await provider.embedQuery("hello");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/embeddings",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer ollama-env",
        }),
      }),
    );
  });

  it("formats embeddinggemma queries with the retrieval query prompt", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [1, 0] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {} as OpenClawConfig,
      provider: "ollama",
      model: "embeddinggemma:latest",
      fallback: "none",
      remote: { baseUrl: "http://127.0.0.1:11434" },
    });

    await provider.embedQuery("bootstrap workspace");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/embeddings",
      expect.objectContaining({
        body: JSON.stringify({
          model: "embeddinggemma:latest",
          prompt: "task: search result | query: bootstrap workspace",
        }),
      }),
    );
  });

  it("formats embeddinggemma documents with title-aware prompts", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [1, 0] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {} as OpenClawConfig,
      provider: "ollama",
      model: "embeddinggemma:latest",
      fallback: "none",
      remote: { baseUrl: "http://127.0.0.1:11434" },
    });

    await provider.embedBatch([formatEmbeddingGemmaDocumentPrompt("memory body", "session-title")]);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/embeddings",
      expect.objectContaining({
        body: JSON.stringify({
          model: "embeddinggemma:latest",
          prompt: "title: session-title | text: memory body",
        }),
      }),
    );
  });

  it("prefers the first markdown heading for embeddinggemma document titles", () => {
    expect(
      getEmbeddingGemmaDocumentTitle(
        "intro\n# Session Summary\nBody text",
        "memory/2026-03-21-session.md",
      ),
    ).toBe("Session Summary");
  });

  it("falls back to the filename stem for embeddinggemma document titles", () => {
    expect(getEmbeddingGemmaDocumentTitleFromPath("memory/2026-03-21-session.md")).toBe(
      "2026-03-21-session",
    );
  });

  it("uses a distinct strategy version for embeddinggemma", () => {
    expect(getOllamaEmbeddingStrategyVersion("embeddinggemma:latest")).toBe(
      "embeddinggemma-rag-v3",
    );
    expect(getOllamaEmbeddingStrategyVersion("nomic-embed-text")).toBe("default-v1");
  });
});
