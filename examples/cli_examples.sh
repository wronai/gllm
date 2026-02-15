#!/usr/bin/env bash
# preLLM CLI Examples (v0.3.8) — shell commands for direct usage
#
# Install: pip install prellm
# Ollama:  ollama serve && ollama pull qwen2.5:3b

echo "=== preLLM CLI Examples (v0.3.8) ==="
echo ""

# ─────────────────────────────────────────────
# 1. Basic query (zero-config)
# ─────────────────────────────────────────────
echo "--- 1. Basic Query ---"
echo '$ prellm query "Explain Docker networking"'
prellm query "Explain Docker networking"
echo ""

# ─────────────────────────────────────────────
# 2. Custom models
# ─────────────────────────────────────────────
echo "--- 2. Custom Models ---"
echo '$ prellm query "Deploy app" --small ollama/qwen2.5:3b --large gpt-4o-mini'
prellm query "Deploy app to staging" \
  --small ollama/qwen2.5:3b \
  --large gpt-4o-mini
echo ""

# ─────────────────────────────────────────────
# 3. Structure strategy with JSON output
# ─────────────────────────────────────────────
echo "--- 3. Structure Strategy (JSON) ---"
prellm query "Refaktoryzuj kod z hardcode'ami" \
  --strategy structure \
  --json
echo ""

# ─────────────────────────────────────────────
# 4. With user context
# ─────────────────────────────────────────────
echo "--- 4. With Context ---"
prellm query "Zdiagnozuj problem z K8s podami" \
  --context "gdansk_embedded_python_docker_k8s" \
  --strategy enrich
echo ""

# ─────────────────────────────────────────────
# 5. With YAML config (domain-specific)
# ─────────────────────────────────────────────
echo "--- 5. Domain Config (K8s) ---"
prellm query "CrashLoopBackOff na namespace backend" \
  --config configs/domains/devops_k8s.yaml \
  --strategy structure
echo ""

# ─────────────────────────────────────────────
# 6. Decompose only (no large LLM call)
# ─────────────────────────────────────────────
echo "--- 6. Decompose Only ---"
prellm decompose "Deploy app to production" \
  --strategy structure \
  --json
echo ""

# ─────────────────────────────────────────────
# 7. With UserMemory context
# ─────────────────────────────────────────────
echo "--- 7. With Memory ---"
echo '$ prellm query "Deploy v2" --memory .prellm/user_memory.db'
prellm query "Deploy v2 to staging" \
  --memory .prellm/user_memory.db
echo ""

# ─────────────────────────────────────────────
# 8. With CodebaseIndexer context
# ─────────────────────────────────────────────
echo "--- 8. With Codebase Context ---"
echo '$ prellm query "refactor deploy" --codebase ./src'
prellm query "Refaktoryzuj funkcję deploy" \
  --codebase .
echo ""

# ─────────────────────────────────────────────
# 9. OpenRouter model (Kimi K2.5)
# ─────────────────────────────────────────────
echo "--- 9. OpenRouter (Kimi K2.5) ---"
echo '$ prellm query "Design API" --large openrouter/moonshotai/kimi-k2.5'
prellm query "Design a REST API for user management" \
  --large openrouter/moonshotai/kimi-k2.5
echo ""

# ─────────────────────────────────────────────
# 10. Generate config
# ─────────────────────────────────────────────
echo "--- 10. Generate Config ---"
prellm init --devops -o /tmp/prellm_example_config.yaml
cat /tmp/prellm_example_config.yaml
echo ""

# ─────────────────────────────────────────────
# 11. Start API server
# ─────────────────────────────────────────────
echo "--- 11. Start API Server ---"
echo '$ prellm serve --small ollama/qwen2.5:3b --large gpt-4o-mini --port 8080'
echo "(not running — use this command to start)"
echo ""

# ─────────────────────────────────────────────
# 12. List models & doctor
# ─────────────────────────────────────────────
echo "--- 12. Models & Doctor ---"
prellm models --provider ollama
prellm doctor
echo ""

echo "=== Done ==="
