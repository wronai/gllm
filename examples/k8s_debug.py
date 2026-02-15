"""preLLM Example — Kubernetes Debugging

Use case: "Dlaczego mój RPi K8s cluster ciągle pada?"
Uses configs/domains/devops_k8s.yaml for K8s-specific domain rules.
"""

from __future__ import annotations

import asyncio


async def main():
    from prellm import preprocess_and_execute

    print("=" * 60)
    print("🧠 preLLM — K8s Debugging Example")
    print("=" * 60)

    # Example 1: CrashLoopBackOff diagnosis
    print("\n--- 1. CrashLoopBackOff ---")
    result = await preprocess_and_execute(
        query="Mój pod backend-api w namespace production ciągle restartuje się z CrashLoopBackOff",
        config_path="configs/domains/devops_k8s.yaml",
        strategy="structure",
    )
    print(f"Content: {result.content[:200]}...")
    if result.decomposition and result.decomposition.matched_rule:
        print(f"Rule: {result.decomposition.matched_rule}")
    if result.decomposition and result.decomposition.missing_fields:
        print(f"Missing: {result.decomposition.missing_fields}")

    # Example 2: OOM diagnosis with context
    print("\n--- 2. OOM with context ---")
    result = await preprocess_and_execute(
        query="Kubernetes pods killed by OOM on RPi cluster",
        config_path="configs/domains/devops_k8s.yaml",
        strategy="enrich",
        user_context={
            "cluster": "rpi-k3s-prod",
            "namespace": "backend",
            "node_ram": "4GB",
            "k8s_version": "1.28",
        },
    )
    print(f"Content: {result.content[:200]}...")

    # Example 3: Scaling decision
    print("\n--- 3. HPA Scaling ---")
    result = await preprocess_and_execute(
        query="Skonfiguruj autoscaling dla deployment frontend z min 2 max 10 replik",
        config_path="configs/domains/devops_k8s.yaml",
    )
    print(f"Content: {result.content[:200]}...")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
