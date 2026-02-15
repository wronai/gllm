"""preLLM Example — Polish Leasing Calculator

Use case: "Stwórz kalkulator leasingu camper van PLN"
Uses configs/domains/polish_finance.yaml for Polish financial domain rules.
"""

from __future__ import annotations

import asyncio


async def main():
    from prellm import preprocess_and_execute

    print("=" * 60)
    print("🧠 preLLM — Polish Finance Example")
    print("=" * 60)

    # Example 1: Leasing calculation
    print("\n--- 1. Leasing Camper Van ---")
    result = await preprocess_and_execute(
        query="Oblicz ratę leasingu operacyjnego camper van za 250000 PLN netto, 48 miesięcy, WIBOR 3M + 2.5%",
        config_path="configs/domains/polish_finance.yaml",
        strategy="structure",
    )
    print(f"Content: {result.content[:200]}...")
    if result.decomposition and result.decomposition.matched_rule:
        print(f"Rule: {result.decomposition.matched_rule}")

    # Example 2: Invoice generation
    print("\n--- 2. Faktura VAT ---")
    result = await preprocess_and_execute(
        query="Wygeneruj fakturę VAT dla NIP 5213000001, nabywca NIP 1234567890, usługa IT 10000 PLN netto",
        config_path="configs/domains/polish_finance.yaml",
    )
    print(f"Content: {result.content[:200]}...")

    # Example 3: Tax calculation
    print("\n--- 3. PIT Calculation ---")
    result = await preprocess_and_execute(
        query="Oblicz podatek PIT dla przychodu 180000 PLN rocznie, liniowy 19% vs skala podatkowa",
        config_path="configs/domains/polish_finance.yaml",
        strategy="enrich",
        user_context="samozatrudnienie_IT_B2B",
    )
    print(f"Content: {result.content[:200]}...")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
