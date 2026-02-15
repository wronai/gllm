"""preLLM Example — Embedded Systems Refactoring

Use case: "Zrefaktoruj mój ESP32 monitoring system"
Uses configs/domains/embedded.yaml for hardware-aware domain rules.
"""

from __future__ import annotations

import asyncio


async def main():
    from prellm import preprocess_and_execute

    print("=" * 60)
    print("🧠 preLLM — Embedded Refactoring Example")
    print("=" * 60)

    # Example 1: ESP32 refactoring with constraints
    print("\n--- 1. ESP32 Monitoring Refactor ---")
    result = await preprocess_and_execute(
        query="Zrefaktoruj mój ESP32 monitoring system - za dużo hardcode'ów, brak OTA, zużywa 200mA w idle",
        config_path="configs/domains/embedded.yaml",
        strategy="structure",
        user_context={
            "mcu": "ESP32-S3",
            "flash": "8MB",
            "ram": "512KB",
            "framework": "ESP-IDF 5.1",
            "sensors": "BME280, MPU6050, GPS NEO-6M",
        },
    )
    print(f"Content: {result.content[:200]}...")
    if result.decomposition and result.decomposition.matched_rule:
        print(f"Rule: {result.decomposition.matched_rule}")
    if result.decomposition and result.decomposition.missing_fields:
        print(f"Missing: {result.decomposition.missing_fields}")

    # Example 2: Power optimization
    print("\n--- 2. Power Optimization ---")
    result = await preprocess_and_execute(
        query="Zoptymalizuj zużycie prądu - bateria 3000mAh musi starczyć na 30 dni, pomiar co 5 minut",
        config_path="configs/domains/embedded.yaml",
        strategy="enrich",
    )
    print(f"Content: {result.content[:200]}...")

    # Example 3: FreeRTOS task design
    print("\n--- 3. FreeRTOS Tasks ---")
    result = await preprocess_and_execute(
        query="Zaprojektuj architekturę tasków FreeRTOS: sensor reading, WiFi upload, display update, watchdog",
        config_path="configs/domains/embedded.yaml",
        user_context={"mcu": "STM32F407", "rtos": "FreeRTOS", "stack": "4KB per task"},
    )
    print(f"Content: {result.content[:200]}...")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
