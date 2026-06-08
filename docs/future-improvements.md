# Future Improvements Roadmap

This document tracks the next useful extensions for DES-Agent after the current router and example updates.

## Recently Completed

1. Example benchmark suite
   - Curated examples now act as a regression benchmark for routing, DES screening, viscosity, and metal-binding.

2. Machine-readable exports
   - DES runs now emit JSON, CSV, and a run manifest automatically.

3. Stronger natural-language normalization
   - Plain-language requests now get normalized before routing, including salt and free-base clarification.

4. Active-learning feedback loop
   - Labeled run memory can now be reused from a single run or a whole history directory so later DES runs can learn from prior `good` / `bad` feedback more directly.
