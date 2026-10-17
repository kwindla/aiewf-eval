# Grok 4.6 (low) — AIEWF medium context

- Cohort: 30 canonical conversations; 26 reached all 30 scripted turns.
- Completion failures: three turn-0 `end_session` calls and one turn-28 `end_session` call; all missing future turns remain fixed-denominator failures.
- Strict pass rate: 740/900 (82.2%), conversation-cluster bootstrap 95% CI 70.9–91.2%.
- Errors: tool 14.2%, instruction 17.7%, KB 10.1%.
- Warm TTFAT: 2000ms P50, 5241ms P95, 17404ms max across 797 later scripted and recovery responses; scripted turn 0 is excluded.
- Warm raw TTFT: 906ms P50, 1638ms P95. Median reasoning-to-answer gap: 1002ms.
- Stateful Responses caching: 98.7% median warm cached-input share; maximum request input was 20,121 tokens, below xAI's 200K higher-price threshold.
- Standard-context price estimate: $0.0104 per observed turn ($8.57 total), using $2/M uncached input, $0.50/M cached input, and $6/M output including reasoning.
