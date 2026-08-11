# Text-model freshness usage sample — 2026-08-06

Conversation minutes estimate actual user + assistant words at 150 spoken words/minute.
Benchmark process time is not used as conversation time. Recovery calls are included when billed.

| Model configuration | Provider | Status | Attempts | Rows | Input tokens | Cached input | Cache write | Output tokens | Est. speech min | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| claude-sonnet-4-6 | Anthropic | complete | 1 | 30 | 180 | 470,341 | 17,896 | 3,483 | 13.48 | `20260806T083608_claude-sonnet-4-6_cf993c71` |
| claude-fable-5 (low) | Anthropic | complete | 1 | 30 | 120 | 655,385 | 24,680 | 4,386 | 10.73 | `20260806T083806_claude-fable-5_6b537ca4` |
| claude-fable-5 (default) | Anthropic | complete | 1 | 30 | 120 | 692,249 | 5,736 | 5,731 | 12.37 | `20260806T084312_claude-fable-5_93f98482` |
| glm-5.2 (none) | BaseTen | complete | 1 | 30 | 440,484 | 407,872 | 0 | 3,206 | 15.18 | `20260806T084936_zai-org_GLM-5.2_e2e5e9fb` |
| claude-haiku-4-5 | Anthropic | complete | 1 | 30 | 180 | 487,644 | 19,113 | 4,576 | 18.13 | `20260806T084616_claude-haiku-4-5_4d252a64` |
| gpt-5.1 | OpenAI | complete | 1 | 31 | 481,351 | 460,544 | 0 | 5,594 | 22.15 | `20260806T083608_gpt-5.1_f75bfc75` |
| gpt-5.6-terra (medium) | OpenAI | complete | 1 | 30 | 388,234 | 374,182 | 0 | 2,049 | 6.92 | `20260806T083737_gpt-5.6-terra_a646f070` |
| gpt-5.5 (none) | OpenAI | complete | 1 | 32 | 427,294 | 381,952 | 0 | 2,053 | 9.61 | `20260806T082003_gpt-5.5_16467539` |
| qwen3.6-27b (thinking off) | BaseTen | complete | 1 | 30 | 460,124 | 0 | 0 | 2,859 | 12.27 | `20260806T084913_Qwen_Qwen3.6-27B_d7928f1e` |
| gemini-3.6-flash (minimal) | AI Studio | complete | 1 | 30 | 424,598 | 233,527 | 0 | 1,651 | 8.02 | `20260806T085314_gemini-3.6-flash_483a3e95` |
| gpt-5.4 (low) | OpenAI | complete | 1 | 31 | 412,949 | 382,976 | 0 | 2,641 | 10.33 | `20260806T083858_gpt-5.4_7ebb8ac3` |
| gpt-5.6-sol (none) | OpenAI | complete | 1 | 31 | 401,161 | 387,089 | 0 | 1,465 | 7.20 | `20260806T084011_gpt-5.6-sol_d99b3147` |
| gpt-4.1 | OpenAI | complete | 1 | 32 | 437,632 | 409,984 | 0 | 2,615 | 13.29 | `20260806T083353_gpt-4.1_cf38fb0c` |
| gpt-5.4 (none, +96 dots) | OpenAI | complete | 1 | 31 | 418,529 | 380,928 | 0 | 2,111 | 10.79 | `20260806T084312_gpt-5.4_843b8e98` |
| inkling (none) | BaseTen | complete | 1 | 32 | 426,654 | 398,368 | 0 | 1,876 | 8.87 | `20260806T085118_thinkingmachines_inkling_4cfc9e3b` |
| gpt-4o | OpenAI | complete | 1 | 32 | 428,452 | 412,032 | 0 | 2,225 | 11.35 | `20260806T084436_gpt-4o_3e8bf5ec` |
| kimi-k2.6 (thinking off) | BaseTen | complete | 1 | 31 | 425,499 | 385,600 | 0 | 2,244 | 12.11 | `20260806T091743_moonshotai_Kimi-K2.6_fba32308` |
| gemini-3.5-flash (minimal) | AI Studio | complete | 1 | 30 | 435,127 | 233,607 | 0 | 2,332 | 11.39 | `20260806T085350_gemini-3.5-flash_3545a83d` |
| claude-sonnet-5 | Anthropic | complete | 1 | 32 | 128 | 712,286 | 24,995 | 4,546 | 12.47 | `20260806T084735_claude-sonnet-5_414eabac` |
| qwen3.6-35b-a3b (thinking off, FP8) | BaseTen | complete | 1 | 36 | 568,893 | 0 | 0 | 3,331 | 15.55 | `20260806T090616_Qwen_Qwen3.6-35B-A3B-FP8_cd7e7f24` |
| gpt-5.6-terra (none) | OpenAI | complete | 1 | 33 | 427,367 | 413,249 | 0 | 1,485 | 7.33 | `20260806T084526_gpt-5.6-terra_049bd0b1` |
| gpt-5.4 (none) | OpenAI | complete | 1 | 34 | 454,508 | 433,152 | 0 | 1,965 | 10.59 | `20260806T084835_gpt-5.4_0642fd08` |
| gemini-2.5-flash (thinking off) | AI Studio | complete | 1 | 33 | 465,386 | 283,767 | 0 | 1,727 | 9.57 | `20260806T085440_gemini-2.5-flash_a6ebce8f` |
| gpt-5.2 | OpenAI | complete | 1 | 34 | 463,951 | 447,872 | 0 | 2,447 | 10.67 | `20260806T084954_gpt-5.2_b1abc574` |
| gpt-5.6-luna (none) | OpenAI | complete | 1 | 34 | 440,137 | 426,070 | 0 | 1,472 | 7.91 | `20260806T085106_gpt-5.6-luna_78e83341` |
| gpt-oss-120b (groq) | Groq | complete | 1 | 33 | 494,470 | 232,192 | 0 | 6,563 | 17.34 | `20260806T085942_openai_gpt-oss-120b_0f174486` |
| poolside/laguna-s-2.1 (thinking off) | OpenRouter | complete | 1 | 35 | 535,914 | 504,320 | 0 | 3,005 | 14.12 | `20260806T090024_poolside_laguna-s-2.1_84846e95` |
| gpt-4.1-mini | OpenAI | complete | 1 | 34 | 457,573 | 188,544 | 0 | 2,107 | 12.35 | `20260806T085218_gpt-4.1-mini_36bc4f88` |
| gpt-5-mini | OpenAI | complete | 1 | 34 | 485,939 | 464,000 | 0 | 3,541 | 16.23 | `20260806T085408_gpt-5-mini_78fdf1f6` |
| gpt-4o-mini | OpenAI | complete | 1 | 32 | 431,191 | 416,256 | 0 | 2,233 | 11.19 | `20260806T085659_gpt-4o-mini_c7b45030` |
| qwen3-8b (thinking off, BaseTen) | BaseTen | complete | 1 | 33 | 532,646 | 0 | 0 | 4,427 | 17.66 | `20260806T091307_qwen_qwen3-8b_f18ed148` |
| gemma-4-26b-a4b-it (thinking off) | BaseTen | complete | 1 | 35 | 507,380 | 489,760 | 0 | 2,260 | 12.73 | `20260806T092333_google_gemma-4-26B-A4B-it_54dc9f77` |
| gemini-3.5-flash-lite (minimal) | AI Studio | complete | 2 | 31 | 444,970 | 0 | 0 | 2,033 | 9.79 | `20260806T083414_gemini-3.5-flash-lite_7db73e40` |
