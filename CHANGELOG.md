## 0.0.1 - 2026-04-19

### Changed
- Migrated from Anthropic SDK to z.ai SDK (breaking change)
  - Created new zai_client.py using ZaiClient from zai SDK
  - Updated requirements.txt: replaced anthropic>=0.49.0 with zai-sdk>=0.2.0
  - Updated imports across codebase to use zai_client
  - Updated .env.example with ZAI_API_KEY environment variable
  - Set ZAI_API_KEY environment variable on Railway services

### Migration Notes
This is a breaking change requiring manual intervention:
- Deployed services will need to pull updated dependencies (pip install zai-sdk)
- ZAI_API_KEY must be set as new environment variable
- Old ANTHROPIC_API_KEY is deprecated but retained for backward compatibility

