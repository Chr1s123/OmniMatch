# Real API Worktree Configuration Design

## Goal

Run the homogeneous-AgentLoop feature worktree with the existing real provider
credentials, without changing deterministic `test` or `submission` defaults and
without duplicating secrets.

## Current State

The main checkout has an ignored `.env` configured for the `dev` profile with:

- an OpenAI-compatible LLM provider and `qwen3.5-flash` model;
- SerpApi product search;
- Serper web search;
- local rate-table shipping, in-memory state, and heuristic evaluation.

The required LLM, SerpApi, and Serper credentials are non-empty. The feature
worktree has no `.env`, so commands started there cannot load those settings.

## Considered Approaches

1. **Relative `.env` symlink (selected):** link the worktree `.env` to the main
   checkout `.env`. This keeps one secret source and the ignored worktree entry
   cannot be committed.
2. **Copy `.env`:** works independently but duplicates credentials and can drift.
3. **Import variables for every command:** avoids a filesystem entry but is easy
   to omit and makes repeatable verification harder.

## Design

Create the ignored relative link `.env -> ../../.env` inside the feature
worktree. Do not modify `.env.example`, provider defaults, tracked source files,
or the submission profile.

Validate in two stages:

1. Load `OmniMatchSettings` from the worktree and print only non-secret provider
   names/modes. All six provider modes must report `real` for the `dev` profile.
2. Make one minimal request to each external provider: LLM planning, SerpApi
   product search, and Serper web search. Rate-table shipping is local and does
   not need a network request. Do not print keys, authorization headers, or full
   environment values.

This is the smallest useful real-provider validation: three external requests in
total. It proves authentication and response normalization while limiting usage.

## Failure Handling

- Configuration validation failures report the missing variable name.
- Authentication, quota, network, or provider response failures are reported by
  provider and category without exposing credentials.
- A failed provider does not trigger retries or additional API calls unless the
  user explicitly requests them.
- If link creation would overwrite an existing worktree `.env`, stop instead.

## Success Criteria

- The worktree loads the `dev` profile with no placeholder provider modes.
- The LLM returns a normalized action response.
- Product search returns a normalized provider result.
- Web search returns a normalized provider result.
- `git status --short` remains clean because `.env` is ignored.
- No secret value is printed or added to Git.
