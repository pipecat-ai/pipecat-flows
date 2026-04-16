---
name: update-docs
description: Update documentation pages to match source code changes on the current branch
---

Update documentation pages in the [pipecat-ai/docs](https://github.com/pipecat-ai/docs)
repository to reflect source code changes on the current branch. Analyzes the diff against
main, maps changed source files to their corresponding doc pages, and makes targeted edits.

## Arguments

`/update-docs [DOCS_PATH]`

- `DOCS_PATH` (optional): Path to the docs repository root. If not provided, ask the user.

## Instructions

### Step 1: Resolve docs path

- If `DOCS_PATH` is provided as an argument, use it. Otherwise ask the user.
- Verify the path exists and contains an `api-reference/pipecat-flows/` subdirectory.
- If verification fails, stop and report the error.

### Step 2: Create docs branch

- Get current pipecat-flows branch name: `git rev-parse --abbrev-ref HEAD`
- In docs repo, create and switch to a new branch:
  ```
  cd DOCS_PATH && git checkout main && git pull && git checkout -b {branch-name}-docs
  ```

### Step 3: Detect changed source files

- From the pipecat-flows repo root, run: `git diff main..HEAD --name-only`
- Filter results to files matching `src/pipecat_flows/*.py`.
- Track only: `types.py`, `manager.py`, `actions.py`, `adapters.py`, `exceptions.py`.
- Ignore: `__init__.py`, `__pycache__/`, test files.
- If no relevant source files changed, stop and report "no doc-affecting changes."

### Step 4: Map source files to doc pages

- Read `.claude/skills/update-docs/SOURCE_DOC_MAPPING.md`.
- For each changed source file, look up its doc page(s) in the API Reference Pages and Guide Pages tables.
- Collect the unique set of doc pages to update.

### Step 5: Analyze each source-doc pair

For each changed source file and its mapped doc page(s):

1. Read the full source file.
2. Read the diff: `git diff main..HEAD -- src/pipecat_flows/<file>`
3. Read the current doc page in full.
4. Compare based on file type:
   - **types.py** — Class fields, type signatures, type aliases vs. `<ParamField>` entries and type sections in `api-reference/pipecat-flows/types.mdx`. Also check guide pages: `nodes-and-messages.mdx`, `functions.mdx`, `context-strategies.mdx`.
   - **manager.py** — `FlowManager.__init__` parameters and methods vs. `<ParamField>` entries in `api-reference/pipecat-flows/flow-manager.mdx`. Also check `state-management.mdx`.
   - **actions.py** — `register_action` signature, built-in action types vs. action sections in `api-reference/pipecat-flows/flow-manager.mdx` and `api-reference/pipecat-flows/types.mdx`. Also check `actions.mdx`.
   - **adapters.py** — Supported LLM providers vs. LLM Provider Support table in `api-reference/pipecat-flows/overview.mdx`. Also check Cross-Provider Compatibility in `nodes-and-messages.mdx`.
   - **exceptions.py** — Exception class hierarchy and docstrings vs. `api-reference/pipecat-flows/exceptions.mdx`.
5. Also check: class names, imports, default values, and behavioral changes noted in docstrings.

### Step 6: Make targeted edits

Apply these conservative rules:

- **Never remove content** unless the corresponding source code was removed.
- **Never rewrite accurate sections** — only update what actually changed.
- **Match existing formatting** — use the same heading levels, list styles, and code block languages.
- **Keep descriptions concise** — match the voice and brevity of surrounding content.
- **Preserve CardGroup, links, and examples** — do not restructure page layout.
- **Don't touch frontmatter** unless a class or module was renamed.

### Step 7: Check guide pages

- For each changed source file, collect class names, renamed parameters, and changed imports from the diff.
- Check each mapped guide page (see SOURCE_DOC_MAPPING.md Guide Pages table) for references to these identifiers.
- If a guide references a changed API: read the full guide, update the specific references.
- If the guide only references concepts generally (not the specific changed APIs): leave it alone.
- Also check `quickstart.mdx` if FlowManager init, FlowsFunctionSchema, or handler return types changed.

### Step 8: Output summary

Print a summary in this format:

```
## Documentation Updates

### Updated reference pages
- `api-reference/pipecat-flows/types.mdx` — <brief description of changes>

### Updated guide pages
- `pipecat-flows/guides/functions.mdx` — <brief description of changes>

### Skipped files
- `src/pipecat_flows/__init__.py` — re-exports only

### No changes needed
- `api-reference/pipecat-flows/exceptions.mdx` — already up to date
```

## Guidelines

- **Be conservative** — only change what the diff warrants.
- **Read before editing** — always read the full doc page before making changes.
- **Preserve voice** — match the existing writing style of each page.
- **One PR at a time** — operate on the current branch's diff against main.
- **Parallel analysis** — when multiple source files map to different doc pages, analyze them in parallel.

## Checklist

Before finishing, verify:

- [ ] All changed source files checked against the mapping table
- [ ] Each doc page edit matches an actual source code change (not guessed)
- [ ] No content removed unless corresponding source was removed
- [ ] New parameters have accurate types and defaults from source
- [ ] Formatting matches existing page style
- [ ] Guide pages checked and updated if they reference changed APIs
