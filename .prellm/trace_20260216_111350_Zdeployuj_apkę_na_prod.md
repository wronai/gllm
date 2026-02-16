# preLLM Execution Trace

> **Query**: `Zdeployuj apkę na prod`
> **Timestamp**: 2026-02-16 11:13:50
> **Total duration**: 14659ms

## Configuration

| Parameter | Value |
|---|---|
| `small_llm` | `ollama/qwen:7b` |
| `large_llm` | `openrouter/google/gemini-3-flash-preview` |
| `strategy` | `classify` |

## Decision Path

### Step 1: Configuration ✅

Resolved models, strategy, and pipeline parameters.

- **Type**: `config`
- **Status**: ok

<details>
<summary>Outputs</summary>

```json
{
  "small_llm": "ollama/qwen:7b",
  "large_llm": "openrouter/google/gemini-3-flash-preview",
  "strategy": "classify",
  "config_path": null,
  "user_context": null
}
```
</details>

---

### Step 2: Pipeline: classify ✅

llm step in 'classify' pipeline

- **Type**: `llm_call`
- **Status**: ok

<details>
<summary>Outputs</summary>

```json
{
  "classification": {
    "intent": "deploy",
    "confidence": 0.9,
    "domain": "mobile"
  }
}
```
</details>

---

### Step 3: Pipeline: match_rule ✅

algo step in 'classify' pipeline

- **Type**: `pipeline_step`
- **Status**: ok

<details>
<summary>Outputs</summary>

```json
{
  "matched_rule": {}
}
```
</details>

---

### Step 4: Pipeline: enrich_if_needed ⏭️

llm step in 'classify' pipeline

- **Type**: `llm_call`
- **Status**: skipped

<details>
<summary>Outputs</summary>

```json
{
  "enriched_query": null
}
```
</details>

---

### Step 5: PreprocessorAgent.preprocess() ✅

Small LLM (ollama/qwen:7b) preprocessed query using 'classify' strategy.

- **Type**: `agent`
- **Duration**: 3138ms
- **Status**: ok

<details>
<summary>Inputs</summary>

```json
{
  "query": "Zdeployuj apkę na prod",
  "pipeline": "classify",
  "user_context": {
    "shell_context": "{\"env_vars\":{\"SHELL\":\"/bin/bash\",\"QT_ACCESSIBILITY\":\"1\",\"SNAP_REVISION\":\"578\",\"PYENV_SHELL\":\"bash\",\"XDG_CONFIG_DIRS\":\"/etc/xdg/xdg-ubuntu:/etc/xdg\",\"NVM_INC\":\"/home/tom/.nvm/versions/node/v20.19.5/include/node\",\"XDG_MENU_PREFIX\":\"gnome-\",\"CONDA_EXE\":\"/home/tom/miniconda3/bin/conda\",\"_CE_M\":\"\",\"QT_IM_MODULES\":\"wayland;ibus\",\"SNAP_REAL_HOME\":\"/home/tom\",\"TERMINAL_EMULATOR\":\"JetBrains-JediTerm\",\"SNAP_USER_COMMON\":\"/home/tom/snap/pycharm-professional/common\",\"PROCESS_LAUNCHED_BY_Q\":\"1\",\"MEMORY_PRESSURE_WRITE\":\"c29tZSAyMDAwMDAgMjAwMDAwMAA=\",\"SNAP_INSTANCE_KEY\":\"\",\"XMODIFIERS\":\"@im=ibus\",\"SNAP_EUID\":\"1000\",\"PWD\":\"/home/tom/github/wronai/prellm\",\"PYENV_VIRTUALENV_INIT\":\"1\",\"LOGNAME\":\"tom\",\"XDG_SESSION_TYPE\":\"wayland\",\"CONDA_PREFIX\":\"/home/tom/miniconda3\",\"PROCESS_LAUNCHED_BY_CW\":\"1\",\"PNPM_HOME\":\"/home/tom/.local/share/pnpm\",\"GPG_AGENT_INFO\":\"/run/user/1000/gnupg/S.gpg-agent:0:1\",\"SYSTEMD_EXEC_PID\":\"9894\",\"DESKTOP_STARTUP_ID\":\"gnome-shell/PyCharm/9894-0-nvidia_TIME50588\",\"SNAP_CONTEXT\":\"1VBG0g",
    "runtime_context": {
      "env_safe": {
        "SHELL": "/bin/bash",
        "QT_ACCESSIBILITY": "1",
        "SNAP_REVISION": "578",
        "PYENV_SHELL": "bash",
        "XDG_CONFIG_DIRS": "/etc/xdg/xdg-ubuntu:/etc/xdg",
        "NVM_INC": "/home/tom/.nvm/versions/node/v20.19.5/include/node",
        "XDG_MENU_PREFIX": "gnome-",
        "GNOME_DESKTOP_SESSION_ID": "this-is-deprecated",
        "CONDA_EXE": "/home/tom/miniconda3/bin/conda",
        "_CE_M": "",
        "QT_IM_MODULES": "wayland;ibus",
        "SNAP_REAL_HOME": "/home/tom",
        "TERMINAL_EMULATOR": "JetBrains-JediTerm",
        "SNAP_USER_COMMON": "/home/tom/snap/pycharm-professional/common",
        "PROCESS_LAUNCHED_BY_Q": "1",
        "GNOME_SHELL_SESSION_MODE": "ubuntu",
        "S
... (truncated)
```
</details>

<details>
<summary>Outputs</summary>

```json
{
  "executor_input": "Zdeployuj apkę na prod"
}
```
</details>

---

### Step 6: ExecutorAgent.execute() ✅

Large LLM (openrouter/google/gemini-3-flash-preview) generated final response.

- **Type**: `llm_call`
- **Duration**: 11459ms
- **Status**: ok

<details>
<summary>Inputs</summary>

```json
{
  "executor_input": "Zdeployuj apkę na prod"
}
```
</details>

<details>
<summary>Outputs</summary>

```json
{
  "content_preview": "Żebyśmy mogli to zrobić sprawnie, potrzebuję od Ciebie kilku konkretów. \"Deploy na prod\" może oznaczać wszystko – od wrzucenia plików na FTP po skomplikowany pipeline w Kubernetesie.\n\nOdpowiedz na te 4 pytania, a przygotuję Ci gotową instrukcję lub skrypt:\n\n1.  **Co to za aplikacja?** (np. React, No",
  "model": "openrouter/google/gemini-3-flash-preview"
}
```
</details>

- **retries**: `0`

---

## Result

**Response** (1566 chars):

```
Żebyśmy mogli to zrobić sprawnie, potrzebuję od Ciebie kilku konkretów. "Deploy na prod" może oznaczać wszystko – od wrzucenia plików na FTP po skomplikowany pipeline w Kubernetesie.

Odpowiedz na te 4 pytania, a przygotuję Ci gotową instrukcję lub skrypt:

1.  **Co to za aplikacja?** (np. React, Node.js, Python/Django, PHP, statyczny HTML).
2.  **Gdzie to ma stać?** (np. AWS, Azure, Google Cloud, Heroku, własny VPS, Vercel, Netlify).
3.  **Czy masz już infrastrukturę?** (Czy serwer jest kupiony...
```

- **model_used**: `openrouter/google/gemini-3-flash-preview`
- **small_model_used**: `ollama/qwen:7b`
- **retries**: `0`
- **strategy**: `classify`
- **classification**: `{'intent': 'deploy', 'confidence': 0.9, 'domain': 'mobile'}`

## Summary

| # | Step | Type | Duration | Status |
|---|---|---|---|---|
| 1 | Configuration | `config` | — | ✅ ok |
| 2 | Pipeline: classify | `llm_call` | — | ✅ ok |
| 3 | Pipeline: match_rule | `pipeline_step` | — | ✅ ok |
| 4 | Pipeline: enrich_if_needed | `llm_call` | — | ⏭️ skipped |
| 5 | PreprocessorAgent.preprocess() | `agent` | 3138ms | ✅ ok |
| 6 | ExecutorAgent.execute() | `llm_call` | 11459ms | ✅ ok |

**Total**: 14659ms
