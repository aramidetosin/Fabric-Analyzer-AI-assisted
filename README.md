# Fabric Analyzer (AI-assisted)

This tool collects operational data from a set of Juniper switches via the XML-RPC REST endpoint and uses an LLM to (1) summarize each switch and (2) reason about the *entire fabric* as a system. Results are rendered as rich terminal tables and saved to JSON for auditing.

---

## Key Goals

* **Zero-touch collection:** Hit the devices’ REST XML-RPC endpoint; no SSH/Junoscript needed.
* **Two-stage AI analysis:**

  1. Summarize each switch (interfaces/BGP/VXLAN/EVPN/LLDP/routes/issues).
  2. Analyze the complete fabric (topology, health, consistency, gaps).
* **Provider-agnostic:** Run with **Claude**, **OpenAI**, or **Ollama** (local/remote).
* **Operator-friendly UX:** Clean `rich` tables + deterministic prompts + token-safe truncation.
* **Auditable outputs:** Save raw collection and AI analyses to timestamped JSON files.

---

## Design Approach

### 1) Source-of-truth = Device XML-RPC

* The script sends a predefined set of Junos XML RPCs to each switch:

  * Interface details, BGP summary, route summary, system info, VXLAN TEPs, EVPN DB, LLDP neighbors, LACP interfaces.
* Results are gathered concurrently (thread pool), per-device.

### 2) Token-aware summarization per device

* Each command’s XML payload is truncated to ~10k chars to prevent prompt overflows while retaining context.
* A strict **JSON-only** schema is enforced in the system prompt so the LLM returns machine-readable output.
* Per-switch output includes: hostname/model/version/role, interface counts/errors, BGP peers & state, VXLAN tunnels/VTEPs, EVPN VNIs/MACs, LLDP neighbors, route mix, and an `issues[]` list.

### 3) Fabric-level reasoning

* A second prompt ingests all per-switch summaries and infers:

  * **Topology** (spine/leaf/border/unknown) + connections via LLDP.
  * **BGP underlay/overlay health** and failed sessions.
  * **VXLAN mesh completeness** (expected vs. actual tunnels, missing links).
  * **EVPN consistency** (VNIs/MAC route distribution).
  * **Issues** with severity & recommendations with priority/benefit.
* Output is again strict JSON to keep it dependable.

### 4) Robust, provider-agnostic LLM layer

* Selectable providers:

  * **Claude** via `langchain_anthropic`
  * **OpenAI** via `langchain_openai`
  * **Ollama** via a small `OllamaWrapper` implementing the LangChain `LLM` interface
* The Ollama wrapper uses a **Pydantic `PrivateAttr`** to hold the `Client` safely and supports a **remote host** (e.g. running on a Mac while the script runs on Linux).

### 5) Operator-grade CLI & outputs

* **Rich UI**: human-scannable tables for fabric summary, per-switch status, BGP matrices, VXLAN/EVPN health, issues, and recommendations.
* **Artifacts** (saved alongside the script):

  * `fabric_data_<timestamp>.json` – raw device responses
  * `fabric_analysis_<provider>_<timestamp>.json` – AI results

---

## Project Structure

```
interface_aiv2.py        # main CLI tool
.env                     # (local) environment variables (ignored by git)
.venv/                   # virtualenv (ignored by git)
fabric_data_*.json       # raw collections (ignored by git)
fabric_analysis_*.json   # AI analyses (ignored by git)
```

---

## How It Works (Flow)

```
[Switch list] ──► [Concurrent XML-RPC collection] ──► fabric_data_*.json
                                     │
                                     ▼
                          [Per-switch AI summary]
                                     │
                                     ▼
                           [Fabric-level AI analysis]
                                     │
                                     ├─► fabric_analysis_*.json (persisted)
                                     └─► Rich tables in terminal
```

---

## Prerequisites

* Python 3.10+
* Juniper devices reachable on HTTP (default `:8080`) with XML-RPC enabled
* Packages (install in venv):

  ```bash
  pip install -U requests python-dotenv rich langchain-openai langchain-anthropic ollama pydantic
  ```
* API keys as needed:

  * `OPENAI_API_KEY` for OpenAI
  * `ANTHROPIC_API_KEY` for Claude
* **Ollama (optional):**

  * Local: `ollama serve`
  * Remote: expose with `OLLAMA_HOST=http://0.0.0.0:11434` on the host running Ollama
    Ensure the client machine can reach `http://<ollama-host>:11434/api/tags`

---

## Configuration

Edit the top of `interface_aiv2.py`:

```python
FABRIC_SWITCHES = ["172.29.129.189", ...]
SWITCH_PORT = 8080
USERNAME = "******"
PASSWORD = "********"
```

You can also place credentials in `.env` and load them into the script if you prefer (the script already calls `load_dotenv()`).

---

## Usage

### Quick start (Claude default)

```bash
python interface_aiv2.py
```

### OpenAI

```bash
python interface_aiv2.py --provider openai
python interface_aiv2.py --provider openai --model gpt-4o
```

### Ollama (local)

```bash
python interface_aiv2.py --provider ollama --model llama3.3:latest
python interface_aiv2.py --provider ollama --model llama3.3:latest --delay 0
```

### Ollama (remote host)

```bash
python interface_aiv2.py \
  --provider ollama \
  --ollama-host http://192.168.1.100:11434 \
  --model gpt-oss:120b \
  --delay 15
```

Or set the environment (recommended) and omit `--ollama-host`:

```bash
export OLLAMA_HOST=http://192.168.1.100:11434
python interface_aiv2.py --provider ollama --model gpt-oss:120b --delay 15
```

---

## Output Files

* **Raw collection:** `fabric_data_YYYYMMDD_HHMMSS.json`
  Device IPs, command names, and the (possibly truncated) XML bodies or errors.

* **AI analysis:** `fabric_analysis_<provider>_YYYYMMDD_HHMMSS.json`

  * `switch_summaries[<ip>].data` – structured per-switch summary.
  * `fabric_analysis` – fabric-level JSON: roles, topology, BGP/VXLAN/EVPN health, issues, recommendations.

---

## Error Handling & Resilience

* **HTTP/REST errors:** Captured per command with reason; the run continues.
* **Concurrency safety:** Thread pool with per-future error capture; fabric run continues.
* **LLM rate limits:** Exponential backoff (10s → 160s) on per-switch analysis retries.
* **Prompt safety:** Strict system prompts that require **JSON-only** outputs; code strips code-fences before parsing.
* **Token control:** XML payloads truncated to reduce context size and cost.
* **Ollama quirks handled:**

  * Differences in `Client.list()` return shapes (uses a defensive parser; HTTP fallback to `/api/tags`).
  * Pydantic `PrivateAttr` for the Ollama `Client` to avoid schema errors.
  * Remote host normalization and connection checks with clear error messages.

---

## Security Notes

* **Secrets:** `.env` is loaded but ignored by git; never commit API keys or device creds.
* **Scopes:** REST reads only; no config change operations in this tool.
* **Network exposure:** If using remote Ollama, ensure port `11434` is reachable only from trusted hosts (firewall).

---

## Extensibility

* **Add/Remove RPCs:** Update `COMMANDS_TO_EXECUTE` with new XML blocks.
* **Adjust roles/topology inference:** Modify the fabric analysis system prompt to reflect your conventions.
* **Custom outputs:** Write additional renderers (CSV/HTML) by reusing the parsed JSON.
* **Other model providers:** Swap or extend the `initialize_llm()` function.

---

## Troubleshooting

* **`Failed to connect to Ollama ...`**

  * Verify from the client: `curl http://<host>:11434/api/tags`
  * On host: `OLLAMA_HOST=http://0.0.0.0:11434 ollama serve`
  * Check macOS firewall allows inbound 11434.

* **`"OllamaWrapper" object has no field "client"`**

  * Ensure you’re on the version that uses `PrivateAttr` (`_client`) inside `OllamaWrapper`.

* **LLM returns non-JSON text**

  * The script strips code fences and attempts `json.loads`. If it still fails, reduce truncation thresholds or try a different model.

* **Large model stalls/timeouts**

  * Increase `--delay`, reduce the switch list, or switch to a smaller Ollama model (e.g., `llama3.3:latest`).

---

## Roadmap

* Optional Junos telemetry ingestion (gRPC/JSNAPy) for richer signals
* Graph export (topology DOT/PNG)
* Policy checks (e.g., “all leafs must peer with all spines”)
* Config drift detection & version correlation