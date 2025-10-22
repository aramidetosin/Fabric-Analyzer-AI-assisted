#!/usr/bin/env python3
"""
Script to query Juniper fabric information via REST API
and use Claude AI, OpenAI, or Ollama via LangChain to analyze the entire fabric as a system
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.llms import LLM

# Load environment variables
load_dotenv()

FABRIC_SWITCHES = [
    "172.29.129.189",
    "172.29.129.124",
    "172.29.129.59",
    "172.29.129.247",
    "172.29.129.182",
]
SWITCH_PORT = 8080
USERNAME = "root"
PASSWORD = "Test123"

console = Console()

# Global LLM instance (will be initialized in main)
llm = None

# Commands to execute on each switch
COMMANDS_TO_EXECUTE = [
    {
        "name": "interfaces",
        "description": "Interface information",
        "xml": "<get-interface-information><detail/></get-interface-information>",
    },
    {
        "name": "bgp_summary",
        "description": "BGP summary",
        "xml": "<get-bgp-summary-information/>",
    },
    {
        "name": "route_summary",
        "description": "Route summary",
        "xml": "<get-route-summary-information/>",
    },
    {
        "name": "system_info",
        "description": "System information",
        "xml": "<get-system-information/>",
    },
    {
        "name": "vxlan_tunnels",
        "description": "VXLAN tunnel endpoints",
        "xml": "<get-vxlan-tunnel-end-point-information/>",
    },
    {
        "name": "evpn_database",
        "description": "EVPN database",
        "xml": "<get-evpn-database-information/>",
    },
    {
        "name": "lldp_neighbors",
        "description": "LLDP neighbors",
        "xml": "<get-lldp-neighbors-information/>",
    },
    {
        "name": "lacp_interfaces",
        "description": "LACP interfaces",
        "xml": "<get-lacp-interface-information/>",
    },
]

def _normalize_host(url: str) -> str:
    """Ensure Ollama host has a scheme and no trailing slash."""
    if not url:
        return "http://localhost:11434"
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url.rstrip("/")


from pydantic import PrivateAttr

class OllamaWrapper(LLM):
    """Wrapper for Ollama to work with LangChain interface using a bound Client."""
    model: str = "llama3.3:latest"
    host: str = "http://localhost:11434"
    temperature: float = 0.0

    # Pydantic-safe private attribute for the client
    _client: object = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from ollama import Client  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. Install with: pip install -U ollama"
            ) from e
        # Bind client to remote host (supports different machine)
        self._client = Client(host=self.host)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop=None, **kwargs):
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def invoke(self, messages, **kwargs):
        """Convert LangChain messages format to a plain prompt and call Ollama."""
        prompt_parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                if msg.__class__.__name__ == "SystemMessage":
                    prompt_parts.append(f"System: {msg.content}")
                elif msg.__class__.__name__ == "HumanMessage":
                    prompt_parts.append(f"Human: {msg.content}")
        full_prompt = "\n\n".join(prompt_parts)
        content = self._call(full_prompt)
        class Response:
            def __init__(self, content):
                self.content = content
        return Response(content)


def _safe_collect_model_names_from_client_list(resp_obj) -> List[str]:
    """
    Handle various return shapes from ollama.Client.list() across versions.
    Prefer 'name', fallback to 'model'.
    """
    models_field = []
    if isinstance(resp_obj, dict):
        models_field = resp_obj.get("models", [])
    elif isinstance(resp_obj, list):
        models_field = resp_obj
    else:
        # Unknown shape; best effort
        models_field = getattr(resp_obj, "models", []) or []

    names: List[str] = []
    for m in models_field:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
        else:
            # object-like
            name = getattr(m, "name", None) or getattr(m, "model", None)
        if name:
            names.append(name)
    return names


def _fallback_http_list_models(host: str) -> List[str]:
    """Fallback to direct HTTP /api/tags to list models."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        payload = r.json()
        names = []
        for m in payload.get("models", []):
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
                if name:
                    names.append(name)
        return names
    except Exception:
        return []


def initialize_llm(provider: str = "claude", model: Optional[str] = None, ollama_host: Optional[str] = None):
    """
    Initialize the LLM provider.

    Args:
        provider: Either 'claude', 'openai', or 'ollama'
        model: Specific model name (optional)
        ollama_host: Ollama server host (e.g., 'http://192.168.1.100:11434')
    """
    global llm

    if provider.lower() == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red bold]ERROR: ANTHROPIC_API_KEY environment variable not set![/red bold]")
            console.print("[yellow]Please set it with: export ANTHROPIC_API_KEY='your-api-key'[/yellow]")
            return None

        model_name = model or "claude-sonnet-4-20250514"
        console.print(f"[dim]Using Claude model: {model_name}[/dim]")
        llm = ChatAnthropic(model=model_name, api_key=api_key, temperature=0)

    elif provider.lower() == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            console.print("[red bold]ERROR: OPENAI_API_KEY environment variable not set![/red bold]")
            console.print("[yellow]Please set it with: export OPENAI_API_KEY='your-api-key'[/yellow]")
            return None

        model_name = model or "gpt-4o"
        console.print(f"[dim]Using OpenAI model: {model_name}[/dim]")
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

    elif provider.lower() == "ollama":
        try:
            from ollama import Client  # type: ignore
        except ImportError:
            console.print("[red]ERROR: ollama package not installed![/red]")
            console.print("[yellow]Install it with: pip install -U ollama[/yellow]")
            return None

        host_env = os.environ.get("OLLAMA_HOST")
        host = _normalize_host(ollama_host or host_env or "http://localhost:11434")
        model_name = model or "llama3.3:latest"

        console.print(f"[dim]Using Ollama model: {model_name}[/dim]")
        console.print(f"[dim]Ollama host: {host}[/dim]")

        try:
            client = Client(host=host)

            # Try client.list() first, defensively parse names
            available_models = []
            try:
                resp = client.list()
                available_models = _safe_collect_model_names_from_client_list(resp)
            except Exception as e_list:
                # Fallback to HTTP endpoint
                available_models = _fallback_http_list_models(host)
                if not available_models:
                    raise e_list

            console.print(f"[green]✓ Connected to Ollama[/green]")
            if available_models:
                console.print(f"[dim]Available models: {', '.join(available_models)}[/dim]")
            else:
                console.print("[yellow]No models reported by Ollama at this host[/yellow]")

            if model_name not in available_models:
                console.print(f"[yellow]Warning: Model '{model_name}' not found in Ollama[/yellow]")
                if available_models:
                    console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
                return None

            llm = OllamaWrapper(model=model_name, host=host, temperature=0.0)

        except Exception as e:
            console.print(f"[red]Failed to connect to Ollama at {host}[/red]")
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Checks:[/yellow]")
            console.print("  • macOS firewall allows inbound to 11434")
            console.print("  • On Mac: OLLAMA_HOST is set, e.g. 'http://0.0.0.0:11434'")
            console.print("  • From Linux: curl http://<mac-ip>:11434/api/tags works")
            return None

    else:
        console.print(f"[red]Unknown provider: {provider}. Use 'claude', 'openai', or 'ollama'[/red]")
        return None

    return llm


def execute_rpc_command(
    switch_ip: str,
    command: Dict,
    username: str = USERNAME,
    password: str = PASSWORD,
    port: int = SWITCH_PORT,
) -> Dict:
    """Execute a single RPC command on a switch."""
    url = f"http://{switch_ip}:{port}/rpc?stop-on-error=1"
    headers = {
        "Content-Type": "application/xml",
        "Accept": "application/xml",
    }

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(username, password),
            headers=headers,
            data=command["xml"],
            timeout=30,
        )
        response.raise_for_status()
        return {
            "success": True,
            "command": command["name"],
            "data": response.text,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "command": command["name"],
            "error": str(e),
        }


def collect_switch_data(switch_ip: str) -> Dict:
    """Collect all data from a single switch."""
    console.print(f"[cyan]Collecting data from {switch_ip}...[/cyan]")

    switch_data = {
        "ip": switch_ip,
        "timestamp": datetime.now().isoformat(),
        "commands": {},
    }

    for command in COMMANDS_TO_EXECUTE:
        result = execute_rpc_command(switch_ip, command)
        switch_data["commands"][command["name"]] = result

        if result["success"]:
            console.print(f"  [green]✓[/green] {command['description']}")
        else:
            console.print(
                f"  [red]✗[/red] {command['description']}: {result.get('error', 'Unknown error')}"
            )

    return switch_data


def collect_fabric_data(switches: List[str]) -> Dict:
    """Collect data from all switches in the fabric concurrently."""
    fabric_data = {
        "collection_timestamp": datetime.now().isoformat(),
        "switches": {},
    }

    console.print(
        f"\n[bold]Collecting data from {len(switches)} switches in fabric...[/bold]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Collecting fabric data...", total=len(switches))

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_switch = {
                executor.submit(collect_switch_data, switch_ip): switch_ip
                for switch_ip in switches
            }

            for future in as_completed(future_to_switch):
                switch_ip = future_to_switch[future]
                try:
                    switch_data = future.result()
                    fabric_data["switches"][switch_ip] = switch_data
                except Exception as e:
                    console.print(
                        f"[red]Error collecting data from {switch_ip}: {e}[/red]"
                    )
                    fabric_data["switches"][switch_ip] = {
                        "ip": switch_ip,
                        "error": str(e),
                    }
                finally:
                    progress.update(task, advance=1)

    return fabric_data


def analyze_single_switch(
    switch_ip: str, switch_data: Dict, retry_count: int = 0
) -> Dict:
    """Analyze a single switch and extract key information."""

    system_prompt = """You are a network engineer analyzing a single Juniper switch.
Extract and summarize key information from the provided JSON-wrapped XML data.

Return ONLY a JSON object with this structure:
{
  "switch_ip": "string",
  "hostname": "string or null",
  "model": "string or null",
  "software_version": "string or null",
  "role": "spine|leaf|border|unknown",
  "interfaces": {
    "total": number,
    "up": number,
    "down": number,
    "with_errors": [{"name": "string", "input_errors": number, "output_errors": number}]
  },
  "bgp": {
    "peers": [{"peer_ip": "string", "state": "string", "received_routes": number}],
    "total_peers": number,
    "established_peers": number
  },
  "vxlan": {
    "vtep_count": number,
    "active_tunnels": number,
    "tunnel_endpoints": ["list of IPs"]
  },
  "evpn": {
    "total_mac_entries": number,
    "total_mac_ip_entries": number,
    "vni_list": [numbers]
  },
  "lldp_neighbors": [
    {"local_interface": "string", "remote_system": "string", "remote_interface": "string"}
  ],
  "routes": {
    "total": number,
    "active": number,
    "by_protocol": {"bgp": number, "direct": number, "static": number}
  },
  "issues": [
    {"severity": "critical|warning|info", "description": "string"}
  ]
}"""

    # Truncate XML data to reduce token count
    truncated_data = {
        "ip": switch_data.get("ip"),
        "timestamp": switch_data.get("timestamp"),
        "commands": {}
    }

    for cmd_name, cmd_result in switch_data.get("commands", {}).items():
        if cmd_result.get("success"):
            xml_data = cmd_result.get("data", "")
            if len(xml_data) > 10000:
                truncated_data["commands"][cmd_name] = {
                    "success": True,
                    "command": cmd_name,
                    "data": xml_data[:10000] + "\n... [truncated for token limit]"
                }
            else:
                truncated_data["commands"][cmd_name] = cmd_result
        else:
            truncated_data["commands"][cmd_name] = cmd_result

    user_prompt = f"""Analyze this Juniper switch data and extract the key information:

{json.dumps(truncated_data, indent=2)}"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        response_text = response.content.strip()

        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed_data = json.loads(response_text)
        return {"success": True, "data": parsed_data}

    except Exception as e:
        error_str = str(e)
        if "rate_limit" in error_str.lower() and retry_count < 5:
            wait_time = (2**retry_count) * 10  # 10, 20, 40, 80, 160 seconds
            console.print(f"  [yellow]⏱  Rate limit hit, waiting {wait_time} seconds before retry...[/yellow]")
            time.sleep(wait_time)
            return analyze_single_switch(switch_ip, switch_data, retry_count + 1)

        console.print(f"[red]Error analyzing {switch_ip}: {e}[/red]")
        return {"success": False, "error": str(e)}


def analyze_switches_individually(fabric_data: Dict, delay: int = 3) -> Dict:
    """Analyze each switch individually to create summaries."""
    console.print("\n[bold cyan]Step 1: Analyzing individual switches...[/bold cyan]")

    switch_summaries = {}

    for idx, (switch_ip, switch_data) in enumerate(fabric_data["switches"].items()):
        if "error" in switch_data:
            switch_summaries[switch_ip] = {
                "success": False,
                "error": switch_data["error"],
            }
            continue

        console.print(f"  [cyan]→ Analyzing {switch_ip}...[/cyan]")
        summary = analyze_single_switch(switch_ip, switch_data)
        switch_summaries[switch_ip] = summary

        if idx < len(fabric_data["switches"]) - 1 and summary.get("success") and delay > 0:
            console.print(f"  [dim]⏱  Waiting {delay} seconds to avoid rate limits...[/dim]")
            time.sleep(delay)

    return switch_summaries


def analyze_fabric_topology(switch_summaries: Dict) -> Dict:
    """Analyze the fabric topology and relationships from switch summaries."""
    console.print("\n[bold cyan]Step 2: Analyzing fabric topology and health...[/bold cyan]")

    system_prompt = """You are a network architect analyzing a Juniper EVPN-VXLAN fabric.
You have been provided with summaries of each switch in the fabric.

Analyze the fabric as a complete system and provide:

1. Topology understanding (spine-leaf, full-mesh, etc.)
2. Inter-switch relationships (from LLDP data)
3. BGP peer relationships and health
4. VXLAN tunnel mesh status
5. EVPN consistency across fabric
6. Overall fabric health and issues
7. Actionable recommendations

Return ONLY a JSON object with this structure:
{
  "fabric_summary": {
    "total_switches": number,
    "topology_type": "spine-leaf|full-mesh|other",
    "switch_roles": {
      "spine": ["ips"],
      "leaf": ["ips"],
      "border": ["ips"],
      "unknown": ["ips"]
    },
    "software_versions": {"version": ["switch_ips"]},
    "health_score": number,
    "health_description": "string"
  },
  "topology": {
    "connections": [
      {
        "from_switch": "ip",
        "from_interface": "name",
        "to_switch": "ip or hostname",
        "to_interface": "name"
      }
    ],
    "spine_leaf_pairs": [
      {"spine": "ip", "leafs": ["ips"]}
    ]
  },
  "bgp_fabric": {
    "total_sessions": number,
    "established_sessions": number,
    "failed_sessions": [
      {"switch": "ip", "peer": "ip", "state": "string"}
    ],
    "underlay_peers": number,
    "overlay_peers": number
  },
  "vxlan_fabric": {
    "total_vteps": number,
    "expected_tunnels": number,
    "actual_tunnels": number,
    "tunnel_matrix": {
      "switch_ip": ["connected_to_ips"]
    },
    "missing_tunnels": [
      {"from": "ip", "to": "ip"}
    ]
  },
  "evpn_fabric": {
    "total_vnis": [numbers],
    "vni_consistency": "consistent|inconsistent",
    "total_mac_routes": number,
    "mac_route_distribution": {"switch_ip": count}
  },
  "fabric_issues": [
    {
      "severity": "critical|warning|info",
      "category": "bgp|vxlan|evpn|interface|configuration",
      "switch": "ip or all",
      "description": "string",
      "impact": "string",
      "recommendation": "string"
    }
  ],
  "recommendations": [
    {
      "priority": "high|medium|low",
      "category": "string",
      "description": "string",
      "expected_benefit": "string"
    }
  ]
}"""

    user_prompt = f"""Analyze these switch summaries and provide comprehensive fabric analysis:

{json.dumps(switch_summaries, indent=2)}

Focus on:
- Inter-switch connectivity and topology
- BGP session health across fabric
- VXLAN tunnel mesh completeness
- EVPN database consistency
- Configuration consistency
- Any anomalies or issues"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        console.print("[dim]Performing fabric-level analysis...[/dim]")
        response = llm.invoke(messages)
        response_text = response.content.strip()

        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed_analysis = json.loads(response_text)
        console.print("[green]✓ Fabric analysis complete![/green]")

        return {
            "success": True,
            "switch_summaries": switch_summaries,
            "fabric_analysis": parsed_analysis,
        }

    except Exception as e:
        console.print(f"[red]Error during fabric analysis: {e}[/red]")
        return {"success": False, "error": str(e), "switch_summaries": switch_summaries}


def display_fabric_analysis(analysis: Dict) -> None:
    """Display the fabric analysis in formatted tables."""
    if not analysis.get("success"):
        console.print(f"[red]Analysis failed: {analysis.get('error')}[/red]")
        return

    data = analysis["fabric_analysis"]

    # Fabric Summary
    console.print("\n")
    summary_table = Table(
        title="Fabric Summary",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold cyan",
    )

    summary_table.add_column("Metric", style="cyan", width=25)
    summary_table.add_column("Value", style="yellow")

    fabric_sum = data.get("fabric_summary", {})
    health_score = fabric_sum.get("health_score", 0)
    health_color = "green" if health_score >= 80 else "yellow" if health_score >= 60 else "red"

    summary_table.add_row("Total Switches", str(fabric_sum.get("total_switches", 0)))
    summary_table.add_row("Topology Type", fabric_sum.get("topology_type", "Unknown"))
    summary_table.add_row("Health Score", f"[{health_color}]{health_score}/100[/{health_color}]")
    summary_table.add_row("Health Status", fabric_sum.get("health_description", "Unknown"))

    roles = fabric_sum.get("switch_roles", {})
    for role, switches in roles.items():
        if switches:
            summary_table.add_row(f"{role.capitalize()} Switches", ", ".join(switches))

    console.print(summary_table)

    # Individual Switch Summary
    console.print("\n")
    switch_table = Table(
        title="Switch Details",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    switch_table.add_column("IP", style="cyan", width=15)
    switch_table.add_column("Hostname", width=15)
    switch_table.add_column("Role", width=10)
    switch_table.add_column("Version", width=15)
    switch_table.add_column("Interfaces", justify="center", width=12)
    switch_table.add_column("BGP Peers", justify="center", width=12)
    switch_table.add_column("VXLAN", justify="center", width=10)

    switch_summaries = analysis.get("switch_summaries", {})
    for switch_ip, summary in switch_summaries.items():
        if not summary.get("success"):
            switch_table.add_row(switch_ip, "[red]Error[/red]", "-", "-", "-", "-", "-")
            continue

        switch_data = summary.get("data", {})
        iface_data = switch_data.get("interfaces", {})
        bgp_data = switch_data.get("bgp", {})
        vxlan_data = switch_data.get("vxlan", {})

        iface_status = f"{iface_data.get('up', 0)}/{iface_data.get('total', 0)}"
        bgp_status = f"{bgp_data.get('established_peers', 0)}/{bgp_data.get('total_peers', 0)}"
        bgp_color = "green" if bgp_data.get("established_peers", 0) == bgp_data.get("total_peers", 0) else "red"

        switch_table.add_row(
            switch_ip,
            (switch_data.get("hostname", "N/A") or "N/A")[:15],
            switch_data.get("role", "unknown"),
            (switch_data.get("software_version", "N/A") or "N/A")[:15],
            iface_status,
            f"[{bgp_color}]{bgp_status}[/{bgp_color}]",
            str(vxlan_data.get("active_tunnels", 0)),
        )

    console.print(switch_table)

    # BGP Per-Switch Details
    console.print("\n")
    bgp_detail_table = Table(
        title="BGP Sessions Per Switch",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )

    bgp_detail_table.add_column("Switch", style="cyan", width=15)
    bgp_detail_table.add_column("Hostname", width=15)
    bgp_detail_table.add_column("Role", width=10)
    bgp_detail_table.add_column("Total", justify="center", width=8)
    bgp_detail_table.add_column("Established", justify="center", width=12)
    bgp_detail_table.add_column("Down", justify="center", width=8)
    bgp_detail_table.add_column("Status", justify="center", width=10)

    for switch_ip, summary in switch_summaries.items():
        if not summary.get("success"):
            bgp_detail_table.add_row(
                switch_ip, "[red]Error[/red]", "-", "-", "-", "-", "[red]ERROR[/red]"
            )
            continue

        switch_data = summary.get("data", {})
        bgp_data = switch_data.get("bgp", {})

        total_peers = bgp_data.get("total_peers", 0)
        established = bgp_data.get("established_peers", 0)
        down = total_peers - established

        if established == total_peers and total_peers > 0:
            status_text = "[green]✓ All Up[/green]"
            established_text = f"[green]{established}[/green]"
            down_text = str(down)
        elif established == 0 and total_peers > 0:
            status_text = "[red]✗ All Down[/red]"
            established_text = f"[red]{established}[/red]"
            down_text = f"[red]{down}[/red]"
        else:
            status_text = "[yellow]⚠ Partial[/yellow]"
            established_text = f"[yellow]{established}[/yellow]"
            down_text = f"[red]{down}[/red]"

        bgp_detail_table.add_row(
            switch_ip,
            (switch_data.get("hostname", "N/A") or "N/A")[:15],
            switch_data.get("role", "unknown"),
            str(total_peers),
            established_text,
            down_text,
            status_text,
        )

    console.print(bgp_detail_table)

    # BGP Fabric Summary
    console.print("\n")
    bgp_table = Table(
        title="BGP Fabric Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )

    bgp_table.add_column("Metric", style="cyan", width=25)
    bgp_table.add_column("Value", style="yellow")

    bgp_fabric = data.get("bgp_fabric", {})
    total_sessions = bgp_fabric.get("total_sessions", 0)
    established = bgp_fabric.get("established_sessions", 0)
    bgp_health_color = "green" if established == total_sessions else "red"

    bgp_table.add_row("Total BGP Sessions", str(total_sessions))
    bgp_table.add_row("Established Sessions", f"[{bgp_health_color}]{established}[/{bgp_health_color}]")
    bgp_table.add_row("Underlay Peers", str(bgp_fabric.get("underlay_peers", 0)))
    bgp_table.add_row("Overlay Peers", str(bgp_fabric.get("overlay_peers", 0)))

    failed_sessions = bgp_fabric.get("failed_sessions", [])
    if failed_sessions:
        bgp_table.add_row("Failed Sessions", f"[red]{len(failed_sessions)}[/red]")

    console.print(bgp_table)

    # VXLAN Fabric Status
    console.print("\n")
    vxlan_table = Table(
        title="VXLAN Fabric Status",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green",
    )

    vxlan_table.add_column("Metric", style="cyan", width=25)
    vxlan_table.add_column("Value", style="yellow")

    vxlan_fabric = data.get("vxlan_fabric", {})
    expected_tunnels = vxlan_fabric.get("expected_tunnels", 0)
    actual_tunnels = vxlan_fabric.get("actual_tunnels", 0)
    tunnel_health_color = "green" if actual_tunnels == expected_tunnels else "red"

    vxlan_table.add_row("Total VTEPs", str(vxlan_fabric.get("total_vteps", 0)))
    vxlan_table.add_row("Expected Tunnels", str(expected_tunnels))
    vxlan_table.add_row("Actual Tunnels", f"[{tunnel_health_color}]{actual_tunnels}[/{tunnel_health_color}]")

    missing_tunnels = vxlan_fabric.get("missing_tunnels", [])
    if missing_tunnels:
        vxlan_table.add_row("Missing Tunnels", f"[red]{len(missing_tunnels)}[/red]")

    console.print(vxlan_table)

    # EVPN Status
    console.print("\n")
    evpn_table = Table(
        title="EVPN Fabric Status",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    evpn_table.add_column("Metric", style="cyan", width=25)
    evpn_table.add_column("Value", style="yellow")

    evpn_fabric = data.get("evpn_fabric", {})
    vni_consistency = evpn_fabric.get("vni_consistency", "unknown")
    vni_color = "green" if vni_consistency == "consistent" else "red"

    vnis = evpn_fabric.get("total_vnis", [])
    evpn_table.add_row("Total VNIs", str(len(vnis)))
    evpn_table.add_row("VNI List", ", ".join(map(str, vnis[:10])))
    evpn_table.add_row("VNI Consistency", f"[{vni_color}]{vni_consistency}[/{vni_color}]")
    evpn_table.add_row("Total MAC Routes", str(evpn_fabric.get("total_mac_routes", 0)))

    console.print(evpn_table)

    # Issues Table
    issues = data.get("fabric_issues", [])
    if issues:
        console.print("\n")
        issues_table = Table(
            title="⚠️  Fabric Issues",
            box=box.HEAVY,
            show_header=True,
            header_style="bold red",
        )

        issues_table.add_column("Severity", style="bold", width=10)
        issues_table.add_column("Category", width=12)
        issues_table.add_column("Switch", width=15)
        issues_table.add_column("Description", width=35)
        issues_table.add_column("Recommendation", width=35)

        for issue in issues[:15]:
            severity = issue.get("severity", "info")
            sev_color = (
                "red" if severity == "critical"
                else "yellow" if severity == "warning"
                else "blue"
            )
            issues_table.add_row(
                f"[{sev_color}]{severity.upper()}[/{sev_color}]",
                issue.get("category", "N/A"),
                issue.get("switch", "N/A"),
                issue.get("description", "N/A"),
                issue.get("recommendation", "N/A"),
            )

        console.print(issues_table)

        if len(issues) > 15:
            console.print(f"[dim]... and {len(issues) - 15} more issues[/dim]")

    # Recommendations
    recommendations = data.get("recommendations", [])
    if recommendations:
        console.print("\n")
        rec_table = Table(
            title="💡 Recommendations",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
        )

        rec_table.add_column("Priority", style="bold", width=10)
        rec_table.add_column("Category", width=15)
        rec_table.add_column("Description", width=45)
        rec_table.add_column("Expected Benefit", width=35)

        for rec in recommendations[:8]:
            priority = rec.get("priority", "low")
            pri_color = "red" if priority == "high" else "yellow" if priority == "medium" else "blue"

            rec_table.add_row(
                f"[{pri_color}]{priority.upper()}[/{pri_color}]",
                rec.get("category", "N/A"),
                rec.get("description", "N/A"),
                rec.get("expected_benefit", "N/A"),
            )

        console.print(rec_table)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Juniper Fabric Analysis Tool with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # ── Claude (default provider) ────────────────────────────────────────────────
  # Uses ANTHROPIC_API_KEY from your environment
  python interface_aiv2.py

  # Use a specific Claude model
  python interface_aiv2.py --provider claude --model claude-3-7-sonnet-20250219

  # ── OpenAI ───────────────────────────────────────────────────────────────────
  # Uses OPENAI_API_KEY from your environment
  python interface_aiv2.py --provider openai

  # OpenAI with an explicit model
  python interface_aiv2.py --provider openai --model gpt-5

  # ── Ollama (local machine) ───────────────────────────────────────────────────
  # Fast local testing with llama3.3 (assumes Ollama on localhost:11434)
  python interface_aiv2.py --provider ollama --model llama3.3:latest

  # Local Ollama with no delay between calls (no rate limits locally)
  python interface_aiv2.py --provider ollama --model llama3.3:latest --delay 0

  # ── Ollama (remote host) ─────────────────────────────────────────────────────
  # Read remote host from environment (recommended)
  #  Bash/Zsh:  export OLLAMA_HOST=http://192.168.1.100:11434
  #  fish:      set -gx OLLAMA_HOST http://192.168.1.100:11434

  # If your Mac’s Ollama listens on all interfaces (good for remote access):
  #  Bash/Zsh:  export OLLAMA_HOST=http://0.0.0.0:11434
  #  fish:      set -gx OLLAMA_HOST http://0.0.0.0:11434

  # Remote Ollama on your Mac at 192.168.1.100, using a large model
  python interface_aiv2.py --provider ollama --ollama-host http://192.168.1.100:11434 --model gpt-oss:120b --delay 15

  python interface_aiv2.py --provider ollama --model gpt-oss:120b --delay 15

  # ── Mixed tips ───────────────────────────────────────────────────────────────
  # Reduce token pressure when using huge models (keep some delay for remote hosts)
  python interface_aiv2.py --provider ollama --model gpt-oss:120b --delay 10

  # Quick switch between providers with explicit models
  python interface_aiv2.py --provider claude --model claude-sonnet-4-20250514 --delay 15
  python interface_aiv2.py --provider openai --model gpt-5 --delay 15
  python interface_aiv2.py --provider ollama --model llama3.3:latest --delay 0

        """,
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["claude", "openai", "ollama"],
        default="claude",
        help="AI provider to use (default: claude)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name to use (optional)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama server host URL (default: http://localhost:11434 or OLLAMA_HOST env var)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Delay in seconds between switch analyses to avoid rate limits (default: 3, use 0 for Ollama)",
    )

    args = parser.parse_args()

    console.print("[bold green]Juniper Fabric Analysis Tool with AI[/bold green]")
    console.print(f"[dim]Analyzing {len(FABRIC_SWITCHES)} switches in fabric[/dim]")
    console.print(f"[dim]AI Provider: {args.provider.upper()}[/dim]\n")

    # Initialize LLM
    if not initialize_llm(args.provider, args.model, args.ollama_host):
        return

    # Step 1: Collect data from all switches
    fabric_data = collect_fabric_data(FABRIC_SWITCHES)

    # Save raw fabric data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = f"fabric_data_{timestamp}.json"
    console.print(f"\n[dim]Saving raw fabric data to {raw_filename}...[/dim]")
    with open(raw_filename, "w") as f:
        json.dump(fabric_data, f, indent=2)
    console.print("[green]✓ Raw data saved![/green]")

    # Step 2: Analyze each switch individually
    switch_summaries = analyze_switches_individually(fabric_data, args.delay)

    # Step 3: Analyze fabric topology
    fabric_analysis = analyze_fabric_topology(switch_summaries)

    # Save analysis
    analysis_filename = f"fabric_analysis_{args.provider}_{timestamp}.json"
    console.print(f"\n[dim]Saving analysis to {analysis_filename}...[/dim]")
    with open(analysis_filename, "w") as f:
        json.dump(fabric_analysis, f, indent=2)
    console.print("[green]✓ Analysis saved![/green]")

    # Step 4: Display results
    console.print("\n[bold]Fabric Analysis Results:[/bold]")
    display_fabric_analysis(fabric_analysis)

    console.print(f"\n[bold green]✓ Analysis complete![/bold green]")
    console.print(f"[dim]AI Provider: {args.provider.upper()}[/dim]")
    console.print(f"[dim]Raw data: {raw_filename}[/dim]")
    console.print(f"[dim]Analysis: {analysis_filename}[/dim]")


if __name__ == "__main__":
    main()
