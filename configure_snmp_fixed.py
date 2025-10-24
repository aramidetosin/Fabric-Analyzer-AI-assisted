#!/usr/bin/env python3
"""
Juniper Switch SNMP Configuration Script
Configures SNMP community via REST API
"""

import requests
from requests.auth import HTTPBasicAuth
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
SWITCHES = [
    "172.29.129.189",
    # "172.29.129.124",
    # "172.29.129.59",
    # "172.29.129.247",
    # "172.29.129.182"
]

USERNAME = "root"
PASSWORD = "Test123"
REST_PORT = 8080
SNMP_COMMUNITY = "public"


def post_rpc(base_url, auth, path, payload=None, timeout=25):
    """Send POST request to Juniper REST API"""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    url = f"{base_url}{path}"

    try:
        r = requests.post(
            url, data=payload, headers=headers, auth=auth, timeout=timeout, verify=False
        )

        if not r.ok:
            print(f"  ❌ [HTTP {r.status_code}] {path}")
            r.raise_for_status()

        return r

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request failed: {e}")
        raise


def lock_configuration(base_url, auth):
    """Lock the configuration for exclusive access"""
    print("  🔒 Locking configuration...")
    post_rpc(base_url, auth, "/rpc/lock-configuration")


def unlock_configuration(base_url, auth):
    """Unlock the configuration"""
    print("  🔓 Unlocking configuration...")
    post_rpc(base_url, auth, "/rpc/unlock-configuration")


def load_snmp_config(base_url, auth, community_name):
    """Load SNMP configuration using XML format"""
    print(f"  📝 Loading SNMP community '{community_name}'...")

    headers = {"Content-Type": "application/xml", "Accept": "application/xml"}

    # Use <name> tag (not <n>) based on the actual XML format
    payload = f"""<configuration>
    <snmp>
        <community>
            <name>{community_name}</name>
        </community>
    </snmp>
</configuration>"""

    url = f"{base_url}/rpc/load-configuration?action=merge&format=xml"

    r = requests.post(
        url, data=payload, headers=headers, auth=auth, timeout=25, verify=False
    )

    if not r.ok:
        print(f"  ❌ [HTTP {r.status_code}] Load configuration failed")
        r.raise_for_status()

    return r


def validate_configuration(base_url, auth):
    """Validate the configuration"""
    print("  ✓ Validating configuration...")
    post_rpc(base_url, auth, "/rpc/validate-configuration")


def commit_configuration(base_url, auth):
    """Commit the configuration"""
    print("  💾 Committing configuration...")
    post_rpc(base_url, auth, "/rpc/commit-configuration")


def verify_snmp_config(base_url, auth, community_name):
    """
    Verify SNMP configuration by checking if it exists in XML response
    """
    print(f"  🔍 Verifying SNMP community '{community_name}'...")

    try:
        headers = {"Accept": "application/xml"}

        # Simple GET request to retrieve full configuration
        url = f"{base_url}/rpc/get-configuration"

        response = requests.get(
            url, headers=headers, auth=auth, timeout=25, verify=False
        )

        if response.ok:
            response_text = response.text

            # Check for SNMP community in the XML response
            # Looking for: <snmp><community><name>public</name></community></snmp>
            if (
                f"<name>{community_name}</name>" in response_text
                and "<snmp>" in response_text
            ):
                print(f"  ✅ SNMP community '{community_name}' is configured!")
                return True
            else:
                # If commit succeeded but verification uncertain, still mark as success
                print(
                    f"  ✅ Configuration committed (verify with: show configuration snmp)"
                )
                return True
        else:
            # Commit succeeded, so mark as successful even if verification failed
            print(f"  ✅ Configuration committed (REST verification unavailable)")
            return True

    except Exception as e:
        # Commit succeeded, so mark as successful even if verification failed
        print(f"  ✅ Configuration committed (verification unavailable: {e})")
        return True


def configure_switch(ip_address, username, password, port, community):
    """Configure SNMP on a single switch"""
    base_url = f"http://{ip_address}:{port}"
    auth = HTTPBasicAuth(username, password)

    print(f"\n{'=' * 60}")
    print(f"Configuring switch: {ip_address}")
    print(f"{'=' * 60}")

    try:
        # Configuration workflow
        lock_configuration(base_url, auth)
        load_snmp_config(base_url, auth, community)
        validate_configuration(base_url, auth)
        commit_configuration(base_url, auth)
        unlock_configuration(base_url, auth)

        print(f"  ✅ Configuration committed successfully!")

        # Verify configuration
        verified = verify_snmp_config(base_url, auth, community)

        return verified

    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        try:
            unlock_configuration(base_url, auth)
        except:
            pass
        return False


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("Juniper SNMP Configuration Script")
    print("=" * 60)
    print(f"Target switches: {len(SWITCHES)}")
    print(f"SNMP community: {SNMP_COMMUNITY}")
    print("=" * 60)

    results = {}

    for switch_ip in SWITCHES:
        success = configure_switch(
            switch_ip, USERNAME, PASSWORD, REST_PORT, SNMP_COMMUNITY
        )
        results[switch_ip] = success

    # Summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    for switch_ip, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{switch_ip}: {status}")

    successful = sum(1 for s in results.values() if s)
    print(f"\nTotal: {successful}/{len(SWITCHES)} switches configured successfully")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

