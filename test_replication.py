#!/usr/bin/env python3
"""Test Replication functionality."""
import requests

BASE_URL = "http://localhost:3000"

def test_replication_status():
    print("=== Testing Replication Status ===\n")

    print("1. Getting replication status...")
    resp = requests.get(f"{BASE_URL}/replication/status")
    if resp.status_code == 200:
        status = resp.json()
        print(f"   Role: {status['role']}")
        print(f"   Current WAL Seq: {status['current_seq']}")
        print(f"   WAL Entries: {status['wal_entries']}")
        print(f"   WAL Size: {status['wal_size_bytes']} bytes")
    else:
        print(f"   Error: {resp.status_code}")

def test_wal_entries():
    print("\n=== Testing WAL Entries ===\n")

    print("1. Getting WAL entries (from seq 0)...")
    resp = requests.get(f"{BASE_URL}/replication/wal?from_seq=0&max_entries=10")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   Current Seq: {data['current_seq']}")
        print(f"   Has More: {data['has_more']}")
        print(f"   Entries: {len(data['entries'])}")
        for entry in data['entries'][:5]:
            print(f"     - Seq {entry['seq']}: {entry['operation_type']} on {entry['collection']}")
    else:
        print(f"   Error: {resp.status_code}")

def main():
    print("VectorDB Replication Test\n")
    print("=" * 50)

    test_replication_status()
    test_wal_entries()

    print("\n" + "=" * 50)
    print("\nNote: Replication features are available when server")
    print("is started with replication enabled.")
    print("\nReplication modes:")
    print("  - standalone: No replication (default)")
    print("  - primary: Accepts writes, maintains WAL")
    print("  - replica: Syncs from primary, read-only")

if __name__ == "__main__":
    main()
