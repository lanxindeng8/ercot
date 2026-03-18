#!/usr/bin/env python3
"""
CLI tool for TrueFlux API key management.

Usage:
    python -m prediction.scripts.manage_keys create --name "My App" --tier pro
    python -m prediction.scripts.manage_keys list
    python -m prediction.scripts.manage_keys revoke --id 3
    python -m prediction.scripts.manage_keys usage --id 1 --days 30
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prediction.src.auth.api_keys import APIKeyManager, TIER_LIMITS


def cmd_create(args, manager: APIKeyManager):
    raw_key = manager.create_key(args.name, args.tier)
    print(f"API key created successfully!")
    print(f"  Name:  {args.name}")
    print(f"  Tier:  {args.tier}")
    print(f"  Key:   {raw_key}")
    print()
    print("  *** Save this key — it cannot be retrieved later ***")


def cmd_list(args, manager: APIKeyManager):
    keys = manager.list_keys()
    if not keys:
        print("No API keys found.")
        return

    print(f"{'ID':<5} {'Prefix':<12} {'Name':<20} {'Tier':<12} {'Active':<8} {'Requests':<10} {'Last Used'}")
    print("-" * 90)
    for k in keys:
        print(
            f"{k.id:<5} {k.key_prefix:<12} {k.name:<20} {k.tier:<12} "
            f"{'yes' if k.active else 'no':<8} {k.request_count:<10} {k.last_used or 'never'}"
        )


def cmd_revoke(args, manager: APIKeyManager):
    if manager.revoke_key(args.id):
        print(f"Key {args.id} revoked.")
    else:
        print(f"Key {args.id} not found.", file=sys.stderr)
        sys.exit(1)


def cmd_usage(args, manager: APIKeyManager):
    usage = manager.get_usage(key_id=args.id, days=args.days)
    if not usage:
        print("No usage records found.")
        return

    print(f"Usage records (last {args.days} days): {len(usage)} requests")
    print(f"{'Endpoint':<40} {'Timestamp':<28} {'Status'}")
    print("-" * 80)
    for u in usage[:50]:  # Show first 50
        print(f"{u['endpoint']:<40} {u['timestamp']:<28} {u.get('status_code', '')}")
    if len(usage) > 50:
        print(f"... and {len(usage) - 50} more")


def main():
    parser = argparse.ArgumentParser(description="TrueFlux API Key Management")
    parser.add_argument("--db", type=str, default=None, help="Path to api_keys.db")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new API key")
    p_create.add_argument("--name", required=True, help="Key name/label")
    p_create.add_argument("--tier", default="free", choices=list(TIER_LIMITS.keys()), help="Access tier")

    sub.add_parser("list", help="List all API keys")

    p_revoke = sub.add_parser("revoke", help="Revoke an API key")
    p_revoke.add_argument("--id", type=int, required=True, help="Key ID to revoke")

    p_usage = sub.add_parser("usage", help="Show usage analytics")
    p_usage.add_argument("--id", type=int, default=None, help="Filter by key ID")
    p_usage.add_argument("--days", type=int, default=7, help="Days to look back")

    args = parser.parse_args()
    db_path = Path(args.db) if args.db else None
    manager = APIKeyManager(db_path)

    {"create": cmd_create, "list": cmd_list, "revoke": cmd_revoke, "usage": cmd_usage}[args.command](args, manager)


if __name__ == "__main__":
    main()
