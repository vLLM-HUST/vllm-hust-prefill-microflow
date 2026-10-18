"""CPU-only deployment preparation and preflight. Never starts a worker."""
import argparse
import json

from .activation import cache_environment, preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("environment")
    prepare.add_argument("--cache-root", required=True)
    prepare.add_argument("--arm", choices=["off", "on"], required=True)
    sub.add_parser("check")
    args = parser.parse_args()
    result = cache_environment(args.cache_root, args.arm) if args.action == "environment" else preflight()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
