#!/usr/bin/env python3
"""
Julie Voice Assistant - Main entry point.

Usage:
    python main.py          # Run CLI interface
    python main.py --help   # Show help
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Julie - French Insurance Voice Assistant"
    )
    parser.add_argument(
        "--interface",
        choices=["cli", "telephony"],
        default="cli",
        help="Interface to use (default: cli)"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version"
    )
    
    args = parser.parse_args()
    
    if args.version:
        from julie import __version__
        print(f"Julie v{__version__}")
        return
    
    if args.interface == "cli":
        from julie.interfaces.cli import run_cli
        run_cli()
    elif args.interface == "telephony":
        print("Telephony interface not yet implemented.")
        print("Coming soon: Asterisk integration")
        sys.exit(1)


if __name__ == "__main__":
    main()
