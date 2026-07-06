#!/usr/bin/env python3
"""
Simple runner to execute project scripts in sequence.

Usage:
  python runner.py [--continue-on-error]

You can also override the script list with `--scripts script1.py script2.py`.
"""
from pathlib import Path
import subprocess
import sys
import argparse
import logging


def run_script(path: Path) -> int:
    return subprocess.run([sys.executable, str(path)]).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project scripts sequentially")
    parser.add_argument('--continue-on-error', action='store_true', help='Continue running remaining scripts if one fails')
    parser.add_argument('--scripts', nargs='*', help='Override default script list (paths relative to runner)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    default_scripts = [
        'data-processing.py',
        'conversation-summarizer.py',
        'sentiment-and-topic-modelling-tuned.py',
    ]

    scripts = args.scripts if args.scripts else default_scripts

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    for script in scripts:
        script_path = (script_dir / script).resolve()
        logging.info('Running %s', script_path.name)
        if not script_path.exists():
            logging.error('Not found: %s', script_path)
            if not args.continue_on_error:
                sys.exit(2)
            else:
                continue

        rc = run_script(script_path)
        if rc != 0:
            logging.error('%s exited with code %d', script_path.name, rc)
            if not args.continue_on_error:
                sys.exit(rc)
            else:
                logging.info('Continuing despite error as requested')

    logging.info('Runner finished')


if __name__ == '__main__':
    main()
