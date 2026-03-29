import subprocess
import sys
from pathlib import Path


def main():
    if not Path('config.yaml').exists():
        print('ERROR: config.yaml not found.')
        print('Make sure you are running from inside the neuropose folder.')
        sys.exit(1)

    print()
    print('=' * 50)
    print('  NeuroPose — Attention Monitoring System')
    print('=' * 50)
    print()
    print('Starting dashboard...')
    print('Open your browser at: http://localhost:8501')
    print('Press Ctrl+C in this terminal to stop.')
    print()

    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        'ui/dashboard.py',
        '--server.headless',          'false',
        '--browser.gatherUsageStats', 'false',
        '--server.port',              '8501',
    ])


if __name__ == '__main__':
    main()