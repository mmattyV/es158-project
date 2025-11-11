"""
Dashboard Launcher

Simple script to launch either the matplotlib or web-based dashboard.
"""

import sys
import subprocess
import argparse


def run_matplotlib_dashboard():
    """Run the matplotlib-based dashboard."""
    print("🚀 Launching Matplotlib Dashboard...")
    print("Features: Real-time plots, interactive controls, grid topology")
    print("Close the window to exit.")
    
    try:
        from dashboard import main
        main()
    except ImportError as e:
        print(f"❌ Error importing dashboard: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")


def run_web_dashboard():
    """Run the Streamlit web dashboard."""
    print("🌐 Launching Web Dashboard...")
    print("Opening in your default browser...")
    print("Press Ctrl+C to stop the server.")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error running web dashboard: {e}")
        print("Make sure streamlit is installed: pip install streamlit")


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Power Grid Dashboard Launcher")
    parser.add_argument(
        "--type", "-t",
        choices=["matplotlib", "web", "both"],
        default="matplotlib",
        help="Type of dashboard to launch (default: matplotlib)"
    )
    
    args = parser.parse_args()
    
    print("⚡ Power Grid Multi-Agent RL Dashboard Launcher")
    print("=" * 50)
    
    if args.type == "matplotlib":
        run_matplotlib_dashboard()
    elif args.type == "web":
        run_web_dashboard()
    elif args.type == "both":
        print("🚀 Launching both dashboards...")
        print("Note: Close matplotlib window first, then stop web server with Ctrl+C")
        run_matplotlib_dashboard()
        run_web_dashboard()


if __name__ == "__main__":
    main()
