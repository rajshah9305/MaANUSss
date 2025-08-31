Manus AI - Autonomous Agent System
Manus AI is a sophisticated autonomous agent system designed for macOS on Apple Silicon. It combines local orchestration with the power of cloud AI APIs (OpenAI, Anthropic, Gemini) to automate computer tasks, analyze on-screen content, and boost productivity.
Table of Contents
 * Features
 * System Architecture
 * Prerequisites
 * Setup and Installation
 * Configuration
 * Running Manus AI
 * Usage
   * Interactive CLI
   * Web Dashboard
 * Safety Features
 * Troubleshooting
Features
 * Multi-Agent Framework: Specialized agents for computer control, vision analysis, and productivity.
 * Cloud AI Integration: Unified client for OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), and Google (Gemini). Includes provider fallback.
 * macOS Native Integration: Optimized for Apple Silicon, using native screencapture and system utilities for efficiency.
 * Async Architecture: Built with asyncio for high-performance, non-blocking task processing.
 * Real-time Monitoring: A web-based dashboard provides live insights into tasks, agents, and system resources.
 * Interactive CLI: Control the system, submit tasks, and view logs directly from your terminal.
 * Safety First: A simulation mode is enabled by default to prevent unintended actions. Actions are logged for review before execution.
 * Voice Control (Experimental): Basic voice command processing using your microphone.
System Architecture
Manus AI runs a local Python application that orchestrates tasks.
 * Input: Tasks are submitted via the CLI, Web UI, or voice commands.
 * Task Orchestrator: Manages a priority queue, routing tasks to the appropriate agent.
 * Agents:
   * ComputerUseAgent: Interacts with the GUI (clicks, types, etc.).
   * VisionAnalyzerAgent: Understands the content of the screen.
   * ProductivityAgent: Handles tasks like email drafting and scheduling.
 * AI Client: A unified interface that sends requests to cloud AI models for reasoning, planning, and vision analysis.
 * System Tools: Interfaces for screen capture, system monitoring, and executing actions.
Prerequisites
 * Hardware: MacBook with Apple Silicon (M1, M2, M3, etc.).
 * OS: macOS 12.0 Monterey or newer.
 * Python: Python 3.11 or newer.
 * Homebrew: The missing package manager for macOS.
Setup and Installation
1. Grant Terminal Permissions
Manus AI requires Accessibility and Screen Recording permissions to see your screen and control your computer.
IMPORTANT: You must grant these permissions to the terminal application you will use to run the script.
 * Open System Settings.
 * Go to Privacy & Security.
 * Find and open Accessibility.
 * Click the + button, find your terminal app (e.g., Terminal.app, iTerm.app), and add it to the list. Ensure the toggle is ON.
 * Go back to Privacy & Security and open Screen Recording.
 * Do the same: click +, add your terminal app, and ensure the toggle is ON.
2. Install System Dependencies
Some Python libraries require system-level packages. Install them using Homebrew.
brew install portaudio

3. Clone the Repository & Set Up Environment
# It's assumed you have saved the provided files into a directory named "manus-ai"
cd manus-ai

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

Configuration
Manus AI is configured using a .env file.
 * Rename the provided .env.template file to .env:
   # (Assuming you named the provided .env file as .env.template)
mv .env.template .env

 * Open the .env file and add your API keys from OpenAI, Anthropic, and/or Google AI.
   # .env file

# --- API Keys (at least one is required) ---
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="AIza..."

# --- System Configuration ---
# Set to "false" to allow the agent to perform real clicks and key presses.
# DANGER: Disabling simulation mode can lead to unexpected behavior.
SIMULATION_MODE="true"

# --- Web Dashboard Configuration ---
WEB_INTERFACE_HOST="127.0.0.1"
WEB_INTERFACE_PORT="8000"

Running Manus AI
To start the system, simply run the main Python script from the root of the project directory:
python manus_ai_standalone.py

The system will initialize, perform startup checks, and present you with the interactive command-line interface (CLI). The web dashboard will also be available at http://127.0.0.1:8000.
Usage
Interactive CLI
The CLI is the primary way to interact with Manus AI. Type help to see a list of available commands.
 * task <description>: Create a new task for the system to execute.
   * Example: task open chrome and search for the latest AI news
 * vision <description>: A shortcut to capture the screen and ask a question about it.
   * Example: vision what application is currently open?
 * status: Display system status, resource usage, and task queue details.
 * logs: Show the most recent activity logs.
 * config: View or change the current configuration (e.g., toggle simulation mode).
 * quit or exit: Gracefully shut down the system.
Web Dashboard
Navigate to http://127.0.0.1:8000 in your web browser to access the real-time monitoring dashboard.
The dashboard displays:
 * System Status: CPU, Memory, and Agent status.
 * Task Queue: A live view of all pending, active, and completed tasks.
 * Logs: A stream of system logs.
 * Task Submission: A form to create new tasks directly from the web interface.
Safety Features
 * Simulation Mode: Enabled by default (SIMULATION_MODE="true"). In this mode, the ComputerUseAgent will not perform any actual mouse or keyboard actions. Instead, it will log the action it would have taken. This is crucial for safely testing and debugging task plans.
 * Human-in-the-Loop: Complex or potentially risky operations will be logged with a request for confirmation before proceeding (future implementation).
 * Permission Checks: The system checks for necessary macOS permissions on startup and will guide you if they are missing.
Troubleshooting
 * Permission Errors: If the agent can't see the screen or control the mouse, double-check that your terminal application has both Accessibility and Screen Recording permissions in System Settings. You may need to restart your terminal after granting them.
 * ModuleNotFoundError: Ensure you have activated the virtual environment (source venv/bin/activate) before running the script.
 * API Errors: Verify that your API keys in the .env file are correct and have sufficient credits.
