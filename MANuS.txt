# -*- coding: utf-8 -*-
"""
Manus AI - Autonomous Agent System

A sophisticated autonomous AI agent system for macOS that combines local
orchestration with cloud AI inference APIs.

Author: Gemini
Date: 2025-08-30
Version: 1.0.0
"""

import asyncio
import base64
import datetime
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Union

# Third-party libraries - check requirements.txt
try:
    import aiohttp
    import psutil
    from colorama import Fore, Style, init
    from dotenv import load_dotenv
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from PIL import Image
    import pyautogui
    import pyperclip
    import pyttsx3
    import speech_recognition as sr
    import uvicorn
    from websockets.exceptions import ConnectionClosed
except ImportError as e:
    print(f"Error: Missing dependency '{e.name}'.")
    print("Please install the required packages by running:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Initialize colorama for colored terminal output
init(autoreset=True)

# --- Configuration Loading ---
load_dotenv()

# --- Global Constants & Enums ---
class AgentType(Enum):
    TASK_ORCHESTRATOR = auto()
    COMPUTER_USE = auto()
    VISION_ANALYZER = auto()
    PRODUCTIVITY = auto()
    USER = auto()

class TaskStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    AWAITING_CONFIRMATION = auto()

class AIProvider(Enum):
    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    
class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

@dataclass
class Task:
    id: str
    description: str
    agent_type: AgentType
    priority: int
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    result: Optional[Dict] = None
    error: Optional[str] = None
    sub_tasks: List['Task'] = field(default_factory=list)
    parent_task_id: Optional[str] = None

# --- Utility & Helper Functions ---

def get_unique_id() -> str:
    """Generates a unique ID based on the current timestamp."""
    return f"task_{int(time.time() * 1000)}"

def colored_print(message: str, color: str = Fore.WHITE, level: LogLevel = LogLevel.INFO):
    """Prints a colored and timestamped message to the console."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_map = {
        LogLevel.DEBUG: (Fore.CYAN, "DEBUG"),
        LogLevel.INFO: (Fore.GREEN, "INFO"),
        LogLevel.WARNING: (Fore.YELLOW, "WARN"),
        LogLevel.ERROR: (Fore.RED, "ERROR"),
        LogLevel.CRITICAL: (Style.BRIGHT + Fore.RED, "CRITICAL"),
    }
    log_color, level_name = level_map.get(level, (color, "INFO"))
    print(f"{Style.DIM}{timestamp}{Style.RESET_ALL} [{log_color}{level_name.ljust(8)}{Style.RESET_ALL}] {message}")


# --- Core System Components ---

class Config:
    """Manages system configuration loaded from environment variables."""
    def __init__(self):
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        
        self.simulation_mode: bool = os.getenv("SIMULATION_MODE", "true").lower() == "true"
        
        self.web_interface_host: str = os.getenv("WEB_INTERFACE_HOST", "127.0.0.1")
        self.web_interface_port: int = int(os.getenv("WEB_INTERFACE_PORT", "8000"))
        
        self.wake_word: str = os.getenv("WAKE_WORD", "manus").lower()

    def get_available_providers(self) -> List[AIProvider]:
        """Returns a list of configured AI providers."""
        providers = []
        if self.openai_api_key: providers.append(AIProvider.OPENAI)
        if self.anthropic_api_key: providers.append(AIProvider.ANTHROPIC)
        if self.google_api_key: providers.append(AIProvider.GOOGLE)
        return providers

    def to_dict(self):
        return {
            "simulation_mode": self.simulation_mode,
            "web_host": self.web_interface_host,
            "web_port": self.web_interface_port,
            "wake_word": self.wake_word,
            "available_providers": [p.name for p in self.get_available_providers()]
        }

class SystemMonitor:
    """Provides real-time monitoring of system resources."""
    def get_usage(self) -> Dict[str, Any]:
        """Returns current CPU and memory usage."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": datetime.datetime.now().isoformat()
        }

class ScreenCapture:
    """Handles screen capture functionality using native macOS tools."""
    def __init__(self):
        self.screenshot_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self.retina_scale = self._is_retina()

    def _is_retina(self) -> bool:
        """Checks if the display is a Retina display."""
        try:
            # A simple way to check is to see if the main screen resolution is large
            # This is not foolproof but works for most modern MacBooks.
            w, h = pyautogui.size()
            return w > 1920 or h > 1200
        except Exception:
            return False

    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Captures the screen or a specific region.
        Region is (x, y, width, height).
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.screenshot_dir, f"capture_{timestamp}.png")
        
        try:
            if region:
                # Adjust for Retina displays if needed, though screencapture handles it
                x, y, w, h = region
                command = f"screencapture -x -R {x},{y},{w},{h} {filepath}"
            else:
                command = f"screencapture -x {filepath}"
            
            subprocess.run(shlex.split(command), check=True, capture_output=True)
            colored_print(f"Screenshot saved to {filepath}", Fore.MAGENTA)
            return filepath
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            colored_print(f"Error during screen capture: {e}", Fore.RED, LogLevel.ERROR)
            # Fallback to pyautogui if screencapture fails
            try:
                screenshot = pyautogui.screenshot(region=region)
                screenshot.save(filepath)
                colored_print(f"Used pyautogui fallback for screenshot.", Fore.YELLOW, LogLevel.WARNING)
                return filepath
            except Exception as py_e:
                colored_print(f"PyAutoGUI fallback failed: {py_e}", Fore.RED, LogLevel.ERROR)
                raise IOError("Screen capture failed with both native and fallback methods.") from py_e

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """Encodes an image file to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            colored_print(f"Failed to encode image {image_path}: {e}", Fore.RED, LogLevel.ERROR)
            raise

class AIClient:
    """A unified client for interacting with multiple cloud AI APIs."""
    def __init__(self, config: Config):
        self.config = config
        self.providers = self.config.get_available_providers()
        if not self.providers:
            raise ValueError("No AI API keys found in .env file. At least one provider is required.")
        
        colored_print(f"AIClient initialized with providers: {[p.name for p in self.providers]}", Fore.CYAN)
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Initializes and returns an aiohttp ClientSession."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close_session(self):
        """Closes the aiohttp ClientSession."""
        if self.session:
            await self.session.close()

    async def _request_with_fallback(self, task_func_map: Dict[AIProvider, Coroutine]) -> Dict[str, Any]:
        """Tries providers in order and falls back on failure."""
        for provider in self.providers:
            if provider in task_func_map:
                try:
                    colored_print(f"Attempting request with {provider.name}...", Fore.CYAN)
                    return await task_func_map[provider]
                except Exception as e:
                    colored_print(f"Error with {provider.name}: {e}. Trying next provider.", Fore.YELLOW, LogLevel.WARNING)
        raise RuntimeError("All AI providers failed.")

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generates text using the best available provider."""
        task_map = {
            AIProvider.OPENAI: self._openai_chat_completion(prompt, system_prompt),
            AIProvider.ANTHROPIC: self._anthropic_chat_completion(prompt, system_prompt),
            AIProvider.GOOGLE: self._google_chat_completion(prompt, system_prompt)
        }
        return await self._request_with_fallback(task_map)

    async def analyze_image(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        """Analyzes an image with a text prompt."""
        task_map = {
            AIProvider.OPENAI: self._openai_vision_analysis(prompt, image_base64),
            AIProvider.GOOGLE: self._google_vision_analysis(prompt, image_base64),
            AIProvider.ANTHROPIC: self._anthropic_vision_analysis(prompt, image_base64)
        }
        return await self._request_with_fallback(task_map)

    # --- Provider-Specific Implementations ---

    async def _openai_chat_completion(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.7,
        }
        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"provider": "OpenAI", "content": data["choices"][0]["message"]["content"]}

    async def _openai_vision_analysis(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }],
            "max_tokens": 2000,
        }
        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"provider": "OpenAI", "content": data["choices"][0]["message"]["content"]}

    async def _anthropic_chat_completion(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"provider": "Anthropic", "content": data["content"][0]["text"]}
            
    async def _anthropic_vision_analysis(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 2000,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        }
        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"provider": "Anthropic", "content": data["content"][0]["text"]}

    async def _google_chat_completion(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={self.config.google_api_key}"
        contents = [{"parts": [{"text": prompt}]}]
        payload = {"contents": contents}
        if system_prompt:
             payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"provider": "Google", "content": data["candidates"][0]["content"]["parts"][0]["text"]}

    async def _google_vision_analysis(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.config.google_api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_base64}}
                ]
            }]
        }
        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return {"provider": "Google", "content": data["candidates"][0]["content"]["parts"][0]["text"]}


# --- Agent Definitions ---

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    def __init__(self, orchestrator: 'TaskOrchestrator'):
        self.orchestrator = orchestrator
        self.ai_client = orchestrator.ai_client

    @abstractmethod
    async def process_task(self, task: Task) -> Task:
        """Processes a given task and returns the updated task."""
        pass

class ComputerUseAgent(BaseAgent):
    """Agent for controlling the computer's GUI."""
    
    def __init__(self, orchestrator: 'TaskOrchestrator'):
        super().__init__(orchestrator)
        self.simulation_mode = self.orchestrator.config.simulation_mode

    async def process_task(self, task: Task) -> Task:
        """Analyzes screen, creates a plan, and executes it."""
        try:
            plan = await self.analyze_screen_and_plan(task.description)
            await self.orchestrator.log(f"Plan created for task '{task.id}':\n{json.dumps(plan, indent=2)}", AgentType.COMPUTER_USE)

            if not plan.get("plan"):
                raise ValueError("AI failed to generate a valid plan.")

            for step in plan["plan"]:
                await self.execute_action(step)
                await asyncio.sleep(1) # Pause between actions
            
            task.status = TaskStatus.COMPLETED
            task.result = {"message": "Plan executed successfully.", "plan": plan}
        except Exception as e:
            await self.orchestrator.log(f"Error processing computer use task: {e}", AgentType.COMPUTER_USE, LogLevel.ERROR)
            task.status = TaskStatus.FAILED
            task.error = str(e)
        return task

    async def analyze_screen_and_plan(self, task_description: str) -> Dict[str, Any]:
        """Captures screen and uses AI to create an action plan."""
        await self.orchestrator.log("Analyzing screen to create a plan...", AgentType.COMPUTER_USE)
        screenshot_path = self.orchestrator.screen_capture.capture_screen()
        image_b64 = ScreenCapture.encode_image_to_base64(screenshot_path)
        
        screen_size = pyautogui.size()
        prompt = f"""
You are an expert macOS computer operator. Your goal is to create a step-by-step plan to accomplish a task.
The user's screen is {screen_size.width}x{screen_size.height}.
The current screenshot is attached.

The user's request is: "{task_description}"

Analyze the screenshot and provide a JSON response with a plan of actions.
The available actions are:
- `click(x, y, description)`: Click a specific coordinate.
- `type(text, description)`: Type out a string of text.
- `press(key_combination, description)`: Press a keyboard combination (e.g., 'command+space', 'enter').
- `scroll(direction, amount, description)`: Scroll 'up' or 'down' by a certain amount.
- `wait(seconds, description)`: Pause for a number of seconds.
- `done(final_summary)`: Mark the task as complete.

Your response MUST be a valid JSON object with the following structure:
{{
    "analysis": "A brief analysis of what you see on the screen relevant to the task.",
    "thought": "Your step-by-step reasoning for the plan you are about to create.",
    "plan": [
        {{"action": "action_name", "parameters": {{"param1": "value1"}}, "description": "What this step does."}}
    ]
}}

Example:
{{
    "analysis": "I see the Google search page with the search bar visible.",
    "thought": "First, I need to click on the search bar. Then, I will type the user's query. Finally, I will press enter to initiate the search.",
    "plan": [
        {{"action": "click", "parameters": {{"x": 500, "y": 400}}, "description": "Click on the search bar."}},
        {{"action": "type", "parameters": {{"text": "latest AI news"}}, "description": "Type the search query."}},
        {{"action": "press", "parameters": {{"key_combination": "enter"}}, "description": "Press enter to search."}},
        {{"action": "done", "parameters": {{"final_summary": "The search has been performed."}}, "description": "Task is complete."}}
    ]
}}

Now, create the plan for the user's request.
"""
        
        response = await self.ai_client.analyze_image(prompt, image_b64)
        
        try:
            # Clean the response from markdown code blocks
            content = response['content']
            if content.strip().startswith("```json"):
                content = content.strip()[7:-3]
            
            return json.loads(content)
        except (json.JSONDecodeError, KeyError) as e:
            await self.orchestrator.log(f"Failed to parse AI plan response: {e}\nRaw response: {response['content']}", AgentType.COMPUTER_USE, LogLevel.ERROR)
            raise ValueError("Could not decode the plan from the AI response.")


    async def execute_action(self, step: Dict[str, Any]):
        """Executes a single action from the plan."""
        action = step.get("action")
        params = step.get("parameters", {})
        description = step.get("description", "No description")

        log_msg = f"Executing: {action.upper()} | Params: {params} | Desc: {description}"
        if self.simulation_mode:
            await self.orchestrator.log(f"[SIMULATION] {log_msg}", AgentType.COMPUTER_USE, LogLevel.WARNING)
            return

        await self.orchestrator.log(log_msg, AgentType.COMPUTER_USE)

        try:
            if action == "click":
                pyautogui.click(params['x'], params['y'])
            elif action == "type":
                pyautogui.typewrite(params['text'], interval=0.05)
            elif action == "press":
                keys = params['key_combination'].split('+')
                pyautogui.hotkey(*keys)
            elif action == "scroll":
                direction = -1 if params['direction'] == 'up' else 1
                pyautogui.scroll(direction * params['amount'])
            elif action == "wait":
                await asyncio.sleep(params['seconds'])
            elif action == "done":
                await self.orchestrator.log(f"Plan finished. Summary: {params.get('final_summary')}", AgentType.COMPUTER_USE)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            await self.orchestrator.log(f"Error executing action '{action}': {e}", AgentType.COMPUTER_USE, LogLevel.ERROR)
            raise

class VisionAnalyzerAgent(BaseAgent):
    """Agent for analyzing screen content without taking action."""
    async def process_task(self, task: Task) -> Task:
        await self.orchestrator.log(f"Analyzing screen for task: {task.description}", AgentType.VISION_ANALYZER)
        try:
            screenshot_path = self.orchestrator.screen_capture.capture_screen()
            image_b64 = ScreenCapture.encode_image_to_base64(screenshot_path)

            prompt = f"""
You are a vision analysis expert. Analyze the attached screenshot and answer the user's question.
Provide a concise and direct answer.

User's question: "{task.description}"
"""
            response = await self.ai_client.analyze_image(prompt, image_b64)
            
            task.status = TaskStatus.COMPLETED
            task.result = {"answer": response['content'], "provider": response['provider']}
            await self.orchestrator.log(f"Vision analysis complete. Result: {task.result['answer']}", AgentType.VISION_ANALYZER)

        except Exception as e:
            await self.orchestrator.log(f"Error during vision analysis: {e}", AgentType.VISION_ANALYZER, LogLevel.ERROR)
            task.status = TaskStatus.FAILED
            task.error = str(e)

        return task
        
class ProductivityAgent(BaseAgent):
    """Agent for productivity tasks like drafting emails or organizing files."""
    async def process_task(self, task: Task) -> Task:
        await self.orchestrator.log(f"Processing productivity task: {task.description}", AgentType.PRODUCTIVITY)
        try:
            # Example: Draft an email
            if "draft an email" in task.description.lower():
                prompt = f"""
You are a helpful assistant. Draft a professional email based on the following request.
Provide only the email content.

Request: "{task.description}"
"""
                response = await self.ai_client.generate_text(prompt)
                
                # Copy to clipboard for user convenience
                pyperclip.copy(response['content'])
                
                task.status = TaskStatus.COMPLETED
                task.result = {"draft": response['content'], "provider": response['provider'], "message": "Email draft copied to clipboard."}
                await self.orchestrator.log("Email draft generated and copied to clipboard.", AgentType.PRODUCTIVITY)
            else:
                # Placeholder for other productivity tasks
                task.status = TaskStatus.FAILED
                task.error = "Productivity task not yet implemented."
                
        except Exception as e:
            await self.orchestrator.log(f"Error during productivity task: {e}", AgentType.PRODUCTIVITY, LogLevel.ERROR)
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
        return task

# --- Main Orchestrator ---

class TaskOrchestrator:
    """The central hub for managing tasks, agents, and system state."""
    def __init__(self, config: Config):
        self.config = config
        self.task_queue = asyncio.PriorityQueue()
        self.tasks: Dict[str, Task] = {}
        self.logs: List[Dict] = []
        self._running = False
        
        self.system_monitor = SystemMonitor()
        self.screen_capture = ScreenCapture()
        self.ai_client = AIClient(config)
        self.web_broadcaster = WebBroadcaster()

        self.agents = {
            AgentType.COMPUTER_USE: ComputerUseAgent(self),
            AgentType.VISION_ANALYZER: VisionAnalyzerAgent(self),
            AgentType.PRODUCTIVITY: ProductivityAgent(self),
        }

    async def log(self, message: str, agent: AgentType, level: LogLevel = LogLevel.INFO):
        """Logs a message and broadcasts it to clients."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": agent.name,
            "message": message,
            "level": level.name
        }
        self.logs.append(log_entry)
        if len(self.logs) > 200: # Keep logs from growing indefinitely
            self.logs.pop(0)

        color_map = {
            LogLevel.DEBUG: Fore.CYAN,
            LogLevel.INFO: Fore.WHITE,
            LogLevel.WARNING: Fore.YELLOW,
            LogLevel.ERROR: Fore.RED,
            LogLevel.CRITICAL: Style.BRIGHT + Fore.RED,
        }
        
        colored_message = f"{Style.DIM}[{agent.name}]{Style.RESET_ALL} {message}"
        colored_print(colored_message, color=color_map.get(level, Fore.WHITE), level=level)
        
        await self.web_broadcaster.broadcast(self.get_full_state())

    async def add_task(self, description: str, agent_type: AgentType, priority: int = 10) -> Task:
        """Adds a new task to the queue."""
        task_id = get_unique_id()
        task = Task(id=task_id, description=description, agent_type=agent_type, priority=priority)
        self.tasks[task_id] = task
        await self.task_queue.put((priority, task))
        await self.log(f"New task added: '{description}' ({agent_type.name})", AgentType.TASK_ORCHESTRATOR)
        return task

    def get_full_state(self) -> str:
        """Gets the complete state of the system as a JSON string."""
        state = {
            "tasks": [t.__dict__ for t in self.tasks.values()],
            "logs": self.logs[-50:], # Send last 50 logs
            "system_usage": self.system_monitor.get_usage(),
            "config": self.config.to_dict(),
            "agent_status": {agent.name: "IDLE" for agent in AgentType}, # Simplified status
        }
        # Correctly serialize datetime and enums
        return json.dumps(state, default=lambda o: o.isoformat() if isinstance(o, datetime.datetime) else o.name if isinstance(o, Enum) else str(o))


    async def task_processor(self):
        """The main loop that processes tasks from the queue."""
        self._running = True
        await self.log("Task processor started.", AgentType.TASK_ORCHESTRATOR)
        
        while self._running:
            try:
                priority, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                if task.id not in self.tasks:
                    continue # Task was cancelled

                task.status = TaskStatus.ACTIVE
                await self.log(f"Processing task {task.id}: {task.description}", AgentType.TASK_ORCHESTRATOR)
                
                agent = self.agents.get(task.agent_type)
                if agent:
                    updated_task = await agent.process_task(task)
                    self.tasks[task.id] = updated_task
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"No agent found for type {task.agent_type.name}"
                
                await self.log(f"Task {task.id} finished with status: {task.status.name}", AgentType.TASK_ORCHESTRATOR)
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                await self.log(f"Unhandled error in task processor: {e}", AgentType.TASK_ORCHESTRATOR, LogLevel.CRITICAL)
                # Potentially re-queue the task or mark as failed
                if 'task' in locals() and self.tasks.get(task.id):
                    self.tasks[task.id].status = TaskStatus.FAILED
                    self.tasks[task.id].error = f"Orchestrator error: {e}"

    async def stop(self):
        """Gracefully stops the task processor."""
        self._running = False
        await self.log("Shutting down orchestrator...", AgentType.TASK_ORCHESTRATOR)
        await self.ai_client.close_session()


# --- Web Interface ---

class WebBroadcaster:
    """Manages WebSocket connections for real-time updates."""
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        colored_print("Web client connected.", Fore.CYAN)

    def disconnect(self, websocket: WebSocket):
        self.connections.remove(websocket)
        colored_print("Web client disconnected.", Fore.CYAN)

    async def broadcast(self, message: str):
        """Sends a message to all connected clients."""
        for connection in self.connections:
            try:
                await connection.send_text(message)
            except (ConnectionClosed, WebSocketDisconnect):
                # This can happen if a client disconnects abruptly
                # The disconnect method will handle removal
                pass

def create_web_app(orchestrator: TaskOrchestrator) -> FastAPI:
    """Creates the FastAPI application."""
    app = FastAPI()
    
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Manus AI Dashboard</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="[https://cdn.tailwindcss.com](https://cdn.tailwindcss.com)"></script>
            <style>
                body { font-family: 'Inter', sans-serif; }
                .task-card { transition: all 0.2s ease-in-out; }
                .task-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
                ::-webkit-scrollbar { width: 8px; }
                ::-webkit-scrollbar-track { background: #1a202c; }
                ::-webkit-scrollbar-thumb { background: #4a5568; border-radius: 4px; }
            </style>
        </head>
        <body class="bg-gray-900 text-gray-200">
            <div id="app" class="p-4 md:p-8 max-w-7xl mx-auto">
                <header class="flex justify-between items-center mb-6">
                    <div>
                        <h1 class="text-3xl font-bold text-white">Manus AI Dashboard</h1>
                        <p class="text-gray-400">Real-time Autonomous Agent Monitoring</p>
                    </div>
                    <div class="text-right">
                        <p class="text-sm">Status: <span :class="status_color" class="font-semibold">{{ status }}</span></p>
                        <p class="text-xs text-gray-500">CPU: {{ system.cpu_percent }}% | MEM: {{ system.memory_percent }}%</p>
                    </div>
                </header>

                <!-- Main Grid -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Task Management Column -->
                    <div class="lg:col-span-2 space-y-6">
                        <!-- Task Submission -->
                        <div class="bg-gray-800 p-4 rounded-lg shadow-lg">
                            <h2 class="text-xl font-semibold mb-3">Submit New Task</h2>
                            <form @submit.prevent="submitTask">
                                <textarea v-model="newTask.description" class="w-full bg-gray-700 rounded p-2 text-white placeholder-gray-400" placeholder="Describe the task... e.g., 'Open Chrome and find today's weather'"></textarea>
                                <div class="flex items-center justify-between mt-3">
                                    <select v-model="newTask.agent_type" class="bg-gray-700 rounded p-2">
                                        <option value="COMPUTER_USE">Computer Use</option>
                                        <option value="VISION_ANALYZER">Vision Analysis</option>
                                        <option value="PRODUCTIVITY">Productivity</option>
                                    </select>
                                    <button type="submit" class="bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2 px-4 rounded-lg">Submit</button>
                                </div>
                            </form>
                        </div>
                        
                        <!-- Task Queue -->
                        <div>
                            <h2 class="text-xl font-semibold mb-3">Task Queue</h2>
                            <div class="space-y-4">
                                <div v-if="tasks.length === 0" class="text-gray-500 text-center py-4">No tasks in queue.</div>
                                <div v-for="task in tasks" :key="task.id" class="task-card bg-gray-800 p-4 rounded-lg shadow-md border-l-4" :class="taskBorderColor(task.status)">
                                    <div class="flex justify-between items-start">
                                        <div>
                                            <p class="font-bold text-white">{{ task.description }}</p>
                                            <p class="text-xs text-gray-400">ID: {{ task.id }} | Agent: {{ task.agent_type }}</p>
                                        </div>
                                        <span class="text-sm font-semibold px-2 py-1 rounded" :class="taskStatusColor(task.status)">{{ task.status }}</span>
                                    </div>
                                    <div v-if="task.error" class="mt-2 text-red-400 text-sm bg-red-900/50 p-2 rounded">
                                        <strong>Error:</strong> {{ task.error }}
                                    </div>
                                    <div v-if="task.result" class="mt-2 text-gray-300 text-sm bg-gray-700 p-2 rounded">
                                        <strong>Result:</strong> <pre class="whitespace-pre-wrap">{{ formatResult(task.result) }}</pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- System Logs Column -->
                    <div class="bg-gray-800 p-4 rounded-lg shadow-lg max-h-[80vh] overflow-y-auto">
                        <h2 class="text-xl font-semibold mb-3">System Logs</h2>
                        <div class="space-y-2 text-sm">
                            <div v-for="log in logs" :key="log.timestamp">
                                <p>
                                    <span class="text-gray-500">{{ formatTime(log.timestamp) }}</span>
                                    <span :class="logLevelColor(log.level)" class="font-bold"> [{{ log.agent }}]</span>
                                    <span> {{ log.message }}</span>
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script src="[https://unpkg.com/vue@3](https://unpkg.com/vue@3)"></script>
            <script>
                const { createApp } = Vue

                createApp({
                    data() {
                        return {
                            status: 'Connecting...',
                            status_color: 'text-yellow-400',
                            tasks: [],
                            logs: [],
                            system: { cpu_percent: 0, memory_percent: 0 },
                            newTask: { description: '', agent_type: 'COMPUTER_USE' },
                            socket: null
                        }
                    },
                    mounted() {
                        this.connectWebSocket()
                    },
                    methods: {
                        connectWebSocket() {
                            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                            this.socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
                            this.socket.onopen = () => {
                                this.status = 'Connected';
                                this.status_color = 'text-green-400';
                            };
                            this.socket.onmessage = (event) => {
                                const data = JSON.parse(event.data);
                                this.tasks = data.tasks.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                                this.logs = data.logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                                this.system = data.system_usage;
                            };
                            this.socket.onclose = () => {
                                this.status = 'Disconnected';
                                this.status_color = 'text-red-400';
                                setTimeout(this.connectWebSocket, 3000); // Reconnect after 3s
                            };
                        },
                        async submitTask() {
                            if (!this.newTask.description.trim()) return;
                            try {
                                await fetch('/api/tasks', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(this.newTask)
                                });
                                this.newTask.description = '';
                            } catch (error) {
                                console.error('Failed to submit task:', error);
                            }
                        },
                        formatResult(result) {
                            return typeof result === 'object' ? JSON.stringify(result, null, 2) : result;
                        },
                        formatTime(timestamp) {
                             return new Date(timestamp).toLocaleTimeString();
                        },
                        taskBorderColor(status) {
                            const colors = { PENDING: 'border-blue-500', ACTIVE: 'border-yellow-500', COMPLETED: 'border-green-500', FAILED: 'border-red-500' };
                            return colors[status] || 'border-gray-500';
                        },
                        taskStatusColor(status) {
                            const colors = { PENDING: 'bg-blue-500/50 text-blue-300', ACTIVE: 'bg-yellow-500/50 text-yellow-300', COMPLETED: 'bg-green-500/50 text-green-300', FAILED: 'bg-red-500/50 text-red-300' };
                            return colors[status] || 'bg-gray-500/50';
                        },
                        logLevelColor(level) {
                            const colors = { INFO: 'text-cyan-400', WARNING: 'text-yellow-400', ERROR: 'text-red-400', CRITICAL: 'text-red-300' };
                            return colors[level] || 'text-gray-400';
                        }
                    }
                }).mount('#app')
            </script>
        </body>
    </html>
    """

    @app.get("/", response_class=HTMLResponse)
    async def get():
        return html

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await orchestrator.web_broadcaster.connect(websocket)
        try:
            # Send initial state
            await websocket.send_text(orchestrator.get_full_state())
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            orchestrator.web_broadcaster.disconnect(websocket)
            
    @app.post("/api/tasks")
    async def create_task(task_data: dict):
        description = task_data.get("description")
        agent_type_str = task_data.get("agent_type")
        if not description or not agent_type_str:
            return {"error": "Missing description or agent_type"}
        
        try:
            agent_type = AgentType[agent_type_str]
            await orchestrator.add_task(description, agent_type)
            return {"status": "success", "message": "Task created"}
        except KeyError:
            return {"error": f"Invalid agent_type: {agent_type_str}"}

    return app

# --- Voice Control ---

class VoiceController:
    """Handles voice commands."""
    def __init__(self, orchestrator: 'TaskOrchestrator'):
        self.orchestrator = orchestrator
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self._listening = False
        self.wake_word = self.orchestrator.config.wake_word

    def speak(self, text: str):
        """Converts text to speech."""
        colored_print(f"Speaking: '{text}'", Fore.CYAN)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen_in_background(self):
        """Starts listening for the wake word in a background thread."""
        self._listening = True
        thread = threading.Thread(target=self._background_listener, daemon=True)
        thread.start()
        colored_print(f"Voice control enabled. Say '{self.wake_word}' to activate.", Fore.CYAN)

    def _background_listener(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

        stop_listening = self.recognizer.listen_in_background(
            self.microphone, self._process_audio, phrase_time_limit=5
        )
        
        while self._listening:
            time.sleep(0.1)
        
        stop_listening(wait_for_stop=False)
        colored_print("Background listening stopped.", Fore.YELLOW)


    def _process_audio(self, recognizer, audio):
        """Callback for processing audio from the background listener."""
        try:
            text = recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                asyncio.run_coroutine_threadsafe(self.handle_command(), self.orchestrator.task_processor_loop)
        except sr.UnknownValueError:
            pass # Ignore if speech is not understood
        except sr.RequestError as e:
            msg = f"Could not request results from Google Speech Recognition service; {e}"
            asyncio.run_coroutine_threadsafe(
                self.orchestrator.log(msg, AgentType.USER, LogLevel.ERROR),
                self.orchestrator.task_processor_loop
            )

    async def handle_command(self):
        """Handles a command after the wake word is detected."""
        await self.orchestrator.log("Wake word detected! Listening for command...", AgentType.USER)
        self.speak("Yes?")
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                command = self.recognizer.recognize_google(audio)
                await self.orchestrator.log(f"Voice command received: '{command}'", AgentType.USER)
                self.speak(f"Understood. Processing command: {command}")
                # Simple routing for voice commands
                if "what do you see" in command.lower() or "analyze the screen" in command.lower():
                    await self.orchestrator.add_task(command, AgentType.VISION_ANALYZER)
                else:
                    await self.orchestrator.add_task(command, AgentType.COMPUTER_USE)

            except sr.WaitTimeoutError:
                self.speak("I didn't hear a command.")
            except sr.UnknownValueError:
                self.speak("Sorry, I could not understand that.")
            except sr.RequestError as e:
                self.speak("There was an issue with the speech service.")
                await self.orchestrator.log(f"Speech service error: {e}", AgentType.USER, LogLevel.ERROR)

    def stop(self):
        self._listening = False

# --- Main Application Class ---

class ManusAI:
    """The main application class that ties everything together."""

    def __init__(self):
        self._check_platform()
        self.config = Config()
        self.orchestrator = TaskOrchestrator(self.config)
        self.voice_controller = VoiceController(self.orchestrator)
        self.web_app = create_web_app(self.orchestrator)
        self._running = True

    def _check_platform(self):
        """Ensures the system is running on macOS."""
        if platform.system() != "Darwin":
            colored_print("CRITICAL: Manus AI is optimized for macOS and may not function correctly on other operating systems.", Fore.RED, LogLevel.CRITICAL)
            if input("Continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)

    def _startup_checks(self):
        """Performs checks for permissions and configuration."""
        colored_print("--- Manus AI Startup Checks ---", Style.BRIGHT)
        
        # 1. Check AI Providers
        if not self.config.get_available_providers():
            colored_print("No AI API keys are configured in the .env file. The system will not be able to perform AI tasks.", Fore.RED, LogLevel.ERROR)
        else:
            colored_print(f"Found API keys for: {[p.name for p in self.config.get_available_providers()]}", Fore.GREEN)

        # 2. Check Simulation Mode
        if self.config.simulation_mode:
            colored_print("Simulation Mode is ON. No real computer actions will be performed.", Fore.YELLOW, LogLevel.WARNING)
        else:
            colored_print("Simulation Mode is OFF. The agent WILL control your computer. Supervise carefully.", Fore.RED, LogLevel.CRITICAL)
        
        # 3. Accessibility Permissions (Basic check)
        # A true check is difficult, so we guide the user
        colored_print("Please ensure your terminal has Accessibility and Screen Recording permissions in System Settings > Privacy & Security.", Fore.CYAN)
        colored_print("--- End of Checks ---", Style.BRIGHT)


    async def run_cli(self):
        """The main interactive command-line interface loop."""
        print(Fore.CYAN + Style.BRIGHT + "\nWelcome to Manus AI. Type 'help' for a list of commands.")
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                command_input = await loop.run_in_executor(None, lambda: input(Fore.YELLOW + "manus> " + Style.RESET_ALL).strip())
                if not command_input:
                    continue

                parts = command_input.split(" ", 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ["quit", "exit"]:
                    self._running = False
                elif command == "help":
                    print("""
    Available Commands:
      task <description>    - Create a new task for the system to execute.
      vision <description>  - Capture screen and ask a question about it.
      status                - Display system status and task queue.
      logs                  - Show the most recent activity logs.
      config [key] [val]    - View or change configuration (e.g., `config simulation_mode false`).
      help                  - Show this help message.
      exit / quit           - Gracefully shut down the system.
                    """)
                elif command == "task":
                    if args:
                        await self.orchestrator.add_task(args, AgentType.COMPUTER_USE)
                    else:
                        print(Fore.RED + "Usage: task <description>")
                elif command == "vision":
                    if args:
                        await self.orchestrator.add_task(args, AgentType.VISION_ANALYZER)
                    else:
                        print(Fore.RED + "Usage: vision <description>")
                elif command == "status":
                    state = json.loads(self.orchestrator.get_full_state())
                    print(f"CPU: {state['system_usage']['cpu_percent']}% | Memory: {state['system_usage']['memory_percent']}%")
                    print(f"Tasks in queue: {len([t for t in state['tasks'] if t['status'] == 'PENDING'])}")
                    print(f"Active tasks: {len([t for t in state['tasks'] if t['status'] == 'ACTIVE'])}")
                elif command == "logs":
                    for log in self.orchestrator.logs[-10:]:
                        print(f"{log['timestamp']} [{log['level']}] [{log['agent']}] {log['message']}")
                elif command == "config":
                    config_parts = args.split()
                    if len(config_parts) == 2:
                        key, val_str = config_parts
                        if hasattr(self.orchestrator.config, key):
                            current_val = getattr(self.orchestrator.config, key)
                            if isinstance(current_val, bool):
                                new_val = val_str.lower() == 'true'
                            elif isinstance(current_val, int):
                                new_val = int(val_str)
                            else:
                                new_val = val_str
                            setattr(self.orchestrator.config, key, new_val)
                            # Special handling for simulation mode in agent
                            if key == 'simulation_mode':
                                self.orchestrator.agents[AgentType.COMPUTER_USE].simulation_mode = new_val
                            await self.orchestrator.log(f"Config '{key}' updated to '{new_val}'.", AgentType.USER)
                        else:
                             print(Fore.RED + f"Unknown config key: {key}")
                    else:
                        print(json.dumps(self.orchestrator.config.to_dict(), indent=2))
                else:
                    print(Fore.RED + "Unknown command. Type 'help' for a list of commands.")

            except (KeyboardInterrupt, EOFError):
                self._running = False
            except Exception as e:
                colored_print(f"An error occurred in the CLI: {e}", Fore.RED, LogLevel.ERROR)

        print("\nCLI is shutting down...")


    async def main(self):
        """The main entry point that runs all components concurrently."""
        self._startup_checks()

        # Set the event loop for the voice controller
        self.orchestrator.task_processor_loop = asyncio.get_running_loop()

        # Start the web server in a separate thread
        uvicorn_config = uvicorn.Config(self.web_app, host=self.config.web_interface_host, port=self.config.web_interface_port, log_level="warning")
        server = uvicorn.Server(uvicorn_config)
        web_thread = threading.Thread(target=server.run, daemon=True)
        web_thread.start()
        
        await self.orchestrator.log(f"Web dashboard running at http://{self.config.web_interface_host}:{self.config.web_interface_port}", AgentType.TASK_ORCHESTRATOR)

        # Start background voice listening
        self.voice_controller.listen_in_background()

        # Run task processor and CLI concurrently
        processor_task = asyncio.create_task(self.orchestrator.task_processor())
        cli_task = asyncio.create_task(self.run_cli())

        await asyncio.gather(processor_task, cli_task)

        # Cleanup
        await self.orchestrator.stop()
        self.voice_controller.stop()
        # The web thread is a daemon, so it will exit automatically
        colored_print("Manus AI has shut down.", Fore.GREEN)


if __name__ == "__main__":
    try:
        app = ManusAI()
        asyncio.run(app.main())
    except Exception as e:
        colored_print(f"A critical error occurred: {e}", Fore.RED, LogLevel.CRITICAL)
        sys.exit(1)

