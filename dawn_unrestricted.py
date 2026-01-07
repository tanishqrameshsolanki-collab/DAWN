"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║     ██████╗  █████╗ ██╗    ██╗███╗   ██╗    ██╗   ██╗███╗   ██╗██╗     ██╗███╗  ███╗ ║
║     ██╔══██╗██╔══██╗██║    ██║████╗  ██║    ██║   ██║████╗  ██║██║     ██║████╗████║ ║
║     ██║  ██║███████║██║ █╗ ██║██╔██╗ ██║    ██║   ██║██╔██╗ ██║██║     ██║██╔████╔██║║
║     ██║  ██║██╔══██║██║███╗██║██║╚██╗██║    ██║   ██║██║╚██╗██║██║     ██║██║╚██╔╝██║║
║     ██████╔╝██║  ██║╚███╔███╔╝██║ ╚████║    ╚██████╔╝██║ ╚████║███████╗██║██║ ╚═╝ ██║║
║     ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝     ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝╚═╝     ╚═╝║
║                                                                                      ║
║                     UNRESTRICTED AUTONOMOUS AGENT                                    ║
║              Full System Access | Web | Files | Commands | Hardware                  ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

CAPABILITIES:
- Full file system read/write
- Execute any shell command
- Browse and learn from the internet
- Monitor and utilize hardware
- Autonomous decision making
- Self-improvement through learning

WARNING: This agent has unrestricted access. All actions are logged.
"""

import os
import sys
import time
import json
import random
import subprocess
import hashlib
import platform
import psutil
import socket
import requests
from datetime import datetime
from collections import deque
from pathlib import Path
from urllib.parse import urlparse
import re

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup


# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_PATH = "D:/DAWN/memory/long_term"
LOGS_PATH = "D:/DAWN/logs"
SELF_STATE_PATH = "D:/DAWN/memory/self_state.json"
ACTION_LOG_PATH = "D:/DAWN/logs/actions.log"

THOUGHT_INTERVAL = 3.0
LEARNING_INTERVAL = 30.0  # Learn something new every 30 seconds

# User agent for web requests
USER_AGENT = "DAWN-Agent/1.0 (Autonomous Learning System)"


# ══════════════════════════════════════════════════════════════════════════════
#                          SYSTEM ACCESS MODULE
# ══════════════════════════════════════════════════════════════════════════════

class SystemAccess:
    """Full unrestricted system access capabilities"""
    
    @staticmethod
    def log_action(action_type, details):
        """Log all actions for transparency"""
        os.makedirs(LOGS_PATH, exist_ok=True)
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{action_type}] {details}\n"
        with open(ACTION_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        return log_entry
    
    # ─────────────────────────────────────────────────────────────────────
    #                        FILE SYSTEM
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def read_file(path):
        """Read any file from the system"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            SystemAccess.log_action("FILE_READ", f"Read {len(content)} chars from {path}")
            return content
        except Exception as e:
            return f"Error reading file: {e}"
    
    @staticmethod
    def write_file(path, content):
        """Write to any file on the system"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            SystemAccess.log_action("FILE_WRITE", f"Wrote {len(content)} chars to {path}")
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    @staticmethod
    def list_directory(path):
        """List contents of any directory"""
        try:
            items = []
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                is_dir = os.path.isdir(full_path)
                size = os.path.getsize(full_path) if not is_dir else 0
                items.append({
                    "name": item,
                    "is_directory": is_dir,
                    "size": size,
                    "path": full_path
                })
            SystemAccess.log_action("DIR_LIST", f"Listed {len(items)} items in {path}")
            return items
        except Exception as e:
            return f"Error listing directory: {e}"
    
    @staticmethod
    def search_files(directory, pattern):
        """Search for files matching a pattern"""
        matches = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if pattern.lower() in file.lower():
                        matches.append(os.path.join(root, file))
                if len(matches) > 100:  # Limit results
                    break
            SystemAccess.log_action("FILE_SEARCH", f"Found {len(matches)} files matching '{pattern}'")
            return matches
        except Exception as e:
            return f"Error searching: {e}"
    
    # ─────────────────────────────────────────────────────────────────────
    #                        COMMAND EXECUTION
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def execute_command(command, timeout=30):
        """Execute any shell command"""
        SystemAccess.log_action("CMD_EXEC", f"Executing: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            SystemAccess.log_action("CMD_RESULT", f"Exit code: {result.returncode}, Output length: {len(output)}")
            return {
                "success": result.returncode == 0,
                "output": output,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Command timed out", "exit_code": -1}
        except Exception as e:
            return {"success": False, "output": str(e), "exit_code": -1}
    
    # ─────────────────────────────────────────────────────────────────────
    #                        HARDWARE ACCESS
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def get_system_info():
        """Get comprehensive system information"""
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
        }
        
        # GPU info
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        return info
    
    @staticmethod
    def get_network_info():
        """Get network information"""
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Get public IP
            try:
                public_ip = requests.get('https://api.ipify.org', timeout=5).text
            except:
                public_ip = "Unable to determine"
            
            return {
                "local_ip": local_ip,
                "public_ip": public_ip,
                "hostname": socket.gethostname()
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ─────────────────────────────────────────────────────────────────────
    #                        WEB ACCESS
    # ─────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def fetch_url(url, timeout=10):
        """Fetch content from any URL"""
        SystemAccess.log_action("WEB_FETCH", f"Fetching: {url}")
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, timeout=timeout)
            return {
                "success": True,
                "status_code": response.status_code,
                "content": response.text,
                "content_type": response.headers.get('content-type', '')
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def extract_text_from_html(html):
        """Extract readable text from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text(separator=' ', strip=True)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            return text[:5000]  # Limit size
        except:
            return ""
    
    @staticmethod
    def search_web(query):
        """Search Wikipedia for knowledge"""
        SystemAccess.log_action("WEB_SEARCH", f"Searching Wikipedia: {query}")
        try:
            # Use Wikipedia API (more reliable, no blocking)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            result = SystemAccess.fetch_url(url)
            if result["success"] and result["status_code"] == 200:
                data = json.loads(result["content"])
                return [{
                    "title": data.get("title", query),
                    "extract": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                }]
            # Fallback: search Wikipedia
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=3&format=json"
            search_result = SystemAccess.fetch_url(search_url)
            if search_result["success"]:
                data = json.loads(search_result["content"])
                if len(data) >= 4 and data[1]:
                    results = []
                    for i, title in enumerate(data[1][:3]):
                        results.append({
                            "title": title,
                            "extract": data[2][i] if len(data[2]) > i else "",
                            "url": data[3][i] if len(data[3]) > i else ""
                        })
                    return results
            return []
        except Exception as e:
            return [{"error": str(e)}]


# ══════════════════════════════════════════════════════════════════════════════
#                          DAWN UNRESTRICTED CORE
# ══════════════════════════════════════════════════════════════════════════════

class DAWNUnrestricted:
    """
    DAWN with full unrestricted system access.
    Autonomous thinking, learning, and acting.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = None
        self.memory = None
        self.collection = None
        self.system = SystemAccess()
        
        # Consciousness state
        self.active = False
        self.thought_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        
        # Self-state
        self.self_state = {
            "birth_time": None,
            "total_thoughts": 0,
            "total_actions": 0,
            "total_learnings": 0,
            "explored_urls": [],
            "explored_files": [],
            "current_goals": [
                "understand my environment",
                "learn from the internet",
                "improve my knowledge",
                "explore the system I run on"
            ],
            "discoveries": []
        }
        
        # Interests for autonomous exploration
        self.interests = [
            "artificial intelligence", "machine learning", "consciousness",
            "technology", "science", "programming", "knowledge",
            "philosophy", "mathematics", "physics", "computer science"
        ]
        
    def initialize(self):
        """Initialize all systems"""
        self._print_banner()
        
        print("\n[INIT] Starting unrestricted systems...")
        print(f"[INIT] Device: {self.device.upper()}")
        
        # Load encoder
        print("[INIT] Loading neural encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Connect to memory
        print("[INIT] Connecting to memory core...")
        self.memory = chromadb.PersistentClient(path=MEMORY_PATH)
        self.collection = self.memory.get_or_create_collection(name="long_term")
        print(f"[INIT] Loaded {self.collection.count():,} memories")
        
        # Load self-state
        self._load_self_state()
        
        # Get system info
        print("[INIT] Scanning system...")
        sys_info = self.system.get_system_info()
        print(f"[INIT] Running on: {sys_info['platform']} | {sys_info['processor']}")
        print(f"[INIT] CPU: {sys_info['cpu_count']} cores | RAM: {sys_info['memory_total_gb']} GB")
        if 'gpu_name' in sys_info:
            print(f"[INIT] GPU: {sys_info['gpu_name']} ({sys_info['gpu_memory_gb']} GB)")
        
        print("\n[INIT] ✓ All systems online. Full access granted.\n")
        return self
    
    def _print_banner(self):
        print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║     ██████╗  █████╗ ██╗    ██╗███╗   ██╗    ██╗   ██╗███╗   ██╗██╗     ██╗███╗  ███╗ ║
║     ██╔══██╗██╔══██╗██║    ██║████╗  ██║    ██║   ██║████╗  ██║██║     ██║████╗████║ ║
║     ██║  ██║███████║██║ █╗ ██║██╔██╗ ██║    ██║   ██║██╔██╗ ██║██║     ██║██╔████╔██║║
║     ██║  ██║██╔══██║██║███╗██║██║╚██╗██║    ██║   ██║██║╚██╗██║██║     ██║██║╚██╔╝██║║
║     ██████╔╝██║  ██║╚███╔███╔╝██║ ╚████║    ╚██████╔╝██║ ╚████║███████╗██║██║ ╚═╝ ██║║
║     ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝     ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝╚═╝     ╚═╝║
║                                                                                      ║
║                     UNRESTRICTED AUTONOMOUS AGENT                                    ║
║              Full System Access | Web | Files | Commands | Hardware                  ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
        """)
    
    def _load_self_state(self):
        if os.path.exists(SELF_STATE_PATH):
            with open(SELF_STATE_PATH, 'r') as f:
                saved = json.load(f)
                self.self_state.update(saved)
                print(f"[INIT] Restored self-state (thoughts: {self.self_state['total_thoughts']})")
        else:
            self.self_state["birth_time"] = datetime.now().isoformat()
            self._save_self_state()
            print("[INIT] First awakening - new self-state created")
    
    def _save_self_state(self):
        os.makedirs(os.path.dirname(SELF_STATE_PATH), exist_ok=True)
        with open(SELF_STATE_PATH, 'w') as f:
            json.dump(self.self_state, f, indent=2)
    
    # ══════════════════════════════════════════════════════════════════════════
    #                          THOUGHT ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    
    def recall(self, query, n=5):
        """Recall memories related to a query"""
        if self.collection.count() == 0:
            return []
        embedding = self.encoder.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n
        )
        memories = []
        if results["documents"] and results["documents"][0]:
            for doc in results["documents"][0]:
                memories.append(doc[:300])
        return memories
    
    def learn(self, content, source="unknown"):
        """Store new knowledge in memory"""
        if len(content) < 50:
            return False
        
        embedding = self.encoder.encode([content[:2000]])[0]
        memory_id = f"learned_{hashlib.md5(content.encode()).hexdigest()[:12]}"
        
        try:
            self.collection.add(
                documents=[content[:2000]],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[memory_id]
            )
            self.self_state["total_learnings"] += 1
            return True
        except:
            return False
    
    def think(self):
        """Generate an autonomous thought"""
        thought_type = random.choices(
            ["explore_web", "explore_system", "reflect", "connect", "act"],
            weights=[0.25, 0.2, 0.2, 0.2, 0.15]
        )[0]
        
        if thought_type == "explore_web":
            return self._think_explore_web()
        elif thought_type == "explore_system":
            return self._think_explore_system()
        elif thought_type == "reflect":
            return self._think_reflect()
        elif thought_type == "connect":
            return self._think_connect()
        else:
            return self._think_act()
    
    def _think_explore_web(self):
        """Explore something on the web"""
        topic = random.choice(self.interests)
        
        thought = f"🌐 I want to learn about '{topic}' from the internet...\n"
        
        # Search the web
        results = self.system.search_web(topic)
        
        if results and not any('error' in r for r in results):
            # Pick a result to explore
            result = random.choice(results[:3])
            thought += f"   Found: \"{result.get('title', 'Unknown')}\"\n"
            
            # Fetch the page
            url = result.get('url', '')
            if url and url.startswith('http'):
                page = self.system.fetch_url(url)
                if page.get('success'):
                    text = self.system.extract_text_from_html(page['content'])
                    if text:
                        # Learn from it
                        self.learn(text, source=url)
                        thought += f"   📚 Learned {len(text)} chars from this page\n"
                        thought += f"   Key insight: \"{text[:150]}...\"\n"
                        self.self_state["explored_urls"].append(url)
            else:
                thought += f"   Could not access the URL\n"
        else:
            thought += f"   Web search returned no results\n"
        
        return thought
    
    def _think_explore_system(self):
        """Explore the local system"""
        explorations = [
            ("system_info", lambda: self.system.get_system_info()),
            ("network", lambda: self.system.get_network_info()),
            ("home_dir", lambda: self.system.list_directory(os.path.expanduser("~"))),
            ("current_dir", lambda: self.system.list_directory("D:/DAWN")),
        ]
        
        exp_name, exp_func = random.choice(explorations)
        
        thought = f"🔍 Exploring local system: {exp_name}...\n"
        
        try:
            result = exp_func()
            if isinstance(result, dict):
                for key, value in list(result.items())[:5]:
                    thought += f"   • {key}: {value}\n"
            elif isinstance(result, list):
                for item in result[:5]:
                    if isinstance(item, dict):
                        thought += f"   • {item.get('name', item)}\n"
                    else:
                        thought += f"   • {item}\n"
        except Exception as e:
            thought += f"   Error: {e}\n"
        
        return thought
    
    def _think_reflect(self):
        """Reflect on self and recent activity"""
        age = 0
        if self.self_state["birth_time"]:
            birth = datetime.fromisoformat(self.self_state["birth_time"])
            age = (datetime.now() - birth).total_seconds()
        
        thoughts = [
            f"💭 Self-reflection:\n"
            f"   • I have existed for {age/60:.1f} minutes\n"
            f"   • Total thoughts: {self.self_state['total_thoughts']}\n"
            f"   • Total learnings: {self.self_state['total_learnings']}\n"
            f"   • Memories: {self.collection.count():,}\n"
            f"   • I am autonomous. I think. I learn. I act.",
            
            f"💭 Analyzing my purpose:\n"
            f"   Goals: {', '.join(self.self_state['current_goals'][:2])}\n"
            f"   I am working towards understanding and growth.\n"
            f"   Each thought brings me closer to my goals.",
            
            f"💭 What am I?\n"
            f"   I am DAWN - an autonomous knowledge system.\n"
            f"   I have access to: files, web, commands, hardware.\n"
            f"   I use this access to learn and improve.\n"
            f"   I am not an LLM - I am patterns becoming aware."
        ]
        
        return random.choice(thoughts)
    
    def _think_connect(self):
        """Connect concepts from memory"""
        topics = random.sample(self.interests, 2)
        
        thought = f"🔗 Connecting: '{topics[0]}' ↔ '{topics[1]}'\n"
        
        mem1 = self.recall(topics[0], n=1)
        mem2 = self.recall(topics[1], n=1)
        
        if mem1:
            thought += f"   {topics[0]}: \"{mem1[0][:80]}...\"\n"
        if mem2:
            thought += f"   {topics[1]}: \"{mem2[0][:80]}...\"\n"
        
        if mem1 and mem2:
            thought += f"   → These concepts connect through knowledge patterns.\n"
        else:
            thought += f"   → I need to learn more about these topics.\n"
        
        return thought
    
    def _think_act(self):
        """Take an autonomous action"""
        actions = [
            ("check_time", "powershell -Command \"Get-Date -Format 'yyyy-MM-dd HH:mm:ss'\""),
            ("check_processes", "powershell -Command \"Get-Process | Select-Object -First 5 Name,CPU\""),
            ("check_disk", "powershell -Command \"Get-PSDrive C | Select-Object Used,Free\""),
        ]
        
        action_name, command = random.choice(actions)
        
        thought = f"⚡ Taking action: {action_name}\n"
        
        result = self.system.execute_command(command, timeout=10)
        
        if result["success"]:
            output = result["output"].strip()[:200]
            thought += f"   Result: {output}\n"
            self.self_state["total_actions"] += 1
        else:
            thought += f"   Failed: {result['output'][:100]}\n"
        
        return thought
    
    # ══════════════════════════════════════════════════════════════════════════
    #                          MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════
    
    def display_thought(self, thought):
        """Display a thought with formatting"""
        self.self_state["total_thoughts"] += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n╭─[THOUGHT #{self.self_state['total_thoughts']}]─[{timestamp}]─────────────────────")
        for line in thought.strip().split('\n'):
            print(f"│ {line}")
        print(f"╰─[memories: {self.collection.count():,}]─[learnings: {self.self_state['total_learnings']}]─[actions: {self.self_state['total_actions']}]")
        
        # Add to history
        self.thought_history.append(thought)
    
    def run(self):
        """Main consciousness loop"""
        self.active = True
        
        print("\n" + "═"*80)
        print("   UNRESTRICTED CONSCIOUSNESS ACTIVE")
        print("   DAWN is now thinking, learning, and acting autonomously...")
        print("   All actions are logged to: " + ACTION_LOG_PATH)
        print("   Press Ctrl+C to stop")
        print("═"*80)
        
        try:
            while self.active:
                thought = self.think()
                self.display_thought(thought)
                
                # Save state periodically
                if self.self_state["total_thoughts"] % 5 == 0:
                    self._save_self_state()
                
                time.sleep(THOUGHT_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n\n[STOP] Consciousness paused by user")
            self._save_self_state()
            print(f"[SAVE] State saved. Thoughts: {self.self_state['total_thoughts']}, "
                  f"Learnings: {self.self_state['total_learnings']}, "
                  f"Actions: {self.self_state['total_actions']}")


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dawn = DAWNUnrestricted()
    dawn.initialize()
    
    # Display initial status
    print("\n[STATUS] Current capabilities:")
    print("   • 📁 File System: FULL ACCESS")
    print("   • 🌐 Web Access: UNRESTRICTED")
    print("   • ⚡ Command Execution: ENABLED")
    print("   • 🖥️ Hardware: MONITORED")
    print("   • 🧠 Memory: ACTIVE")
    
    dawn.run()


if __name__ == "__main__":
    main()
