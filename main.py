import streamlit as st
import os
import sys
import subprocess
import tempfile
import uuid
import json
import re
import inspect
import shutil
import urllib.parse
import requests
import zipfile
import ast
from pathlib import Path
from datetime import datetime
from enum import Enum, auto
from io import BytesIO

# ============================================================================
# APP METADATA & CONFIGURATION
# ============================================================================

APP_NAME = "EfficientManim"
APP_VERSION = "1.0.0-stable"
PROJECT_EXT = ".efp"

st.set_page_config(
    page_title=f"{APP_NAME} Web",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DEPENDENCY CHECKS & IMPORTS
# ============================================================================

# Check for Manim
try:
    import manim
    from manim import *
    from manim.utils.color import ManimColor
    MANIM_AVAILABLE = True
    MANIM_VER = manim.__version__
except ImportError:
    MANIM_AVAILABLE = False
    MANIM_VER = "Not Installed"

# Check for Google GenAI
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Check for Pydub
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ============================================================================
# DATA STRUCTURES & ENUMS
# ============================================================================

class NodeType(Enum):
    """Defines the category of a node."""
    MOBJECT = auto()
    ANIMATION = auto()
    SCENE_CONFIG = auto()

class NodeData:
    """
    Represents a single node in the generation graph.
    Can be a generic Mobject (Circle, Text) or an Animation (FadeIn, Rotate).
    """
    def __init__(self, name, node_type, cls_name):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = node_type
        self.cls_name = cls_name
        self.params = {}          # Dictionary of parameter values
        self.param_metadata = {}  # Metadata: enabled, type, hints
        self.audio_asset_id = None
        self.is_ai_generated = False
        self.ai_source = None
        self.preview_path = None
        
        # Load initial defaults based on Manim inspection
        if MANIM_AVAILABLE:
            self._load_defaults()

    def _load_defaults(self):
        """Introspect Manim classes to populate default parameters."""
        try:
            # Handle special string cases immediately
            if self.cls_name == "Text":
                self.params["text"] = "Hello World"
                return
            if self.cls_name == "MathTex":
                self.params["arg0"] = "E = mc^2"
                return

            cls = getattr(manim, self.cls_name, None)
            if not cls:
                return

            sig = inspect.signature(cls.__init__)
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'args', 'kwargs', 'mobject', 'scene'):
                    continue
                
                # Skip highly complex types for the web UI defaults
                if param.default is not inspect.Parameter.empty:
                    val = param.default
                    if isinstance(val, (int, float, str, bool)):
                        self.params[param_name] = val
                    elif val is None:
                        self.params[param_name] = None
                    elif isinstance(val, (list, tuple)) and len(val) <= 3:
                        # Simple vectors
                        self.params[param_name] = val
        except Exception:
            pass

    def is_param_enabled(self, param_name):
        """Check if parameter is active for code generation."""
        return self.param_metadata.get(param_name, {}).get("enabled", True)

    def set_param_enabled(self, param_name, enabled):
        """Enable or disable a parameter."""
        if param_name not in self.param_metadata:
            self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["enabled"] = enabled

    def should_escape_string(self, param_name):
        """Determines if the string value should be treated as raw code or a string literal."""
        return self.param_metadata.get(param_name, {}).get("escape", False)

    def set_escape_string(self, param_name, escape):
        if param_name not in self.param_metadata:
            self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["escape"] = escape

    def to_dict(self):
        """Serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.name,
            "cls_name": self.cls_name,
            "params": self.params,
            "param_metadata": self.param_metadata,
            "audio_asset_id": self.audio_asset_id,
            "is_ai_generated": self.is_ai_generated,
            "ai_source": self.ai_source,
            "preview_path": self.preview_path
        }

    @staticmethod
    def from_dict(data):
        """Deserialization."""
        node = NodeData(data["name"], NodeType[data["type"]], data["cls_name"])
        node.id = data["id"]
        node.params = data["params"]
        node.param_metadata = data.get("param_metadata", {})
        node.audio_asset_id = data.get("audio_asset_id")
        node.is_ai_generated = data.get("is_ai_generated", False)
        node.ai_source = data.get("ai_source")
        node.preview_path = data.get("preview_path")
        return node

class Asset:
    """Represents an external file (Image, Audio, SVG)."""
    def __init__(self, name, path, kind):
        self.id = str(uuid.uuid4())
        self.name = name
        self.original_path = str(Path(path).as_posix())
        self.current_path = str(Path(path).as_posix())
        self.kind = kind # 'image', 'audio', 'video', 'svg'
        self.local_file = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "original": self.original_path,
            "kind": self.kind,
            "local": self.local_file
        }

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    defaults = {
        "nodes": {},
        "connections": [], # List of dicts: {'start': node_id, 'end': node_id}
        "assets": {},
        "logs": [],
        "temp_dir": Path(tempfile.mkdtemp(prefix="eff_manim_")),
        "selected_node_id": None,
        "generated_code": "",
        "project_name": "Untitled",
        "ai_code_cache": "",
        "last_latex_img": None,
        "render_history": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
TEMP_DIR = st.session_state.temp_dir

# ============================================================================
# LOGGING & UTILITIES
# ============================================================================

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {level}: {msg}"
    st.session_state.logs.append(entry)
    # Keep log size manageable
    if len(st.session_state.logs) > 200:
        st.session_state.logs.pop(0)

class TypeSafeParser:
    """Helper class to guess types for UI generation."""
    
    @staticmethod
    def is_asset_param(param_name):
        n = param_name.lower()
        if any(x in n for x in ["filename", "file", "image", "sound", "svg"]):
            return True
        return False

    @staticmethod
    def is_color_param(param_name):
        kw = {'color', 'fill', 'stroke', 'bg', 'fg'}
        # Exclude opacity/width/etc
        if any(x in param_name.lower() for x in ['opacity', 'width', 'width']):
            return False
        return any(k in param_name.lower() for k in kw)

    @staticmethod
    def is_numeric_param(param_name, value):
        if isinstance(value, (int, float)):
            return True
        kw = {'radius', 'width', 'height', 'scale', 'size', 'thickness',
              'font_size', 'length', 'rate', 'opacity', 'alpha',
              'x', 'y', 'z', 'angle', 'run_time'}
        return any(k in param_name.lower() for k in kw)

    @staticmethod
    def parse_numeric(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def parse_color(value):
        if isinstance(value, str) and value.startswith("#") and len(value) == 7:
            return value
        return "#FFFFFF"

def get_asset_path(asset_id):
    if asset_id in st.session_state.assets:
        return st.session_state.assets[asset_id].current_path
    return None

# ============================================================================
# COMPILER ENGINE
# ============================================================================

def format_param_value(param_name, value, node_data):
    """
    Converts a stored parameter value into a Python code string.
    """
    try:
        # Check if value matches an Asset ID
        if isinstance(value, str) and value in st.session_state.assets:
            path = get_asset_path(value)
            if path:
                # Use raw string for paths
                clean_path = path.replace("\\", "/")
                return f'r"{clean_path}"'
            return '""'
        
        # Check if value matches a Node ID (linking nodes via code variables)
        if isinstance(value, str) and len(value) == 36 and value in st.session_state.nodes:
            return f"m_{value[:6]}"
        
        # Check if it's a raw latex string
        if isinstance(value, str) and value.startswith('r"""') and value.endswith('"""'):
            return value

        # Check explicit escape flag
        if node_data.should_escape_string(param_name):
            # If strictly escaped, return as is (assumed valid python code/variable)
            return str(value)
        
        # Color handling
        if TypeSafeParser.is_color_param(param_name):
            return f'"{value}"'
        
        # Numeric handling
        if TypeSafeParser.is_numeric_param(param_name, value):
            return str(value)
        
        # Boolean
        if isinstance(value, bool):
            return str(value)
        
        # Default string handling
        return f'"{str(value)}"'
    except Exception as e:
        log(f"Format Error ({param_name}): {e}", "WARN")
        return f'"{value}"'

def compile_graph():
    """
    Traverses the graph (session state) and generates a complete Manim Python script.
    """
    code = [
        "from manim import *",
        "import numpy as np"
    ]
    
    if PYDUB_AVAILABLE:
        code.append("from pydub import AudioSegment")
    
    code.append("")
    code.append("class EfficientScene(Scene):")
    code.append("    def construct(self):")
    
    nodes = st.session_state.nodes
    conns = st.session_state.connections
    
    # 1. Instantiate Mobjects
    # -----------------------
    mobjects = [n for n in nodes.values() if n.type == NodeType.MOBJECT]
    m_vars = {} # Map ID -> Variable Name
    
    code.append("        # --- Initialize Mobjects ---")
    
    for m in mobjects:
        args_str_list = []
        
        # Handle positional args (e.g. text for Text mobject)
        pos_args = {}
        named_args = {}
        
        for k, v in m.params.items():
            if not m.is_param_enabled(k):
                continue
            
            # Convention: keys starting with "argX" are positional
            if k.startswith("arg") and k[3:].isdigit():
                idx = int(k[3:])
                pos_args[idx] = format_param_value(k, v, m)
            else:
                named_args[k] = v
        
        # Sort and add positional args
        for i in sorted(pos_args.keys()):
            args_str_list.append(pos_args[i])
            
        # Add named args
        for k, v in named_args.items():
            val_str = format_param_value(k, v, m)
            args_str_list.append(f"{k}={val_str}")
            
        var_name = f"m_{m.id[:6]}"
        m_vars[m.id] = var_name
        
        code.append(f"        {var_name} = {m.cls_name}({', '.join(args_str_list)})")
        
        # Mobjects usually need to be added or they won't exist for animations
        # However, Create/FadeIn handles adding. 
        # We only explicitly add if there are no 'Creation' animations targeting it
        # For simplicity in this version, we assume animations handle visibility or we use Scene.add
        # But to be safe, we don't 'self.add' everything immediately if it's going to be Created.
        # Let's assume user connects a 'Create' animation, otherwise we 'self.add' at the start?
        # Better approach: Just define them. The animations determine flow.
    
    # 2. Sequence Animations
    # ----------------------
    # We need an order. For now, we use the creation order of the Animation nodes,
    # or ideally, a 'sequence' index parameter. Here we take list order.
    animations = [n for n in nodes.values() if n.type == NodeType.ANIMATION]
    # Sort animations if we had a sequence ID, for now, arbitrary/creation order
    
    code.append("")
    code.append("        # --- Animations ---")
    
    for anim in animations:
        # Find targets (Mobjects connected TO this animation)
        # In our connection list: start=Mobject, end=Animation
        targets = []
        for c in conns:
            if c['end'] == anim.id and c['start'] in m_vars:
                targets.append(m_vars[c['start']])
        
        anim_args = []
        
        # Add targets as first arguments (except for Wait)
        if anim.cls_name != "Wait":
            anim_args.extend(targets)
        
        # Handle Audio Integration
        if anim.audio_asset_id and PYDUB_AVAILABLE:
            path = get_asset_path(anim.audio_asset_id)
            if path:
                clean_path = path.replace("\\", "/")
                # Create a temporary audio variable
                audio_var = f"audio_{anim.id[:6]}"
                code.append(f"        {audio_var} = AudioSegment.from_file(r'{clean_path}')")
                code.append(f"        self.add_sound(r'{clean_path}')")
                # Override run_time to match audio duration
                anim.params['run_time'] = f"{audio_var}.duration_seconds"
        
        # Process parameters
        for k, v in anim.params.items():
            if not anim.is_param_enabled(k):
                continue
            
            # Special handling for run_time if it was overridden by audio
            if k == 'run_time' and "duration_seconds" in str(v):
                anim_args.append(f"{k}={v}")
            else:
                val_str = format_param_value(k, v, anim)
                anim_args.append(f"{k}={val_str}")
        
        # Generate the play call
        if anim.cls_name == "Wait":
            # Wait takes run_time as pos arg or kwarg
            rt = anim.params.get('run_time', 1.0)
            code.append(f"        self.wait({rt})")
        elif targets or anim.cls_name in ["FadeIn", "FadeOut", "Create", "Write"]: 
            # Some anims work without explicit connection if args are passed manually,
            # but usually require a mobject.
            if targets:
                constr = f"{anim.cls_name}({', '.join(anim_args)})"
                code.append(f"        self.play({constr})")
            else:
                code.append(f"        # Skipped {anim.name}: No target mobject connected")
        else:
            code.append(f"        # Skipped {anim.name}: Invalid configuration")

    final_code = "\n".join(code)
    st.session_state.generated_code = final_code
    return final_code

def render_manim(code, mode="image", fps=15, quality="l"):
    """
    Executes Manim in a subprocess.
    """
    if not MANIM_AVAILABLE:
        st.error("Manim library is not installed in this environment.")
        return None

    # Write code to temp file
    script_path = TEMP_DIR / f"render_{uuid.uuid4().hex}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    output_dir = TEMP_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Construct command
    # manim -ql -s script.py EfficientScene --format=png/mp4
    cmd = [sys.executable, "-m", "manim", "--disable_caching"]
    
    if mode == "image":
        cmd.append("-s") # Save last frame
        cmd.append("--format=png")
    else:
        cmd.append("--format=mp4")
    
    cmd.append(f"-q{quality}")
    cmd.append(f"--fps={fps}")
    cmd.append(f"--media_dir={str(output_dir)}")
    cmd.append(str(script_path))
    cmd.append("EfficientScene")

    log(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run subprocess
        env = os.environ.copy()
        # Ensure path compatibility
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            creationflags = 0

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(TEMP_DIR),
            env=env,
            timeout=300 # 5 minute timeout
        )

        if process.returncode != 0:
            log(f"Render STDERR: {process.stderr}", "ERROR")
            st.error("Render Failed. See logs for details.")
            st.code(process.stderr)
            return None

        # Locate output file
        # Manim directory structure: media_dir/videos/script_name/quality/EfficientScene.ext
        # Or for images: media_dir/images/script_name/EfficientScene.png
        
        search_dir = output_dir
        ext = ".png" if mode == "image" else ".mp4"
        
        found_files = list(search_dir.rglob(f"*{ext}"))
        if not found_files:
            log("Manim finished but no output file found.", "WARN")
            return None
            
        # Return the most recently modified file
        latest_file = max(found_files, key=os.path.getmtime)
        return latest_file

    except subprocess.TimeoutExpired:
        st.error("Rendering timed out.")
        return None
    except Exception as e:
        log(f"Execution Error: {e}", "ERROR")
        st.error(f"Execution Error: {e}")
        return None

# ============================================================================
# AI INTEGRATION
# ============================================================================

def generate_ai_code_gemini(prompt, model_name, api_key):
    """Call Google Gemini API."""
    if not GENAI_AVAILABLE:
        st.error("Google GenAI library missing.")
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        
        sys_prompt = (
            "You are a Python Manim expert. "
            "Generate a complete 'EfficientScene(Scene)' class.\n"
            "Rules:\n"
            "1. Instantiate mobjects with variable names (e.g. circle_1 = Circle()).\n"
            "2. Use self.play() for animations.\n"
            "3. Do NOT use config dictionaries, keep it linear.\n"
            "4. Output ONLY valid Python code inside ```python blocks."
        )
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=sys_prompt)]),
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            ]
        )
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        log(f"Gemini Error: {e}", "ERROR")
        return None

def parse_ai_nodes(code_text):
    """
    Parses Python code to reverse-engineer Nodes.
    Note: This is a heuristic parser using Regex and AST.
    """
    st.session_state.nodes = {}
    st.session_state.connections = []
    
    # Extract code block
    match = re.search(r"```python(.*?)```", code_text, re.DOTALL)
    if match:
        code_text = match.group(1)
    
    # Variable tracking
    vars_to_ids = {}
    
    try:
        tree = ast.parse(code_text)
        
        for node in ast.walk(tree):
            # Find Assignments: var = Class(...)
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call) and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    if isinstance(node.value.func, ast.Name):
                        cls_name = node.value.func.id
                        
                        # Check if it looks like a Manim Mobject
                        if hasattr(manim, cls_name):
                            # Create Node
                            new_node = NodeData(var_name, NodeType.MOBJECT, cls_name)
                            new_node.is_ai_generated = True
                            
                            # extract params
                            for kw in node.value.keywords:
                                if isinstance(kw.value, ast.Constant):
                                    new_node.params[kw.arg] = kw.value.value
                            
                            st.session_state.nodes[new_node.id] = new_node
                            vars_to_ids[var_name] = new_node.id
                            
            # Find Plays: self.play(Anim(var))
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name == "play":
                    for arg in node.args:
                        if isinstance(arg, ast.Call):
                            # Animation Call
                            if isinstance(arg.func, ast.Name):
                                anim_cls = arg.func.id
                                anim_node = NodeData(anim_cls, NodeType.ANIMATION, anim_cls)
                                anim_node.is_ai_generated = True
                                st.session_state.nodes[anim_node.id] = anim_node
                                
                                # Check args for targets
                                for sub_arg in arg.args:
                                    if isinstance(sub_arg, ast.Name) and sub_arg.id in vars_to_ids:
                                        target_id = vars_to_ids[sub_arg.id]
                                        st.session_state.connections.append({
                                            "start": target_id,
                                            "end": anim_node.id
                                        })
    except Exception as e:
        log(f"AI Parse Error: {e}", "ERROR")
        st.warning(f"Could not fully parse AI code: {e}")

# ============================================================================
# PROJECT I/O
# ============================================================================

def save_project_zip():
    meta = {
        "name": st.session_state.project_name,
        "created": str(datetime.now()),
        "version": APP_VERSION
    }
    
    # Serialize Nodes
    nodes_data = [n.to_dict() for n in st.session_state.nodes.values()]
    graph_data = {
        "nodes": nodes_data,
        "connections": st.session_state.connections
    }
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(meta, indent=2))
        zf.writestr("graph.json", json.dumps(graph_data, indent=2))
        zf.writestr("code.py", st.session_state.generated_code)
        
        asset_manifest = []
        for a in st.session_state.assets.values():
            s_path = Path(a.current_path)
            if s_path.exists():
                # Store assets in a folder inside zip
                arcname = f"assets/{a.id}{s_path.suffix}"
                zf.write(s_path, arcname)
                d = a.to_dict()
                d["local"] = arcname
                asset_manifest.append(d)
        
        zf.writestr("assets.json", json.dumps(asset_manifest, indent=2))
    
    return zip_buffer.getvalue()

def load_project_zip(file_obj):
    try:
        with zipfile.ZipFile(file_obj, 'r') as zf:
            # Clear current state
            st.session_state.nodes = {}
            st.session_state.connections = []
            st.session_state.assets = {}
            
            # Extract to temp
            proj_dir = TEMP_DIR / f"loaded_{uuid.uuid4().hex}"
            proj_dir.mkdir(exist_ok=True)
            zf.extractall(proj_dir)
            
            # Load Graph
            with open(proj_dir / "graph.json", "r") as f:
                g_data = json.load(f)
                for nd in g_data["nodes"]:
                    n = NodeData.from_dict(nd)
                    st.session_state.nodes[n.id] = n
                st.session_state.connections = g_data.get("connections", [])
            
            # Load Assets
            if (proj_dir / "assets.json").exists():
                with open(proj_dir / "assets.json", "r") as f:
                    a_list = json.load(f)
                    for ad in a_list:
                        # Re-point paths to extracted temp location
                        local_name = ad.get("local", "")
                        extracted_path = proj_dir / local_name
                        if extracted_path.exists():
                            a = Asset(ad["name"], str(extracted_path), ad["kind"])
                            a.id = ad["id"]
                            st.session_state.assets[a.id] = a
                            
            # Load Code
            if (proj_dir / "code.py").exists():
                with open(proj_dir / "code.py", "r") as f:
                    st.session_state.generated_code = f.read()

        st.success("Project loaded successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Failed to load project: {e}")
        log(f"Load Error: {e}", "ERROR")

# ============================================================================
# UI: SIDEBAR
# ============================================================================

with st.sidebar:
    st.title(f"{APP_NAME} 🕸️")
    st.caption(f"v{APP_VERSION}")
    
    # --- File Management ---
    with st.expander("📁 Project", expanded=True):
        st.session_state.project_name = st.text_input("Project Name", st.session_state.project_name)
        
        c1, c2 = st.columns(2)
        if c1.button("New"):
            st.session_state.nodes = {}
            st.session_state.connections = []
            st.rerun()
        
        if c2.button("Save"):
            zip_data = save_project_zip()
            st.download_button(
                "Download", 
                zip_data, 
                file_name=f"{st.session_state.project_name}{PROJECT_EXT}",
                mime="application/zip"
            )

        uploaded_file = st.file_uploader("Load Project", type=["efp", "zip"])
        if uploaded_file:
            if st.button("Import File"):
                load_project_zip(uploaded_file)

    # --- Settings ---
    with st.expander("⚙️ Settings"):
        api_key = st.text_input("Gemini API Key", type="password", help="Required for AI/TTS")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            
        gemini_model = st.selectbox("AI Model", ["gemini-2.0-flash-exp", "gemini-1.5-pro"])
        
        render_quality = st.select_slider(
            "Render Quality",
            options=["Low (480p)", "Medium (720p)", "High (1080p)"],
            value="Low (480p)"
        )
        fps_val = st.number_input("FPS", 15, 60, 15)

    # --- Assets ---
    st.subheader("📂 Assets")
    uploaded_asset = st.file_uploader("Upload", type=['png','jpg','jpeg','mp3','wav','svg'], accept_multiple_files=True)
    if uploaded_asset:
        for uf in uploaded_asset:
            dest = TEMP_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            
            kind = "unknown"
            if "image" in uf.type: kind = "image"
            elif "audio" in uf.type: kind = "audio"
            elif "svg" in uf.type: kind = "image"
            
            asset = Asset(uf.name, str(dest), kind)
            st.session_state.assets[asset.id] = asset
    
    # Asset List
    for a in st.session_state.assets.values():
        icon = "🎵" if a.kind == 'audio' else "🖼️"
        st.caption(f"{icon} {a.name}")

# ============================================================================
# UI: MAIN TABS
# ============================================================================

tab_graph, tab_props, tab_ai, tab_latex, tab_voice, tab_render = st.tabs([
    "🕸️ Graph", "✏️ Properties", "🤖 AI Gen", "✒️ LaTeX", "🎙️ Audio", "🎬 Render"
])

# ----------------------------------------------------------------------------
# TAB 1: GRAPH EDITOR
# ----------------------------------------------------------------------------
with tab_graph:
    c1, c2, c3 = st.columns([1, 1.5, 1.5])
    
    # --- Add Node Column ---
    with c1:
        st.markdown("### ➕ Add Node")
        node_cat = st.radio("Type", ["Mobject", "Animation"], horizontal=True)
        
        if node_cat == "Mobject":
            # Extended list of common Manim Mobjects
            options = ["Circle", "Square", "Rectangle", "Triangle", "Line", "Arrow", 
                       "Text", "MathTex", "Star", "Dot", "RegularPolygon", "NumberPlane", "Axes"]
            cls_sel = st.selectbox("Class", options)
        else:
            options = ["Create", "FadeIn", "FadeOut", "Write", "DrawBorderThenFill", 
                       "Rotate", "GrowFromCenter", "Transform", "Wait"]
            cls_sel = st.selectbox("Class", options)
            
        n_name = st.text_input("Name", value=f"My{cls_sel}")
        
        if st.button("Add to Scene", use_container_width=True):
            ntype = NodeType.MOBJECT if node_cat == "Mobject" else NodeType.ANIMATION
            node = NodeData(n_name, ntype, cls_sel)
            st.session_state.nodes[node.id] = node
            st.toast(f"Added {n_name}")
            st.rerun()

    # --- Mobject List ---
    with c2:
        st.markdown("### 📦 Objects")
        mobs = [n for n in st.session_state.nodes.values() if n.type == NodeType.MOBJECT]
        if not mobs:
            st.caption("No objects yet.")
        
        for m in mobs:
            with st.container(border=True):
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(f"**{m.name}** (`{m.cls_name}`)")
                
                if col_b.button("Edit", key=f"ed_{m.id}"):
                    st.session_state.selected_node_id = m.id
                    
                if col_b.button("🗑️", key=f"del_{m.id}"):
                    del st.session_state.nodes[m.id]
                    # Cleanup connections
                    st.session_state.connections = [c for c in st.session_state.connections 
                                                  if c['start'] != m.id and c['end'] != m.id]
                    st.rerun()

    # --- Animation List (Connections) ---
    with c3:
        st.markdown("### 🎬 Action Stack")
        anims = [n for n in st.session_state.nodes.values() if n.type == NodeType.ANIMATION]
        mobs_dict = {m.id: m.name for m in mobs}
        
        if not anims:
            st.caption("No animations yet.")
            
        for idx, a in enumerate(anims):
            with st.container(border=True):
                st.markdown(f"**{idx+1}. {a.name}** (`{a.cls_name}`)")
                
                # Connection Logic
                if a.cls_name != "Wait":
                    # Find current target
                    current_target = None
                    for c in st.session_state.connections:
                        if c['end'] == a.id:
                            current_target = c['start']
                            break
                    
                    # Dropdown to select target
                    new_target = st.selectbox(
                        "Target Object", 
                        options=[None] + list(mobs_dict.keys()),
                        format_func=lambda x: mobs_dict[x] if x else "(None)",
                        key=f"target_sel_{a.id}",
                        index=0 if not current_target else (list(mobs_dict.keys()).index(current_target) + 1)
                    )
                    
                    # Update Connection if changed
                    if new_target != current_target:
                        # Remove old
                        st.session_state.connections = [c for c in st.session_state.connections if c['end'] != a.id]
                        # Add new
                        if new_target:
                            st.session_state.connections.append({"start": new_target, "end": a.id})
                        st.rerun()
                
                c_del_a, c_edit_a = st.columns(2)
                if c_edit_a.button("Edit Params", key=f"ed_a_{a.id}"):
                    st.session_state.selected_node_id = a.id
                if c_del_a.button("Remove", key=f"del_a_{a.id}"):
                    del st.session_state.nodes[a.id]
                    st.session_state.connections = [c for c in st.session_state.connections if c['end'] != a.id]
                    st.rerun()

# ----------------------------------------------------------------------------
# TAB 2: PROPERTIES
# ----------------------------------------------------------------------------
with tab_props:
    nid = st.session_state.selected_node_id
    if nid and nid in st.session_state.nodes:
        node = st.session_state.nodes[nid]
        
        st.header(f"Editing: {node.name}")
        st.caption(f"ID: {node.id}")
        
        # Rename
        new_name = st.text_input("Node Name", node.name)
        if new_name != node.name:
            node.name = new_name
            st.rerun()
            
        st.divider()
        
        # --- Parameter Editor ---
        params_to_remove = []
        
        for k, v in list(node.params.items()):
            col_k, col_v, col_del = st.columns([2, 4, 1])
            
            col_k.text(k)
            
            # Dynamic Input Types
            key_id = f"p_{nid}_{k}"
            new_val = v
            
            # 1. Color
            if TypeSafeParser.is_color_param(k):
                try:
                    current_hex = TypeSafeParser.parse_color(v)
                    new_val = col_v.color_picker("Color", current_hex, label_visibility="collapsed", key=key_id)
                except:
                    new_val = col_v.text_input("Val", str(v), label_visibility="collapsed", key=key_id)
            
            # 2. Asset
            elif TypeSafeParser.is_asset_param(k) and st.session_state.assets:
                opts = {a.id: a.name for a in st.session_state.assets.values()}
                # Current selection
                curr_idx = 0
                if v in opts:
                    curr_keys = list(opts.keys())
                    if v in curr_keys:
                        curr_idx = curr_keys.index(v) + 1
                
                sel = col_v.selectbox("Asset", [None] + list(opts.keys()), 
                                      format_func=lambda x: opts[x] if x else "None", 
                                      index=curr_idx, label_visibility="collapsed", key=key_id)
                if sel: new_val = sel
                
            # 3. Numeric
            elif TypeSafeParser.is_numeric_param(k, v):
                try:
                    f_val = float(v) if v is not None else 0.0
                    new_val = col_v.number_input("Num", value=f_val, label_visibility="collapsed", key=key_id)
                except:
                    new_val = col_v.text_input("Val", str(v), label_visibility="collapsed", key=key_id)
                    
            # 4. Boolean
            elif isinstance(v, bool):
                new_val = col_v.checkbox("True/False", value=v, label_visibility="collapsed", key=key_id)
                
            # 5. Generic String
            else:
                new_val = col_v.text_input("Val", str(v), label_visibility="collapsed", key=key_id)
            
            # Update
            if new_val != v:
                node.params[k] = new_val
                
            if col_del.button("❌", key=f"del_param_{key_id}"):
                params_to_remove.append(k)

        # Apply removals
        for k in params_to_remove:
            del node.params[k]
            st.rerun()

        # Add new param
        with st.expander("Add Custom Parameter"):
            with st.form("add_p"):
                nk = st.text_input("Param Name (e.g., color, radius)")
                nv = st.text_input("Value")
                if st.form_submit_button("Add"):
                    node.params[nk] = nv
                    st.rerun()

        # Preview for Mobjects
        if node.type == NodeType.MOBJECT:
            st.divider()
            if st.button("Generate Preview Snapshot"):
                with st.spinner("Rendering Preview..."):
                    # Construct minimal scene
                    code_snippet = f"""
from manim import *
class Preview(Scene):
    def construct(self):
        obj = {node.cls_name}(**{{}}) # Placeholder
        # Re-inject logic manually for preview would be complex, 
        # so we rely on the main compiler logic usually.
                    """ 
                    # Actually, let's just use the main compiler but only for this node
                    # Hack: create temp state with only this node
                    # Ideally, we call format_param_value
                    
                    args = []
                    for k, v in node.params.items():
                        args.append(f"{k}={format_param_value(k, v, node)}")
                    
                    full_code = f"""
from manim import *
class EfficientScene(Scene):
    def construct(self):
        obj = {node.cls_name}({', '.join(args)})
        self.add(obj)
                    """
                    path = render_manim(full_code, "image", 15, "l")
                    if path:
                        node.preview_path = str(path)
            
            if node.preview_path and os.path.exists(node.preview_path):
                st.image(node.preview_path, caption="Node Preview")

    else:
        st.info("Select a node from the Graph tab to edit properties.")

# ----------------------------------------------------------------------------
# TAB 3: AI GENERATION
# ----------------------------------------------------------------------------
with tab_ai:
    st.header("🤖 Generative Animation")
    st.markdown("Describe the scene you want, and the AI will build the graph.")
    
    prompt = st.text_area("Prompt", placeholder="A blue circle appears on the left, then transforms into a red square on the right.")
    
    if st.button("Generate Nodes"):
        if not api_key:
            st.error("Please set Gemini API Key in Sidebar Settings.")
        else:
            with st.spinner("Dreaming up math..."):
                code = generate_ai_code_gemini(prompt, gemini_model, api_key)
                if code:
                    st.session_state.ai_code_cache = code
                    st.success("Code generated! Review below.")
    
    if st.session_state.ai_code_cache:
        with st.expander("Raw AI Code"):
            st.code(st.session_state.ai_code_cache, language="python")
            
        if st.button("Merge into Graph"):
            parse_ai_nodes(st.session_state.ai_code_cache)
            st.success("Nodes merged successfully!")
            st.rerun()

# ----------------------------------------------------------------------------
# TAB 4: LATEX
# ----------------------------------------------------------------------------
with tab_latex:
    st.header("✒️ LaTeX Helper")
    
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        tex_in = st.text_area("Equation", r"e^{i\pi} + 1 = 0")
        if st.button("Render Preview"):
            # Use Manim itself to render a preview image of latex
            # This is more robust than external APIs
            if not MANIM_AVAILABLE:
                st.error("Manim not available.")
            else:
                p_code = f"""
from manim import *
class EfficientScene(Scene):
    def construct(self):
        t = MathTex(r"{tex_in}")
        self.add(t)
                """
                with st.spinner("Rendering LaTeX..."):
                    path = render_manim(p_code, "image", 15, "l")
                    if path:
                        # Convert to bytes for display logic if needed, or just path
                        st.session_state.last_latex_img = str(path)
    
    with l_col2:
        if st.session_state.last_latex_img:
            st.image(st.session_state.last_latex_img, caption="Preview")
            
            st.markdown("#### Apply to Node")
            # Filter for Text/MathTex nodes
            txt_nodes = {n.id: n.name for n in st.session_state.nodes.values() 
                         if "Tex" in n.cls_name or "Text" in n.cls_name}
            
            if txt_nodes:
                target_n = st.selectbox("Select Node", list(txt_nodes.keys()), 
                                        format_func=lambda x: txt_nodes[x])
                
                if st.button("Apply LaTeX to Node"):
                    n = st.session_state.nodes[target_n]
                    # MathTex uses arg0 usually, Text uses 'text'
                    formatted = f'r"""{tex_in}"""'
                    
                    if "MathTex" in n.cls_name:
                        n.params["arg0"] = formatted
                        n.set_escape_string("arg0", True) # Don't quote again
                    else:
                        n.params["text"] = formatted
                        n.set_escape_string("text", True)
                    
                    st.success(f"Updated {n.name}")
            else:
                st.warning("Create a MathTex node first to apply this equation.")

# ----------------------------------------------------------------------------
# TAB 5: AUDIO (TTS)
# ----------------------------------------------------------------------------
with tab_voice:
    st.header("🎙️ Voice & Sync")
    
    tts_text = st.text_area("Script", "The area of a circle is pi times the radius squared.")
    # Gemini Voices (Standard ones)
    tts_voice = st.selectbox("Voice", ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Zephyr"])
    
    if st.button("Generate Speech"):
        if not api_key:
            st.error("API Key required.")
        elif not GENAI_AVAILABLE:
            st.error("GenAI lib missing.")
        else:
            try:
                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    response_modalities=["audio"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=tts_voice)
                        )
                    )
                )
                with st.spinner("Synthesizing..."):
                    response = client.models.generate_content(
                        model=gemini_model,
                        contents=types.Content(parts=[types.Part.from_text(text=tts_text)]),
                        config=config
                    )
                    
                    if response.candidates and response.candidates[0].content.parts:
                        part = response.candidates[0].content.parts[0]
                        if part.inline_data:
                            audio_bytes = part.inline_data.data
                            fname = f"tts_{uuid.uuid4().hex[:6]}.wav"
                            fpath = TEMP_DIR / fname
                            with open(fpath, "wb") as f:
                                f.write(audio_bytes)
                            
                            # Register Asset
                            asset = Asset(f"TTS: {tts_text[:15]}...", str(fpath), "audio")
                            st.session_state.assets[asset.id] = asset
                            st.audio(audio_bytes)
                            st.success("Audio asset created.")
            except Exception as e:
                st.error(f"TTS Failed: {e}")

    st.divider()
    st.subheader("Link Audio to Animation")
    st.info("Linking audio to an animation will automatically set the animation's run_time to match the audio duration.")
    
    # Linking Logic
    link_anim = st.selectbox("Animation", [n.id for n in st.session_state.nodes.values() if n.type == NodeType.ANIMATION],
                             format_func=lambda x: st.session_state.nodes[x].name)
    link_audio = st.selectbox("Audio Asset", [a.id for a in st.session_state.assets.values() if a.kind == 'audio'],
                              format_func=lambda x: st.session_state.assets[x].name)
    
    if link_anim and link_audio:
        if st.button("Link"):
            st.session_state.nodes[link_anim].audio_asset_id = link_audio
            st.success("Linked!")

# ----------------------------------------------------------------------------
# TAB 6: RENDER
# ----------------------------------------------------------------------------
with tab_render:
    st.header("🎬 Production")
    
    # 1. Compile
    code_preview = compile_graph()
    with st.expander("Inspect Python Code", expanded=False):
        st.code(code_preview, language="python")
    
    # 2. Config
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        mode = st.radio("Output Mode", ["Preview (Image)", "Video (MP4)"])
    with col_r2:
        qual_map = {"Low (480p)": "l", "Medium (720p)": "m", "High (1080p)": "h"}
        q_code = qual_map.get(render_quality, "l")
        st.write(f"Settings: {render_quality} @ {fps_val} FPS")
    
    # 3. Action
    if st.button("🚀 START RENDER", type="primary", use_container_width=True):
        if not st.session_state.nodes:
            st.warning("Scene is empty. Add nodes first.")
        else:
            m = "image" if "Image" in mode else "video"
            
            with st.spinner(f"Rendering {m}... Check logs below for progress."):
                result_path = render_manim(code_preview, m, fps_val, q_code)
                
                if result_path:
                    st.success("Render Successful!")
                    if m == "image":
                        st.image(str(result_path))
                    else:
                        st.video(str(result_path))
                    
                    with open(result_path, "rb") as f:
                        st.download_button("Download Output", f, file_name=result_path.name)
                else:
                    st.error("Render failed.")

# ============================================================================
# FOOTER / LOGS
# ============================================================================

st.markdown("---")
with st.expander("Console Logs"):
    if st.session_state.logs:
        st.text("\n".join(st.session_state.logs[-10:]))
    else:
        st.caption("No logs available.")
        