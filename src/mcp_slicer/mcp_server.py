# server.py
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent, ImageContent
import requests
import atexit
import json
import base64
import os
import uuid
from urllib.parse import urlparse
from typing import Optional
from io import BytesIO
from PIL import Image

# Create an MCP server (MCP SDK 2.x: FastMCP renamed to MCPServer)
mcp = MCPServer("SlicerMCP")

SLICER_WEB_SERVER_URL = os.environ.get("SLICER_WEB_SERVER_URL", "http://localhost:2016/slicer").rstrip("/")


def get_proxy_config(url: str) -> Optional[dict]:
    """
    Get proxy configuration for requests based on URL.
    
    For localhost/127.0.0.1 connections, always disable proxy to avoid
    connection issues when system proxy is configured but not running.
    
    For other connections, return None to use system default proxy settings
    (if configured via environment variables).
    
    Args:
        url: The target URL for the request
        
    Returns:
        dict: Proxy configuration dict for requests library
        - For localhost: {'http': None, 'https': None} to disable proxy
        - For other hosts: None to use system defaults
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    
    # Always bypass proxy for localhost connections
    if hostname in ("localhost", "127.0.0.1", "::1") or hostname.startswith("127."):
        return {'http': None, 'https': None}
    
    # For other connections, return None to use system proxy settings
    # if available via HTTP_PROXY, HTTPS_PROXY environment variables
    return None

# One pooled HTTP client per MCP server process: connections to Slicer are reused
# instead of being opened for every tool call.
_http = requests.Session()
if get_proxy_config(SLICER_WEB_SERVER_URL) is not None:
    # Loopback Slicer: ignore HTTP(S)_PROXY settings entirely (a per-request
    # proxies={...: None} does not override environment proxies on a Session).
    _http.trust_env = False


# ---------------------------------------------------------------------------
# Sessions
#
# Several agents may share one Slicer instance. Each agent works in its own
# Slicer web server session (requires a Slicer WebServer with session support):
# a private Python namespace for execute_python_code, ownership tracking of the
# nodes it creates, and deletes that only affect its own nodes. Sessions are
# light-weight: the scene, loaded data and views are shared, not copied.
#
# Every MCP server process gets a default session, so separate agent processes
# are isolated automatically. Agents that share one MCP server process (e.g.
# sub-agents) should call create_session and pass the returned id as `session`.
#
# Environment:
#   SLICER_MCP_SESSION  default session id (kept when the process exits);
#                       "shared" or "none" disables sessions (legacy behavior).
# ---------------------------------------------------------------------------

_SESSION_ENV = os.environ.get("SLICER_MCP_SESSION", "").strip()
if _SESSION_ENV.lower() in ("shared", "none"):
    DEFAULT_SESSION = None
else:
    DEFAULT_SESSION = _SESSION_ENV or f"mcp-{uuid.uuid4().hex[:8]}"
_DEFAULT_SESSION_IS_EPHEMERAL = not _SESSION_ENV
_default_session_used = False


def _resolve_session(session: Optional[str]) -> Optional[str]:
    """Session id to send to Slicer: explicit one, else this process' default."""
    global _default_session_used
    if session:
        return session
    if DEFAULT_SESSION:
        _default_session_used = True
    return DEFAULT_SESSION


def _slicer_request(method: str, path: str, session: Optional[str] = None,
                    params: Optional[dict] = None, **kwargs) -> requests.Response:
    """Send a request to the Slicer web server within a session."""
    params = dict(params or {})
    session_id = _resolve_session(session)
    if session_id:
        params["session"] = session_id
    url = f"{SLICER_WEB_SERVER_URL}{path}"
    return _http.request(method, url, params=params, **kwargs)


def _session_api(method: str, path: str = "", params: Optional[dict] = None) -> dict:
    url = f"{SLICER_WEB_SERVER_URL}/session{path}"
    try:
        response = _http.request(method, url, params=params or {})
        data = response.json()
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON response (HTTP {response.status_code}): {response.text[:200]}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    if response.status_code != 200:
        message = data.get("message", "") if isinstance(data, dict) else str(data)
        if "unknown command" in message:
            message = ("This Slicer WebServer does not support sessions; "
                       "requests from all agents share one namespace. ") + message
        return {"error": message}
    return data


@atexit.register
def _close_default_session() -> None:
    # Free the namespace of this process' auto-generated session. Nodes are kept
    # (so results stay visible in Slicer) but released from session ownership.
    if DEFAULT_SESSION and _DEFAULT_SESSION_IS_EPHEMERAL and _default_session_used:
        try:
            _http.delete(f"{SLICER_WEB_SERVER_URL}/session/{DEFAULT_SESSION}",
                         params={"removeNodes": "false"}, timeout=2)
        except Exception:
            pass


@mcp.tool()
def create_session(name: str = None, session_id: str = None) -> dict:
    """
    Create (or resume) an isolated Slicer session for one agent.

    Use this when several agents share this MCP server (e.g. sub-agents): each
    agent creates its own session and passes the returned "id" as the `session`
    argument of every other tool. Without `session`, tools use this MCP
    process' default session.

    Within a session, execute_python_code variables are private to the session,
    nodes created are tagged as owned by it, and deletes only affect its own
    nodes. The scene, loaded data and views are shared (not copied), so sessions
    are cheap and existing nodes can be read and reused by every session.

    Parameters:
        name: Optional human-readable name (e.g. the agent's task).
        session_id: Optional id to create or resume (1-64 chars of A-Z a-z 0-9 _ . -).
            Reusing an id resumes that session with its variables and nodes.

    Returns the session info: {"id", "name", "nodeIDs", "variables", ...} or {"error": ...}.
    """
    params = {}
    if name:
        params["name"] = name
    if session_id:
        params["id"] = session_id
    return _session_api("POST", params=params)


@mcp.tool()
def list_sessions() -> dict:
    """
    List the sessions active in Slicer and this MCP process' default session.

    Returns {"default_session": str|None, "sessions": [{"id", "name", "nodeCount", "variables", "lastAccess", ...}]}
    or {"error": ...}.
    """
    result = _session_api("GET")
    result["default_session"] = DEFAULT_SESSION
    return result


@mcp.tool()
def close_session(session: str = None, remove_nodes: bool = True) -> dict:
    """
    Close a session and free its Python namespace.

    Parameters:
        session: Session id to close (default: this MCP process' default session).
        remove_nodes: If True (default), remove the nodes this session created
            from the scene. If False, keep them in the scene for others to use.

    Returns {"success": True, "removedNodeIDs": [...]} or {"error": ...}.
    """
    session_id = session or DEFAULT_SESSION
    if not session_id:
        return {"error": "No session specified and sessions are disabled for this MCP process"}
    return _session_api("DELETE", f"/{session_id}",
                        params={"removeNodes": "true" if remove_nodes else "false"})


# Add list_nodes tool
@mcp.tool()
def list_nodes(filter_type: str = "names", class_name: str = None,
              name: str = None, id: str = None, session: str = None,
              only_session_nodes: bool = False) -> dict:
    """
    List MRML nodes via the Slicer Web Server API.

    The filter_type parameter specifies the type of node information to retrieve.
    Possible values include "names" (node names), "ids" (node IDs), and "properties" (node properties).
    The default value is "names".

    The class_name, name, and id parameters are optional and can be used to further filter nodes.
    The class_name parameter allows filtering nodes by class name.
    The name parameter allows filtering nodes by name.
    The id parameter allows filtering nodes by ID.
    The session parameter selects the session (default: this MCP process' default session).
    If only_session_nodes is True, only nodes created in that session are listed;
    otherwise all nodes of the shared scene are listed.

    Examples:
    - List the names of all nodes: {"tool": "list_nodes", "arguments": {"filter_type": "names"}}
    - List the IDs of nodes of a specific class: {"tool": "list_nodes", "arguments": {"filter_type": "ids", "class_name": "vtkMRMLModelNode"}}
    - List the properties of nodes with a specific name: {"tool": "list_nodes", "arguments": {"filter_type": "properties", "name": "MyModel"}}
    - List nodes with a specific ID: {"tool": "list_nodes", "arguments": {"filter_type": "ids", "id": "vtkMRMLModelNode123"}}

    Returns a dictionary containing node information.
    If filter_type is "names" or "ids", the returned dictionary contains a "nodes" key, whose value is a list containing node names or IDs.
    Example: {"nodes": ["node1", "node2", ...]} or {"nodes": ["id1", "id2", ...]}
    If filter_type is "properties", the returned dictionary contains a "nodes" key, whose value is a dictionary containing node properties.
    Example: {"nodes": {"node1": {"property1": "value1", "property2": "value2"}, ...}}
    If an error occurs, a dictionary containing an "error" key is returned, whose value is a string describing the error.
    """
    try:
        # Build API endpoint based on filter type
        endpoint_map = {
            "names": "/mrml/names",
            "ids": "/mrml/ids",
            "properties": "/mrml/properties"
        }
        
        if filter_type not in endpoint_map:
            return {"error": "Invalid filter_type specified"}
            
        # Build query parameters
        params = {}
        if only_session_nodes:
            params["scope"] = "session"
        if class_name:
            params["class"] = class_name
        if name:
            params["name"] = name
        if id:
            params["id"] = id

        # Send GET request to Slicer Web Server
        # Smart proxy handling: disable for localhost, use system default for others
        response = _slicer_request("GET", endpoint_map[filter_type], session=session, params=params)
        response.raise_for_status()
        
        # Process response based on filter type
        if filter_type == "properties":
            return {"nodes": response.json()}
            
        return {"nodes": response.json()}

    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error {e.response.status_code}: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON response: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Node listing failed: {str(e)}"}


# Add execute_python_code tool
@mcp.tool()
def execute_python_code(code: str, session: str = None) -> dict:
    """
    Execute Python code in 3D Slicer.

    Parameters:
    code (str): The Python code to execute.

    The code parameter is a string containing the Python code to be executed in 3D Slicer's Python environment.
    The code should be executable by Python's `exec()` function. To get return values, the code should assign the result to a variable named `__execResult`.

    session (str, optional): Session id (from create_session). Default: this MCP process' default session.
    Variables persist between calls of the same session and are invisible to other sessions.
    Nodes added to the scene are owned by the session; `__session__.nodes()` lists them.
    The scene is shared: reuse existing nodes (e.g. loaded volumes) instead of loading copies,
    and do not modify or remove nodes that other sessions own.

    Examples:
    - Create a sphere model: {"tool": "execute_python_code", "arguments": {"code": "sphere = slicer.vtkMRMLModelNode(); slicer.mrmlScene.AddNode(sphere); sphere.SetName('MySphere'); __execResult = sphere.GetID()"}}
    - Get the number of nodes in the current scene: {"tool": "execute_python_code", "arguments": {"code": "__execResult = len(slicer.mrmlScene.GetNodes())"}}
    - Calculate 1+1: {"tool": "execute_python_code", "arguments": {"code": "__execResult = 1 + 1"}}

    Returns:
        dict: A dictionary containing the execution result.

        If the code execution is successful, the dictionary will contain the following key-value pairs:
        - "success": True
        - "message": The result of the code execution. If the code assigns the result to `__execResult`, the value of `__execResult` is returned, otherwise it returns empty.

        If the code execution fails, the dictionary will contain the following key-value pairs:
        - "success": False
        - "message": A string containing an error message indicating the cause of the failure. The error message may come from the Slicer Web Server or the Python interpreter.

    Examples:
    - Successful execution: {"success": True, "message": 2}  # Assuming the result of 1+1 is 2
    - Successful execution: {"success": True, "message": "vtkMRMLScene1"} # Assuming the created sphere id is vtkMRMLScene1
    - Python execution error: {"success": False, "message": "Server error: name 'slicer' is not defined"}
    - Connection error: {"success": False, "message": "Connection error: ..."}
    - HTTP error: {"success": False, "message": "HTTP Error 404: Not Found"}
    """
    headers = {'Content-Type': 'text/plain'}
    try:
        response = _slicer_request("POST", "/exec", session=session, data=code.encode('utf-8'), headers=headers)
        result_data = response.json()
        
        if isinstance(result_data, dict) and not result_data.get("success", True):
            return {
                "success": False,
                "message": result_data.get("message", "Unknown Python execution error")
                }
            
        return {
            "success": True,
            "message": result_data.get("__execResult") if isinstance(result_data, dict) and "__execResult" in result_data else result_data
            }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "message": f"HTTP Error {e.response.status_code}: {str(e)}"
            }
    except json.JSONDecodeError:
        return {
            "success": False,
            "message": f"Invalid JSON response: {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Connection error: {str(e)}"
            }


# Add capture_screenshot tool
@mcp.tool()
def capture_screenshot(
    view_type: str = "application",
    view_name: str = None,
    camera_axis: str = None,
    image_size: int = None,
    session: str = None
) -> list:
    """Capture a screenshot from 3D Slicer for visual inspection.

    This MCP tool gives the model a "what you see is what you get"
    view of Slicer, so it can reason about the current GUI state
    (panels, slice views, 3D rendering, overlays, etc.) and take
    follow‑up actions.

    Behavior by ``view_type``:

    - "application":
        Capture the full Slicer application window, including all
        dock widgets, slice views and 3D views, as a compressed JPEG.

    - "slice":
        Capture a single slice view (Red/Yellow/Green) as it currently
        appears on screen. The screenshot includes whatever UI elements
        are visible in that view (crosshair, orientation markers,
        labels, segment overlays, etc.).

    - "3d":
        Capture the 3D rendering view. Optionally, a camera axis can
        be specified to look from a standard anatomical direction.

    Parameters:
        view_type (str): Type of screenshot to capture. Options:
            - "application" (default): Full application window.
            - "slice": One named slice view.
            - "3d": The 3D rendering view.

        view_name (str): For slice views, which slice to capture.
            Options: "red", "yellow", "green". Only used when
            ``view_type="slice"``.

        camera_axis (str): For 3D views, camera view direction.
            Options: "L" (Left), "R" (Right), "A" (Anterior),
            "P" (Posterior), "I" (Inferior), "S" (Superior).
            Only used when ``view_type="3d"``.

        session (Optional[str]): Session id (default: this MCP process'
            default session). Note that views are shared by all sessions.

        image_size (Optional[int]): Target pixel size for the output image.
            - For slice views, this is treated as a maximum for the
              longer side of the captured pixmap; the image is scaled
              with preserved aspect ratio up to this size.
            - For application and 3D views, the screenshot is captured
              at the current window size and may be downscaled if its
              longest side exceeds an internal maximum (for size
              control). Callers should not rely on an exact pixel
              size, only that larger values produce higher‑resolution
              images within reasonable limits.

    Returns:
        list[mcp.types.Content]:
            - ``TextContent``: A short, human‑readable description of
                what was captured (e.g. which view and approximate size).
            - ``ImageContent``: The screenshot image in base64‑encoded
                JPEG format when capture succeeds, or omitted if only an
                error message is returned.

    Note:
        This tool is primarily intended for multimodal models. Callers
        should inspect the returned screenshot with their vision
        capabilities (rather than relying only on the text summary)
        when reasoning about Slicer's state, UI layout, overlays, or
        other visual details.

        The internal capture mechanism (e.g. which Slicer APIs or
        endpoints are used) is considered an implementation detail and may
        evolve over time without breaking this public contract.
    """
    try:
        # Determine behavior based on view_type
        if view_type == "application":
            api_path = "/screenshot"
            params = {}
            description = "full application window"
            
        elif view_type == "slice":
            if not view_name:
                return [TextContent(type="text", text="Error: view_name is required for slice screenshots (red, yellow, or green)")]

            # Use internal /exec-based slice screenshot implementation to
            # capture the render window with full GUI elements.
            python_code = _build_slice_screenshot_python_code(
                view_name=view_name,
                image_size=image_size,
            )
            result = execute_python_code(python_code, session=session)
            return _parse_slice_screenshot_result(result)
                
        elif view_type == "3d":
            api_path = "/threeD"
            params = {}
            description = "3D view"
            
            # Add optional parameters
            if camera_axis:
                params["lookFromAxis"] = camera_axis.upper()
                description += f" from {camera_axis} axis"
                
        else:
            return [TextContent(type="text", text=f"Error: Invalid view_type '{view_type}'. Must be 'application', 'slice', or '3d'")]

        # Make the request to Slicer Web Server
        # Smart proxy handling: disable for localhost, use system default for others
        response = _slicer_request("GET", api_path, session=session, params=params)
        response.raise_for_status()
        
        # Check if response is an image
        if response.headers.get('Content-Type', '').startswith('image/'):
            # Optimize image to reduce size for multimodal model context limits
            try:
                # Open image from response bytes
                img = Image.open(BytesIO(response.content))
                
                # Convert RGBA to RGB if necessary (for JPEG compatibility)
                if img.mode == 'RGBA':
                    # Create white background
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = rgb_img
                elif img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if image is too large (max dimension 1920px)
                max_dimension = 1920
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Compress image to JPEG with quality 85
                output_buffer = BytesIO()
                img.save(output_buffer, format='JPEG', quality=85, optimize=True)
                compressed_bytes = output_buffer.getvalue()
                
                # Convert to base64
                image_base64 = base64.b64encode(compressed_bytes).decode('utf-8')
                
                # Return using MCP's content types
                return [
                    TextContent(
                        type="text",
                        text=f"Screenshot of {description} captured successfully (compressed)"
                    ),
                    ImageContent(
                        type="image",
                        data=image_base64,
                        mime_type="image/jpeg"
                    )
                ]
            except Exception as img_error:
                # Fallback to original image if compression fails
                original_image_base64 = base64.b64encode(response.content).decode('utf-8')
                original_mime_type = response.headers.get('Content-Type', 'image/png')
                return [
                    TextContent(
                        type="text",
                        text=f"Screenshot captured but compression failed: {str(img_error)}. Using original image."
                    ),
                    ImageContent(
                        type="image",
                        data=original_image_base64,
                        mime_type=original_mime_type
                    )
                ]
        else:
            # Unexpected response type
            return [TextContent(type="text", text=f"Error: Unexpected response type: {response.headers.get('Content-Type')}")]
            
    except requests.exceptions.HTTPError as e:
        return [TextContent(type="text", text=f"HTTP Error {e.response.status_code}: {str(e)}")]
    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=f"Connection error: {str(e)}. Make sure Slicer Web Server is running.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


def _build_slice_screenshot_python_code(
    view_name: str,
    image_size: int = None,
) -> str:
    """Build Python code to capture a slice view screenshot via ``/exec``.

    The generated code runs inside 3D Slicer using the Web Server ``/exec`` API.
    It locates the requested slice widget (Red/Yellow/Green), forces a render,
    grabs the corresponding ``sliceView`` Qt widget, optionally rescales the
    pixmap, and returns a JSON object containing a base64-encoded JPEG image and
    basic metadata.

    The code does **not** modify any slice offsets, orientations, crosshair
    state, or orientation markers; it simply captures the current visual state
    of the slice view.

    Args:
        view_name: Name of the slice view ("red", "yellow", or "green").
        image_size: Optional maximum size (in pixels) for the longer image
            dimension. When provided, the pixmap is scaled with preserved
            aspect ratio so that the longer side equals ``image_size``.

    Returns:
        A Python code string suitable for sending to the Slicer Web Server
        ``/exec`` endpoint.
    """
    # Prepare optional parameter code (only resize is handled here)
    resize_code = ""
    if image_size:
        resize_code = f"""
        # Resize image
        pixmap = pixmap.scaled(
            {image_size}, {image_size},
            qt.Qt.KeepAspectRatio,
            qt.Qt.SmoothTransformation
        )"""
    
    # Build complete Python code
    python_code = f"""
import qt
import base64
import json

try:
    # Get slice widget
    layoutManager = slicer.app.layoutManager()
    sliceWidget = layoutManager.sliceWidget('{view_name.capitalize()}')
    
    if not sliceWidget:
        __execResult = json.dumps({{"error": f"Slice widget '{view_name.capitalize()}' not found"}})
    else:
        # Force render update
        sliceWidget.sliceView().forceRender()
        slicer.app.processEvents()
        
        # Capture slice view
        sliceView = sliceWidget.sliceView()
        pixmap = sliceView.grab(){resize_code}
        
        # Convert to QImage
        qimage = pixmap.toImage()
        
        # Save as JPEG to reduce size
        buffer = qt.QBuffer()
        buffer.open(qt.QIODevice.WriteOnly)
        qimage.save(buffer, "JPEG", 85)
        
        # Base64 encode - convert QByteArray to bytes properly
        byte_array = buffer.data()
        image_bytes = byte_array.data()  # Extract raw bytes from QByteArray
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Return result
        __execResult = json.dumps({{
            "success": True,
            "image": image_base64,
            "format": "jpeg",
            "view": "{view_name}",
            "size": {{"width": qimage.width(), "height": qimage.height()}}
        }})
        
except Exception as e:
    import traceback
    __execResult = json.dumps({{
        "error": str(e),
        "traceback": traceback.format_exc()
    }})
"""
    return python_code


def _parse_slice_screenshot_result(result: dict) -> list:
    """
    Parse slice screenshot execution result.
    
    Args:
        result: Dictionary returned from execute_python_code
    
    Returns:
        List of MCP content items (TextContent and/or ImageContent)
    """
    if not result["success"]:
        return [TextContent(
            type="text",
            text=f"Code execution failed: {result['message']}"
        )]
    
    try:
        # Parse JSON response
        data = json.loads(result["message"])
        
        if "error" in data:
            error_msg = f"Screenshot failed: {data['error']}"
            if "traceback" in data:
                error_msg += f"\n\nTraceback:\n{data['traceback']}"
            return [TextContent(type="text", text=error_msg)]
        
        # Extract image and metadata
        image_base64 = data["image"]
        metadata = data.get("size", {})
        
        # Build concise description text
        description = f"Successfully captured {data['view']} slice view "
        description += f"({metadata.get('width', '?')}x{metadata.get('height', '?')})"
        
        return [
            TextContent(type="text", text=description),
            ImageContent(
                type="image",
                data=image_base64,
                mime_type="image/jpeg"
            )
        ]
        
    except json.JSONDecodeError as e:
        return [TextContent(
            type="text",
            text=f"JSON parsing failed: {str(e)}\n\nResponse length: {len(result.get('message', ''))} characters"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Processing failed: {str(e)}"
        )]
