"""Tkinter GUI for planner sessions.

The UI handles chat-style messaging, user prompts, and optional camera/depth
preview panels during navigation.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
from typing import Optional, Callable
from datetime import datetime
from enum import Enum
import numpy as np

# PIL for image conversion (tkinter-compatible)
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Video display will be disabled.")

# OpenCV for frame processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Video display will be disabled.")


class MessageType(Enum):
    """Types of messages displayed in the GUI."""
    SYSTEM = "system"           # System status messages (navigation, scanning)
    ASSISTANT = "assistant"     # Messages from the drone assistant (VLM responses)
    USER = "user"               # User messages
    SUCCESS = "success"         # Success messages
    ERROR = "error"             # Error messages
    STATUS = "status"           # Status updates (brief, e.g., "Flying to Kitchen...")


class PlannerGUI:
    """
    Graphical User Interface for the VLM-based Planner.
    
    Provides a clean, chat-like interface for interacting with the drone
    planner without the clutter of debugging logs.
    
    Attributes:
        engine: The PlannerEngine instance to use
        root: The tkinter root window
        message_queue: Queue for thread-safe message passing
        input_queue: Queue for passing user input back to the executor
    """
    
    # UI color palette.
    COLORS = {
        'bg': '#0d0d0d',               # Deep black background
        'bg_secondary': '#1c1c1e',     # Slightly lighter (iOS dark mode)
        'bg_tertiary': '#2c2c2e',      # Card/elevated surface
        'text': '#ffffff',             # Pure white text
        'text_muted': '#8e8e93',       # iOS system gray
        'text_secondary': '#ebebf5',   # Secondary text (slightly dimmed)
        'accent': '#0a84ff',           # iOS blue
        'accent_hover': '#409cff',     # Lighter blue for hover
        'success': '#30d158',          # iOS green
        'error': '#ff453a',            # iOS red
        'warning': '#ff9f0a',          # iOS orange
        'user_bubble': '#0a84ff',      # Blue for user messages
        'assistant_bubble': '#2c2c2e', # Dark gray for assistant
        'status': '#64d2ff',           # iOS cyan
        'input_bg': '#1c1c1e',         # Input field background
        'input_border': '#3a3a3c',     # Subtle border
        'button_primary': '#0a84ff',   # Primary button (blue)
        'button_primary_hover': '#409cff',
        'button_secondary': '#2c2c2e', # Secondary button
        'button_secondary_hover': '#3a3a3c',
        'button_danger': '#ff453a',    # Danger button (red)
        'button_danger_hover': '#ff6961',
        'divider': '#3a3a3c',          # Divider line color
    }
    
    def __init__(self, engine=None):
        """
        Initialize the Planner GUI.
        
        Args:
            engine: PlannerEngine instance (can be set later via set_engine)
        """
        self.engine = engine
        self.root = None
        self.message_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.input_event = threading.Event()
        self.is_waiting_for_input = False
        self.current_prompt = ""
        self.session_active = False
        self.executor_thread = None
        
        # Widget references
        self.chat_display = None
        self.input_field = None
        self.send_button = None
        self.status_label = None
        self.start_button = None
        
        # Video display references
        self.video_label = None
        self.depth_label = None
        self.video_running = False
        self.video_status_label = None
        
    def set_engine(self, engine):
        """Set the PlannerEngine instance."""
        self.engine = engine
        
    def _setup_window(self):
        """Set up the main window with side-by-side chat/video layout."""
        self.root = tk.Tk()
        self.root.title("Drone Buddy - Object Finder")
        self.root.geometry("1400x900")  # Fallback size
        self.root.minsize(1200, 750)
        self.root.configure(bg=self.COLORS['bg'])
        
        # Start maximized/fullscreen on Windows
        self.root.state('zoomed')
        
        # Configure grid weights for responsiveness - 50%/50% split
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)  # Chat column (50%)
        self.root.grid_columnconfigure(1, weight=1)  # Video column (50%)
        
        # Try to set a dark title bar on Windows.
        try:
            from ctypes import windll, byref, sizeof, c_int
            HWND = windll.user32.GetParent(self.root.winfo_id())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            windll.dwmapi.DwmSetWindowAttribute(
                HWND, DWMWA_USE_IMMERSIVE_DARK_MODE, 
                byref(c_int(1)), sizeof(c_int)
            )
        except:
            pass
        
        self._create_header()
        self._create_main_content()  # New: creates side-by-side layout
        self._create_status_bar()
        
    def _create_main_content(self):
        """Create the main content area with chat on left and video on right."""
        # Main container frame
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        main_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=8, pady=8)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)  # Chat (50%)
        main_frame.grid_columnconfigure(1, weight=1)  # Video (50%)
        
        # Left side: Chat + Input
        left_frame = tk.Frame(main_frame, bg=self.COLORS['bg'])
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        left_frame.grid_rowconfigure(0, weight=1)  # Chat area expands
        left_frame.grid_rowconfigure(1, weight=0)  # Input area fixed
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Chat area
        self._create_chat_area(left_frame)
        
        # Input area
        self._create_input_area(left_frame)
        
        # Right side: Video display
        self._create_video_area(main_frame)
        
    def _create_video_area(self, parent):
        """Create the video display area showing camera feed and depth map."""
        # Video container frame
        video_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], padx=8, pady=8)
        video_frame.grid(row=0, column=1, sticky='nsew')
        video_frame.grid_rowconfigure(1, weight=1)  # Camera row
        video_frame.grid_rowconfigure(3, weight=1)  # Depth row
        video_frame.grid_columnconfigure(0, weight=1)
        
        # Video title
        video_title = tk.Label(
            video_frame,
            text="📹 Drone Camera",
            font=('Segoe UI Semibold', 12),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text']
        )
        video_title.grid(row=0, column=0, pady=(0, 4), sticky='w')
        
        # Camera feed label - use fixed aspect ratio container
        # Each panel gets equal height (window_height - header - footer) / 2
        video_display_width = 480
        video_display_height = 270  # Reduced to fit both panels (16:9 aspect ratio)
        
        self.video_label = tk.Label(
            video_frame,
            bg=self.COLORS['bg_tertiary'],
            width=video_display_width,
            height=video_display_height
        )
        self.video_label.grid(row=1, column=0, pady=(0, 8), sticky='nsew')
        
        # Set placeholder image
        self._set_video_placeholder(self.video_label, "Camera Feed", "Waiting for drone connection...", 
                                     video_display_width, video_display_height)
        
        # Depth map title
        depth_title = tk.Label(
            video_frame,
            text="🌊 Depth Map",
            font=('Segoe UI Semibold', 12),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text']
        )
        depth_title.grid(row=2, column=0, pady=(8, 4), sticky='w')
        
        # Depth map label - same size as camera
        self.depth_label = tk.Label(
            video_frame,
            bg=self.COLORS['bg_tertiary'],
            width=video_display_width,
            height=video_display_height
        )
        self.depth_label.grid(row=3, column=0, sticky='nsew')
        
        # Set placeholder for depth
        self._set_video_placeholder(self.depth_label, "Depth Map", "No MiDaS model loaded",
                                     video_display_width, video_display_height)
        
        # Status label for video
        self.video_status_label = tk.Label(
            video_frame,
            text="● Disconnected",
            font=('Segoe UI', 9),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted']
        )
        self.video_status_label.grid(row=4, column=0, pady=(8, 0), sticky='w')
    
    def _set_video_placeholder(self, label, title: str, subtitle: str, width: int = 480, height: int = 270):
        """Set a placeholder image on a video label."""
        if not PIL_AVAILABLE or not CV2_AVAILABLE:
            label.config(text=f"{title}\n{subtitle}", fg=self.COLORS['text_muted'])
            return
            
        # Create placeholder frame
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        placeholder[:] = (30, 30, 30)  # Dark gray background (BGR)
        
        # Add text - center based on size
        text_x = width // 2 - 80
        text_y = height // 2 - 10
        cv2.putText(placeholder, title, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        cv2.putText(placeholder, subtitle, (text_x - 40, text_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        # Convert to tkinter-compatible image
        img_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        label.config(image=img_tk)
        label.image = img_tk  # Keep reference to prevent garbage collection
        
    def _create_header(self):
        """Create the header section with Apple-inspired styling."""
        header_frame = tk.Frame(self.root, bg=self.COLORS['bg_secondary'], pady=16)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew')
        
        # Container for centered content
        content_frame = tk.Frame(header_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack()
        
        # Icon and Title row
        title_row = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        title_row.pack()
        
        # Title with SF Pro-like font
        title_label = tk.Label(
            title_row,
            text="Drone Object Finder",
            font=('Segoe UI Semibold', 22),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text']
        )
        title_label.pack()
        
        # Subtitle with muted color
        subtitle_label = tk.Label(
            content_frame,
            text="Powered by Vision Language Models",
            font=('Segoe UI', 11),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted']
        )
        subtitle_label.pack(pady=(4, 0))
        
        # Subtle separator line
        separator = tk.Frame(self.root, bg=self.COLORS['divider'], height=1)
        separator.grid(row=0, column=0, sticky='sew', pady=(0, 0))
        
    def _create_chat_area(self, parent=None):
        """Create the main chat/message display area."""
        if parent is None:
            parent = self.root
            
        # Outer frame with padding
        outer_frame = tk.Frame(parent, bg=self.COLORS['bg'], padx=8, pady=8)
        outer_frame.grid(row=0, column=0, sticky='nsew')
        outer_frame.grid_rowconfigure(0, weight=1)
        outer_frame.grid_columnconfigure(0, weight=1)
        
        # Inner card-like frame
        chat_frame = tk.Frame(outer_frame, bg=self.COLORS['bg_tertiary'], padx=2, pady=2)
        chat_frame.grid(row=0, column=0, sticky='nsew')
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text'],
            insertbackground=self.COLORS['text'],
            selectbackground=self.COLORS['accent'],
            relief=tk.FLAT,
            padx=16,
            pady=16,
            state=tk.DISABLED,
            cursor='arrow',
            borderwidth=0,
            highlightthickness=0
        )
        self.chat_display.grid(row=0, column=0, sticky='nsew')
        
        # Configure scrollbar styling (limited in tkinter)
        self.chat_display.vbar.configure(
            bg=self.COLORS['bg_secondary'],
            troughcolor=self.COLORS['bg_tertiary'],
            activebackground=self.COLORS['text_muted']
        )
        
        # Configure tags for different message types with improved styling
        self.chat_display.tag_configure(
            'system', 
            foreground=self.COLORS['text_muted'],
            font=('Segoe UI', 10),
            spacing1=6, spacing3=6,
            lmargin1=8, lmargin2=8
        )
        self.chat_display.tag_configure(
            'assistant', 
            foreground=self.COLORS['text_secondary'],
            font=('Segoe UI', 11),
            lmargin1=24, lmargin2=24,
            spacing1=4, spacing3=8
        )
        self.chat_display.tag_configure(
            'assistant_label',
            foreground=self.COLORS['accent'],
            font=('Segoe UI Semibold', 10),
            spacing1=14
        )
        self.chat_display.tag_configure(
            'user',
            foreground=self.COLORS['text'],
            font=('Segoe UI', 11),
            lmargin1=24, lmargin2=24,
            spacing1=4, spacing3=8
        )
        self.chat_display.tag_configure(
            'user_label',
            foreground=self.COLORS['warning'],
            font=('Segoe UI Semibold', 10),
            spacing1=14
        )
        self.chat_display.tag_configure(
            'success',
            foreground=self.COLORS['success'],
            font=('Segoe UI Semibold', 11),
            spacing1=12, spacing3=12,
            lmargin1=8
        )
        self.chat_display.tag_configure(
            'error',
            foreground=self.COLORS['error'],
            font=('Segoe UI', 11),
            spacing1=6, spacing3=6,
            lmargin1=8
        )
        self.chat_display.tag_configure(
            'status',
            foreground=self.COLORS['status'],
            font=('Segoe UI', 10),
            spacing1=2, spacing3=2,
            lmargin1=32
        )
        self.chat_display.tag_configure(
            'divider',
            foreground=self.COLORS['divider'],
            font=('Segoe UI', 9),
            justify='center',
            spacing1=20, spacing3=20
        )
        
    def _create_input_area(self, parent=None):
        """Create the input area with refined Apple-style buttons."""
        if parent is None:
            parent = self.root
            
        input_frame = tk.Frame(parent, bg=self.COLORS['bg'], padx=8, pady=8)
        input_frame.grid(row=1, column=0, sticky='ew')
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Prompt label with refined styling
        self.prompt_label = tk.Label(
            input_frame,
            text="Enter what you'd like the drone to find:",
            font=('Segoe UI', 10),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text_muted'],
            anchor='w'
        )
        self.prompt_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))
        
        # Input container with border effect
        input_container = tk.Frame(input_frame, bg=self.COLORS['input_border'], padx=1, pady=1)
        input_container.grid(row=1, column=0, sticky='ew', padx=(0, 12))
        input_container.grid_columnconfigure(0, weight=1)
        
        # Input field with refined styling
        self.input_field = tk.Entry(
            input_container,
            font=('Segoe UI', 12),
            bg=self.COLORS['input_bg'],
            fg=self.COLORS['text'],
            insertbackground=self.COLORS['accent'],
            relief=tk.FLAT,
            disabledbackground=self.COLORS['bg_secondary'],
            disabledforeground=self.COLORS['text_muted'],
            highlightthickness=0,
            borderwidth=0
        )
        self.input_field.grid(row=0, column=0, sticky='ew', ipady=12, ipadx=12)
        self.input_field.bind('<Return>', self._on_send)
        
        # Focus effects for input field
        def on_focus_in(e):
            input_container.configure(bg=self.COLORS['accent'])
        def on_focus_out(e):
            input_container.configure(bg=self.COLORS['input_border'])
        self.input_field.bind('<FocusIn>', on_focus_in)
        self.input_field.bind('<FocusOut>', on_focus_out)
        
        # Send button with Apple-style
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            font=('Segoe UI Semibold', 11),
            bg=self.COLORS['button_primary'],
            fg='#ffffff',
            activebackground=self.COLORS['button_primary_hover'],
            activeforeground='#ffffff',
            relief=tk.FLAT,
            padx=24,
            pady=10,
            command=self._on_send,
            cursor='hand2',
            borderwidth=0,
            highlightthickness=0
        )
        self.send_button.grid(row=1, column=1, sticky='e')
        
        # Button hover effects
        def send_hover_enter(e):
            if self.send_button['state'] != 'disabled':
                self.send_button.configure(bg=self.COLORS['button_primary_hover'])
        def send_hover_leave(e):
            if self.send_button['state'] != 'disabled':
                self.send_button.configure(bg=self.COLORS['button_primary'])
        self.send_button.bind('<Enter>', send_hover_enter)
        self.send_button.bind('<Leave>', send_hover_leave)
        
        # Button row
        button_frame = tk.Frame(input_frame, bg=self.COLORS['bg'])
        button_frame.grid(row=2, column=0, columnspan=2, pady=(16, 0), sticky='ew')
        
        # Start new search button (secondary style)
        self.start_button = tk.Button(
            button_frame,
            text="🔍  New Search",
            font=('Segoe UI Semibold', 10),
            bg=self.COLORS['button_secondary'],
            fg=self.COLORS['text'],
            activebackground=self.COLORS['button_secondary_hover'],
            activeforeground=self.COLORS['text'],
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=self._start_search_mode,
            cursor='hand2',
            borderwidth=0,
            highlightthickness=0
        )
        self.start_button.pack(side='left')
        
        # Hover effect for start button
        def start_hover_enter(e):
            if self.start_button['state'] != 'disabled':
                self.start_button.configure(bg=self.COLORS['button_secondary_hover'])
        def start_hover_leave(e):
            if self.start_button['state'] != 'disabled':
                self.start_button.configure(bg=self.COLORS['button_secondary'])
        self.start_button.bind('<Enter>', start_hover_enter)
        self.start_button.bind('<Leave>', start_hover_leave)
        
        # Exit button (danger style)
        exit_button = tk.Button(
            button_frame,
            text="Exit",
            font=('Segoe UI Semibold', 10),
            bg=self.COLORS['button_danger'],
            fg='#ffffff',
            activebackground=self.COLORS['button_danger_hover'],
            activeforeground='#ffffff',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=self._on_exit,
            cursor='hand2',
            borderwidth=0,
            highlightthickness=0
        )
        exit_button.pack(side='right')
        
        # Hover effect for exit button
        def exit_hover_enter(e):
            exit_button.configure(bg=self.COLORS['button_danger_hover'])
        def exit_hover_leave(e):
            exit_button.configure(bg=self.COLORS['button_danger'])
        exit_button.bind('<Enter>', exit_hover_enter)
        exit_button.bind('<Leave>', exit_hover_leave)
        
    def _create_status_bar(self):
        """Create a refined status bar at the bottom."""
        # Separator line
        separator = tk.Frame(self.root, bg=self.COLORS['divider'], height=1)
        separator.grid(row=2, column=0, columnspan=2, sticky='ew')
        
        status_frame = tk.Frame(self.root, bg=self.COLORS['bg_secondary'], pady=8)
        status_frame.grid(row=3, column=0, columnspan=2, sticky='ew')
        
        # Status indicator dot and text
        status_container = tk.Frame(status_frame, bg=self.COLORS['bg_secondary'])
        status_container.pack()
        
        self.status_dot = tk.Label(
            status_container,
            text="●",
            font=('Segoe UI', 8),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['success']
        )
        self.status_dot.pack(side='left', padx=(0, 6))
        
        self.status_label = tk.Label(
            status_container,
            text="Ready",
            font=('Segoe UI', 10),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted']
        )
        self.status_label.pack(side='left')
        
    def _append_message(self, text: str, msg_type: MessageType):
        """Append a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        if msg_type == MessageType.ASSISTANT:
            self.chat_display.insert(tk.END, "🤖 Drone Assistant\n", 'assistant_label')
            self.chat_display.insert(tk.END, f"{text}\n", 'assistant')
        elif msg_type == MessageType.USER:
            self.chat_display.insert(tk.END, "You\n", 'user_label')
            self.chat_display.insert(tk.END, f"{text}\n", 'user')
        elif msg_type == MessageType.SUCCESS:
            self.chat_display.insert(tk.END, f"✅ {text}\n", 'success')
        elif msg_type == MessageType.ERROR:
            self.chat_display.insert(tk.END, f"❌ {text}\n", 'error')
        elif msg_type == MessageType.STATUS:
            self.chat_display.insert(tk.END, f"  ↳ {text}\n", 'status')
        elif msg_type == MessageType.SYSTEM:
            self.chat_display.insert(tk.END, f"{text}\n", 'system')
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def _add_divider(self, text: str = ""):
        """Add a refined visual divider to the chat."""
        self.chat_display.config(state=tk.NORMAL)
        if text:
            self.chat_display.insert(tk.END, f"\n{'─' * 15}  {text}  {'─' * 15}\n\n", 'divider')
        else:
            self.chat_display.insert(tk.END, f"\n{'─' * 40}\n\n", 'divider')
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def _set_status(self, text: str):
        """Update the status bar text and indicator."""
        if self.status_label:
            self.status_label.config(text=text)
            # Update status dot color based on state
            if hasattr(self, 'status_dot'):
                if 'progress' in text.lower() or 'searching' in text.lower():
                    self.status_dot.config(fg=self.COLORS['warning'])
                elif 'error' in text.lower():
                    self.status_dot.config(fg=self.COLORS['error'])
                else:
                    self.status_dot.config(fg=self.COLORS['success'])
            
    def _enable_input(self, prompt: str = "Type your message...", show_in_chat: bool = True):
        """Enable the input field and set prompt.
        
        Args:
            prompt: The prompt text to display
            show_in_chat: Not used anymore - prompts are sent by executor before callback
        """
        self.is_waiting_for_input = True
        self.current_prompt = prompt
        
        # Always show generic message in label area for confirmation prompts
        # The actual prompt is shown in chat by the executor before calling callback
        if 'waiting' in prompt.lower() or 'what would you like' in prompt.lower():
            self.prompt_label.config(text=prompt)
        else:
            # For confirmation prompts, show generic message in label area
            self.prompt_label.config(text="Enter your response below:")
        
        self.input_field.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.input_field.focus_set()
        
        # DON'T show prompt in chat here - it causes ordering issues
        # The executor sends the prompt message BEFORE calling user_input_callback
        
    def _disable_input(self):
        """Disable the input field."""
        self.is_waiting_for_input = False
        self.prompt_label.config(text="Waiting for drone operations...")
        self.input_field.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        
    def _on_send(self, event=None):
        """Handle send button click or Enter key."""
        if not self.is_waiting_for_input:
            return
            
        text = self.input_field.get().strip()
        if not text:
            return
            
        # Display user message
        self._append_message(text, MessageType.USER)
        
        # Clear input
        self.input_field.delete(0, tk.END)
        
        # If waiting for input in executor thread, send the response
        self.input_queue.put(text)
        self.input_event.set()
        self._disable_input()
        
    def _start_search_mode(self):
        """Enable input for starting a new search."""
        if self.session_active:
            messagebox.showwarning(
                "Session Active", 
                "A search session is currently active. Please wait for it to complete."
            )
            return
            
        self._add_divider("New Search")
        self._enable_input("What would you like the drone to find?", show_in_chat=False)
        self.start_button.config(state=tk.DISABLED)
        
        # Wait for user input, then start the search
        def wait_and_search():
            self.input_event.wait()
            self.input_event.clear()
            
            try:
                user_request = self.input_queue.get_nowait()
                if user_request:
                    self._execute_search(user_request)
            except queue.Empty:
                pass
            finally:
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
                
        threading.Thread(target=wait_and_search, daemon=True).start()
        
    def _execute_search(self, user_request: str):
        """Execute a search in a background thread."""
        if not self.engine:
            self._append_message("Error: No engine configured", MessageType.ERROR)
            return
            
        self.session_active = True
        self._set_status("Search in progress...")
        
        def run_search():
            try:
                result = self.engine.find_object(user_request)
                
                # Display result summary
                self.root.after(0, lambda: self._display_result(result))
                
            except Exception as e:
                self.root.after(0, lambda: self._append_message(
                    f"Search failed: {str(e)}", MessageType.ERROR
                ))
            finally:
                self.session_active = False
                self.root.after(0, lambda: self._set_status("Ready"))
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
                
        self.executor_thread = threading.Thread(target=run_search, daemon=True)
        self.executor_thread.start()
        
    def _display_result(self, result):
        """Display the session result summary."""
        self._add_divider("Session Complete")
        
        if result.success:
            self._append_message(
                f"Successfully found '{result.target_object}' at {result.found_at_waypoint}!",
                MessageType.SUCCESS
            )
        else:
            self._append_message(
                f"Could not find '{result.target_object}'",
                MessageType.ERROR
            )
            if result.error_message:
                self._append_message(f"Reason: {result.error_message}", MessageType.SYSTEM)
                
        # Statistics
        duration_str = f"{result.session_duration:.1f} seconds"
        if result.round_durations and len(result.round_durations) > 1:
            round_strs = [f"Round {i+1}: {d:.1f}s" for i, d in enumerate(result.round_durations)]
            duration_str += f"  ({', '.join(round_strs)})"
        stats = (
            f"Session Statistics:\n"
            f"  • Waypoints visited: {', '.join(result.waypoints_visited) or 'None'}\n"
            f"  • Scans performed: {result.scans_performed}\n"
            f"  • Duration: {duration_str}"
        )
        self._append_message(stats, MessageType.SYSTEM)
        
    def _on_exit(self):
        """Handle exit button click."""
        if self.session_active:
            if not messagebox.askyesno(
                "Confirm Exit",
                "A search session is active. Are you sure you want to exit?"
            ):
                return
        self.root.quit()
        self.root.destroy()
        
    def _process_message_queue(self):
        """Process messages from the queue (called periodically from main thread)."""
        try:
            while True:
                msg_type, text = self.message_queue.get_nowait()
                self._append_message(text, msg_type)
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            if self.root:
                self.root.after(100, self._process_message_queue)
                
    def get_user_input_callback(self) -> Callable[[str], str]:
        """
        Get a callback function for user input that works with the GUI.
        
        This callback is passed to the PlannerExecutor to handle user input
        through the GUI instead of console input().
        
        Returns:
            Callback function that prompts user via GUI and returns their response
        """
        def gui_input_callback(prompt: str) -> str:
            """Get user input through the GUI."""
            # Schedule input enabling on main thread
            self.root.after(0, lambda: self._enable_input(prompt.strip()))
            
            # Wait for user input
            self.input_event.wait()
            self.input_event.clear()
            
            try:
                return self.input_queue.get_nowait()
            except queue.Empty:
                return ""
                
        return gui_input_callback
    
    def get_message_callback(self) -> Callable[[str], None]:
        """
        Get a callback function for displaying messages from the executor.
        
        This callback replaces the executor's print statements to route
        messages through the GUI.
        
        Returns:
            Callback function that displays messages in the GUI
        """
        def gui_message_callback(message: str):
            """Display a message in the GUI."""
            # Parse message to determine type based on content
            msg_type = MessageType.ASSISTANT
            message_stripped = message.strip()
            
            # Check for explicit [STATUS] prefix (forced STATUS type)
            if message_stripped.startswith('[STATUS]'):
                msg_type = MessageType.STATUS
                message_stripped = message_stripped[8:]  # Remove the prefix
                self.message_queue.put((msg_type, message_stripped))
                return
            
            # Simple content-based classification
            message_lower = message_stripped.lower()
            
            # STATUS type: ONLY for brief action updates (light blue, smaller)
            # These are single-line action status updates during execution
            status_phrases = [
                'flying to ',       # "Flying to kitchen..."
                'navigating to ',   # "Navigating to bathroom..."
                'taking off',       # "Taking off..."
                'returning to start',  # "Returning to start position..."
                'landed safely',    # "Landed safely at start position."
                'scanning area',    # "Scanning area at kitchen..."
                'drone has already landed',  # "Drone has already landed safely."
                'path blocked by obstacle',  # Obstacle timeout messages
                'could not reach start',     # Failed to reach start
            ]
            
            # Only classify as STATUS if it starts with or is a brief action phrase
            # Avoid classifying multi-line messages or VLM plans as STATUS
            is_single_line = '\n' not in message_stripped
            is_short_message = len(message_stripped) < 150  # Slightly longer for obstacle messages
            matches_status_phrase = any(phrase in message_lower for phrase in status_phrases)
            
            if is_single_line and is_short_message and matches_status_phrase:
                msg_type = MessageType.STATUS
            elif 'error' in message_lower or 'failed' in message_lower:
                msg_type = MessageType.ERROR
            elif ('success' in message_lower or 'found' in message_lower) and '!' in message:
                msg_type = MessageType.SUCCESS
            # Everything else (including VLM plans, reasoning, etc.) is ASSISTANT
                
            self.message_queue.put((msg_type, message_stripped))
            
        return gui_message_callback
    
    def start_video_display(self):
        """Mark video display as ready."""
        if not PIL_AVAILABLE or not CV2_AVAILABLE:
            return
        self.video_running = True
        
    def stop_video_display(self):
        """Stop the video display update loop."""
        self.video_running = False
        
    def set_frame_source(self, nav_manager):
        """
        Set the navigation manager as frame source and register callback.
        
        The navigation manager will send frames via callback instead of using cv2.imshow.
        
        Args:
            nav_manager: WaypointNavigationManager instance
        """
        if nav_manager is not None:
            # Register our callback with the navigation manager
            nav_manager.set_external_frame_callback(self._on_navigation_frame)
            
            # Update status
            if self.video_status_label:
                self.root.after(0, lambda: self.video_status_label.config(
                    text="● Connected",
                    fg=self.COLORS['success']
                ))
    
    def _on_navigation_frame(self, camera_frame_bgr, depth_frame_bgr):
        """
        Callback to receive frames from navigation manager.
        
        Called by WaypointNavigationManager._video_display_loop() when 
        external_frame_callback is set. Frames are already processed and 
        ready for display (BGR format).
        
        Args:
            camera_frame_bgr: Processed camera frame (BGR)
            depth_frame_bgr: Processed depth frame (BGR)
        """
        if not self.video_running or not self.root or not PIL_AVAILABLE:
            return
            
        try:
            # Convert BGR to RGB for PIL/tkinter
            camera_rgb = cv2.cvtColor(camera_frame_bgr, cv2.COLOR_BGR2RGB)
            depth_rgb = cv2.cvtColor(depth_frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize for GUI display - must match panel size (480x270)
            display_width = 480
            display_height = 270
            
            camera_resized = cv2.resize(camera_rgb, (display_width, display_height))
            depth_resized = cv2.resize(depth_rgb, (display_width, display_height))
            
            # Schedule GUI update on main thread
            def update_gui():
                try:
                    if not self.video_running or not self.root:
                        return
                        
                    # Camera frame
                    img_camera = Image.fromarray(camera_resized)
                    img_camera_tk = ImageTk.PhotoImage(image=img_camera)
                    if self.video_label:
                        self.video_label.config(image=img_camera_tk)
                        self.video_label.image = img_camera_tk
                    
                    # Depth frame
                    img_depth = Image.fromarray(depth_resized)
                    img_depth_tk = ImageTk.PhotoImage(image=img_depth)
                    if self.depth_label:
                        self.depth_label.config(image=img_depth_tk)
                        self.depth_label.image = img_depth_tk
                except:
                    pass  # Silent fail for GUI updates
            
            self.root.after(0, update_gui)
            
        except Exception as e:
            pass  # Silent fail for frame processing
        
    def run(self):
        """
        Run the GUI application.
        
        This starts the tkinter main loop and displays the interface.
        """
        self._setup_window()
        
        # Initial welcome message
        self._append_message(
            "Welcome to the Drone Object Finder!\n"
            "Type what you'd like me to find and press Enter or click Send.",
            MessageType.ASSISTANT
        )
        
        if not self.engine:
            self._append_message(
                "Note: No engine configured. Set the engine before starting a search.",
                MessageType.SYSTEM
            )
        
        # Enable input immediately so user can start searching right away
        self._enable_input("What would you like the drone to find?", show_in_chat=False)
        self._start_initial_search_listener()
            
        # Start message queue processing
        self.root.after(100, self._process_message_queue)
        
        # Start video display (will show placeholders until drone connects)
        self.start_video_display()
        
        # Run main loop
        self.root.mainloop()
        
        # Cleanup on exit
        self.stop_video_display()
    
    def _start_initial_search_listener(self):
        """Start listening for the initial search input."""
        def wait_and_search():
            self.input_event.wait()
            self.input_event.clear()
            
            try:
                user_request = self.input_queue.get_nowait()
                if user_request:
                    self._execute_search(user_request)
            except queue.Empty:
                pass
            finally:
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
                
        threading.Thread(target=wait_and_search, daemon=True).start()


class PlannerGUIApp:
    """
    Complete GUI Application wrapper that handles engine setup and GUI integration.
    
    This class provides a simpler interface for launching the GUI with
    proper callback integration.
    
    Example:
        from dronebuddylib import EngineConfigurations, AtomicEngineConfigurations
        from dronebuddylib.atoms.planning.planner_gui import PlannerGUIApp
        
        config = EngineConfigurations({})
        # ... configure ...
        
        app = PlannerGUIApp(config)
        app.run()
    """
    
    def __init__(self, config=None, planner_config=None):
        """
        Initialize the GUI application.
        
        Args:
            config: EngineConfigurations instance (for standard pattern)
            planner_config: PlannerConfigs instance (alternative)
        """
        self.config = config
        self.planner_config = planner_config
        self.gui = None
        self.engine = None
        
    def run(self):
        """
        Run the GUI application.
        
        Creates the PlannerEngine with GUI callbacks and starts the interface.
        """
        # Disable OpenCV video window since GUI handles video display
        try:
            from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_waypoint_nav_coordinator import TelloWaypointNavCoordinator
            TelloWaypointNavCoordinator._disable_cv2_video_window = True
        except ImportError:
            pass
        
        # Create GUI first
        self.gui = PlannerGUI()
        
        # Get callbacks
        user_input_callback = self.gui.get_user_input_callback()
        message_callback = self.gui.get_message_callback()
        
        try:
            # Import here to avoid circular imports
            from dronebuddylib.atoms.planning import PlannerEngine
            from dronebuddylib.atoms.planning.planner_executor import PlannerExecutor
            
            # Create engine with GUI callbacks
            if self.planner_config:
                self.engine = PlannerEngine.from_config(
                    self.planner_config,
                    user_input_callback=user_input_callback
                )
            elif self.config:
                self.engine = PlannerEngine(
                    self.config,
                    user_input_callback=user_input_callback
                )
            else:
                raise ValueError("Either config or planner_config must be provided")
                
            # Override the executor's message function
            original_send_message = self.engine.executor._send_message_to_user
            self.engine.executor._send_message_to_user = message_callback
            
            # Set up video source callback to connect GUI video display to nav_manager
            def on_video_source(nav_manager):
                """Called when navigation initializes with nav_manager available."""
                if self.gui and nav_manager:
                    # Set frame source directly - this is called from navigation thread
                    # The nav_manager.set_external_frame_callback is thread-safe
                    self.gui.set_frame_source(nav_manager)
            
            self.engine.executor.set_video_source_callback(on_video_source)
            
            # Set engine in GUI
            self.gui.set_engine(self.engine)
            
        except Exception as e:
            # GUI can still run, but search will fail until engine init is fixed.
            print(f"Warning: Failed to initialize engine: {e}")
            
        # Run the GUI
        self.gui.run()
