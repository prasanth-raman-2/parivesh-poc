import json
import os
import re
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from copy import deepcopy

import litellm
from litellm.exceptions import ContextWindowExceededError, BadRequestError

# PDF parsing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")

from app.summarize.tools.summarize_tools import (
    FUNCTION_SCHEMAS,
    read_file, write_file, create_directory, append_file,
    list_directory, copy_file, edit_file, edit_file_batch,
    file_exists, get_file_info, read_lines, search_in_file
)


# ============================================================================
# PROGRESS TRACKER
# ============================================================================

@dataclass
class ProgressTracker:
    """Tracks document processing progress."""
    
    total_pages: int = 0
    total_lines: int = 0
    is_pdf: bool = True
    
    pages_read: Set[int] = field(default_factory=set)
    line_ranges_read: List[tuple] = field(default_factory=list)
    
    template_created: bool = False
    sections_identified: List[str] = field(default_factory=list)
    sections_filled: Dict[str, bool] = field(default_factory=dict)
    
    current_phase: str = "INIT"
    extraction_complete: bool = False
    
    section_summaries: Dict[str, str] = field(default_factory=dict)
    
    def mark_pages_read(self, start: int, end: int) -> None:
        for page in range(start, end + 1):
            self.pages_read.add(page)
    
    def mark_lines_read(self, start: int, end: int) -> None:
        self.line_ranges_read.append((start, end))
    
    def get_unread_pages(self) -> List[int]:
        all_pages = set(range(1, self.total_pages + 1))
        return sorted(all_pages - self.pages_read)
    
    def get_next_chunk(self, chunk_size: int = 10) -> Optional[tuple]:
        if self.is_pdf:
            unread = self.get_unread_pages()
            if not unread:
                return None
            start = unread[0]
            end = min(start + chunk_size - 1, self.total_pages)
            return (start, end)
        else:
            if not self.line_ranges_read:
                return (1, min(chunk_size * 100, self.total_lines))
            max_read = max(end for _, end in self.line_ranges_read)
            if max_read >= self.total_lines:
                return None
            return (max_read + 1, min(max_read + chunk_size * 100, self.total_lines))
    
    def get_progress_percentage(self) -> float:
        if self.is_pdf:
            if self.total_pages == 0:
                return 0.0
            return (len(self.pages_read) / self.total_pages) * 100
        else:
            if self.total_lines == 0:
                return 0.0
            lines_read = sum(end - start + 1 for start, end in self.line_ranges_read)
            return min((lines_read / self.total_lines) * 100, 100.0)
    
    def is_reading_complete(self) -> bool:
        if self.is_pdf:
            return len(self.pages_read) >= self.total_pages
        else:
            if not self.line_ranges_read:
                return False
            max_read = max(end for _, end in self.line_ranges_read)
            return max_read >= self.total_lines
    
    def mark_section_filled(self, section_name: str, summary: str = "") -> None:
        self.sections_filled[section_name] = True
        if summary:
            self.section_summaries[section_name] = summary[:500]
    
    def get_unfilled_sections(self) -> List[str]:
        return [s for s in self.sections_identified if not self.sections_filled.get(s, False)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "total_lines": self.total_lines,
            "is_pdf": self.is_pdf,
            "pages_read": sorted(self.pages_read),
            "pages_read_count": len(self.pages_read),
            "line_ranges_read": self.line_ranges_read,
            "progress_percentage": round(self.get_progress_percentage(), 1),
            "reading_complete": self.is_reading_complete(),
            "template_created": self.template_created,
            "current_phase": self.current_phase,
            "sections_identified": self.sections_identified,
            "sections_filled": self.sections_filled,
            "unfilled_sections": self.get_unfilled_sections(),
            "next_chunk_to_read": self.get_next_chunk()
        }
    
    def get_context_summary(self) -> str:
        """Generate a summary for context recovery."""
        parts = [
            f"=== PROGRESS STATE ===",
            f"Total pages: {self.total_pages}",
            f"Pages read: {len(self.pages_read)} ({self.get_progress_percentage():.1f}%)",
            f"Pages read: {sorted(self.pages_read)}",
            f"Phase: {self.current_phase}",
            f"Template created: {self.template_created}",
            f"Sections filled: {list(self.sections_filled.keys())}",
            f"Unfilled: {self.get_unfilled_sections()}",
        ]
        
        next_chunk = self.get_next_chunk()
        if next_chunk:
            parts.append(f"NEXT: Read pages {next_chunk[0]}-{next_chunk[1]}")
        
        if self.section_summaries:
            parts.append("\n=== SECTION SUMMARIES ===")
            for section, content in self.section_summaries.items():
                parts.append(f"[{section}]: {content[:150]}...")
        
        return "\n".join(parts)


# Global progress tracker
_progress_tracker: Optional[ProgressTracker] = None


def get_progress() -> str:
    global _progress_tracker
    if _progress_tracker is None:
        return json.dumps({"error": "Progress tracker not initialized"})
    return json.dumps(_progress_tracker.to_dict(), indent=2)


def update_progress(
    pages_read_start: Optional[int] = None,
    pages_read_end: Optional[int] = None,
    lines_read_start: Optional[int] = None,
    lines_read_end: Optional[int] = None,
    template_created: Optional[bool] = None,
    sections_identified: Optional[List[str]] = None,
    section_filled: Optional[str] = None,
    section_summary: Optional[str] = None,
    current_phase: Optional[str] = None
) -> str:
    global _progress_tracker
    if _progress_tracker is None:
        return json.dumps({"error": "Progress tracker not initialized"})
    
    if pages_read_start is not None and pages_read_end is not None:
        _progress_tracker.mark_pages_read(pages_read_start, pages_read_end)
    
    if lines_read_start is not None and lines_read_end is not None:
        _progress_tracker.mark_lines_read(lines_read_start, lines_read_end)
    
    if template_created is not None:
        _progress_tracker.template_created = template_created
    
    if sections_identified is not None:
        _progress_tracker.sections_identified = sections_identified
        for section in sections_identified:
            if section not in _progress_tracker.sections_filled:
                _progress_tracker.sections_filled[section] = False
    
    if section_filled is not None:
        _progress_tracker.mark_section_filled(section_filled, section_summary or "")
    
    if current_phase is not None:
        _progress_tracker.current_phase = current_phase
    
    return json.dumps(_progress_tracker.to_dict(), indent=2)


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

@dataclass
class Checkpoint:
    """Checkpoint for resuming interrupted processing."""
    
    config_dict: Dict[str, Any]
    progress_tracker: ProgressTracker
    iteration: int
    timestamp: str
    output_file_content: Optional[str] = None
    
    @classmethod
    def create(cls, config: 'SummarizationConfig', progress: ProgressTracker, 
               iteration: int, output_content: Optional[str] = None) -> 'Checkpoint':
        return cls(
            config_dict={
                "input_path": config.input_path,
                "output_path": config.output_path,
                "model": config.model,
                "max_iterations": config.max_iterations,
                "chunk_size_pages": config.chunk_size_pages,
                "chunk_size_lines": config.chunk_size_lines,
            },
            progress_tracker=progress,
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            output_file_content=output_content
        )


class CheckpointManager:
    """Manages saving and loading checkpoints."""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _get_checkpoint_path(self, input_path: str) -> Path:
        safe_name = Path(input_path).stem.replace(" ", "_")[:50]
        return self.checkpoint_dir / f"checkpoint_{safe_name}.pkl"
    
    def save(self, checkpoint: Checkpoint) -> str:
        path = self._get_checkpoint_path(checkpoint.config_dict["input_path"])
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        return str(path)
    
    def load(self, input_path: str) -> Optional[Checkpoint]:
        path = self._get_checkpoint_path(input_path)
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def delete(self, input_path: str) -> bool:
        path = self._get_checkpoint_path(input_path)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                with open(path, 'rb') as f:
                    cp = pickle.load(f)
                    checkpoints.append({
                        "file": str(path),
                        "input": cp.config_dict["input_path"],
                        "progress": cp.progress_tracker.get_progress_percentage(),
                        "iteration": cp.iteration,
                        "timestamp": cp.timestamp
                    })
            except Exception as e:
                print(f"Error loading checkpoint {path}: {e}")
        return checkpoints


# ============================================================================
# CONTEXT MANAGER - Fixed Smart Truncation
# ============================================================================

class ContextManager:
    """Manages conversation context with proper tool call/response handling."""
    
    def __init__(
        self,
        max_tokens: int = 200000,
        min_recent_messages: int = 10,
        tokens_per_char: float = 0.3
    ):
        self.max_tokens = max_tokens
        self.min_recent_messages = min_recent_messages
        self.tokens_per_char = tokens_per_char
    
    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total_chars += len(str(content))
            # Account for tool_calls in assistant messages
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.function.name)
                    total_chars += len(tc.function.arguments)
        return int(total_chars * self.tokens_per_char)
    
    def _get_message_role(self, msg: Dict[str, Any]) -> str:
        """Get role from message, handling both dict and object formats."""
        if isinstance(msg, dict):
            return msg.get("role", "")
        return getattr(msg, "role", "")
    
    def _has_tool_calls(self, msg: Any) -> bool:
        """Check if message has tool_calls."""
        if isinstance(msg, dict):
            return bool(msg.get("tool_calls"))
        return bool(getattr(msg, "tool_calls", None))
    
    def _get_tool_call_ids(self, msg: Any) -> Set[str]:
        """Extract tool_call IDs from an assistant message."""
        ids = set()
        tool_calls = None
        
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls", [])
        else:
            tool_calls = getattr(msg, "tool_calls", []) or []
        
        for tc in tool_calls:
            if isinstance(tc, dict):
                ids.add(tc.get("id", ""))
            else:
                ids.add(getattr(tc, "id", ""))
        
        return ids
    
    def _get_tool_call_id(self, msg: Dict[str, Any]) -> Optional[str]:
        """Get tool_call_id from a tool message."""
        return msg.get("tool_call_id")
    
    def _group_messages_by_conversation_units(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group messages into conversation units that must stay together.
        
        A conversation unit is:
        - A single user message
        - An assistant message WITHOUT tool_calls
        - An assistant message WITH tool_calls + ALL its corresponding tool responses
        """
        units = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            role = self._get_message_role(msg)
            
            if role == "assistant" and self._has_tool_calls(msg):
                # Start a new unit with assistant + all tool responses
                unit = [msg]
                tool_call_ids = self._get_tool_call_ids(msg)
                i += 1
                
                # Collect all corresponding tool responses
                while i < len(messages):
                    next_msg = messages[i]
                    next_role = self._get_message_role(next_msg)
                    
                    if next_role == "tool":
                        tool_id = self._get_tool_call_id(next_msg)
                        if tool_id in tool_call_ids:
                            unit.append(next_msg)
                            i += 1
                        else:
                            # Tool response for different call - shouldn't happen
                            # but handle gracefully
                            break
                    else:
                        # Next message is not a tool response
                        break
                
                units.append(unit)
            else:
                # Single message unit (user, system, or assistant without tools)
                units.append([msg])
                i += 1
        
        return units
    
    def _estimate_unit_tokens(self, unit: List[Dict[str, Any]]) -> int:
        """Estimate tokens for a conversation unit."""
        return self.estimate_tokens(unit)
    
    def truncate_if_needed(
        self,
        messages: List[Dict[str, Any]],
        progress_tracker: Optional[ProgressTracker] = None
    ) -> List[Dict[str, Any]]:
        """
        Truncate messages while preserving:
        1. System prompt (first message)
        2. Initial user prompt (second message)  
        3. Complete tool call/response groups (never break apart)
        4. Recent conversation units
        """
        estimated_tokens = self.estimate_tokens(messages)
        
        if estimated_tokens <= self.max_tokens:
            return messages
        
        print(f"\n⚠️ Context too large ({estimated_tokens} est. tokens). Truncating...")
        
        # Group messages into conversation units
        units = self._group_messages_by_conversation_units(messages)
        
        if len(units) <= 3:
            # Can't truncate further
            return messages
        
        # Always keep first 2 units (system + initial user)
        preserved_start_units = units[:2]
        remaining_units = units[2:]
        
        # Calculate tokens for preserved start
        start_tokens = sum(self._estimate_unit_tokens(u) for u in preserved_start_units)
        
        # Create recovery message
        context_summary = ""
        if progress_tracker:
            context_summary = progress_tracker.get_context_summary()
        
        recovery_message = {
            "role": "user",
            "content": f"""⚠️ CONTEXT TRUNCATED

Previous conversation was truncated. Here's the current state:

{context_summary}

CONTINUE from where you left off. Call `get_progress()` to see current state."""
        }
        recovery_tokens = int(len(recovery_message["content"]) * self.tokens_per_char)
        
        # Calculate how many recent units we can keep
        available_tokens = self.max_tokens - start_tokens - recovery_tokens - 10000  # Buffer
        
        # Select recent units from the end, keeping complete groups
        kept_units = []
        tokens_used = 0
        
        for unit in reversed(remaining_units):
            unit_tokens = self._estimate_unit_tokens(unit)
            if tokens_used + unit_tokens <= available_tokens:
                kept_units.insert(0, unit)
                tokens_used += unit_tokens
            else:
                break
        
        # Ensure we keep at least a few recent units
        if len(kept_units) < self.min_recent_messages // 2 and remaining_units:
            # Force keep last few units even if over budget
            kept_units = remaining_units[-max(3, self.min_recent_messages // 3):]
        
        # Flatten units back to messages
        truncated = []
        for unit in preserved_start_units:
            truncated.extend(unit)
        
        truncated.append(recovery_message)
        
        for unit in kept_units:
            truncated.extend(unit)
        
        # Validate the result
        truncated = self._validate_and_fix_messages(truncated)
        
        new_estimate = self.estimate_tokens(truncated)
        print(f"  Reduced from ~{estimated_tokens} to ~{new_estimate} tokens")
        print(f"  Kept {len(preserved_start_units)} start units + 1 recovery + {len(kept_units)} recent units")
        
        return truncated
    
    def _validate_and_fix_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate message sequence and fix any issues.
        Ensures tool messages always follow assistant messages with tool_calls.
        """
        validated = []
        pending_tool_ids: Set[str] = set()
        
        for msg in messages:
            role = self._get_message_role(msg)
            
            if role == "tool":
                tool_id = self._get_tool_call_id(msg)
                if tool_id in pending_tool_ids:
                    validated.append(msg)
                    pending_tool_ids.discard(tool_id)
                else:
                    # Orphan tool message - skip it
                    print(f"  ⚠️ Skipping orphan tool message (no matching tool_call)")
                    continue
            elif role == "assistant" and self._has_tool_calls(msg):
                validated.append(msg)
                pending_tool_ids = self._get_tool_call_ids(msg)
            else:
                # Clear pending tool IDs if we see a non-tool message
                # This handles cases where tool responses were lost
                if pending_tool_ids:
                    print(f"  ⚠️ {len(pending_tool_ids)} tool responses missing, inserting placeholders")
                    for tid in pending_tool_ids:
                        validated.append({
                            "role": "tool",
                            "tool_call_id": tid,
                            "content": "(response truncated)"
                        })
                    pending_tool_ids = set()
                
                validated.append(msg)
        
        # Handle any remaining pending tool IDs at the end
        if pending_tool_ids:
            print(f"  ⚠️ Adding {len(pending_tool_ids)} missing tool responses at end")
            for tid in pending_tool_ids:
                validated.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": "(response truncated)"
                })
        
        return validated


# ============================================================================
# PDF FUNCTIONS
# ============================================================================

def read_pdf_file(file_path: str) -> str:
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF not installed")
    
    doc = fitz.open(file_path)
    text_content = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_content.append(f"\n--- Page {page_num + 1} ---\n")
        text_content.append(page.get_text())
    doc.close()
    return "".join(text_content)


def read_pdf_pages(file_path: str, start_page: int = 1, end_page: Optional[int] = None) -> str:
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF not installed")
    
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    start_idx = max(0, start_page - 1)
    end_idx = total_pages if end_page is None else min(end_page, total_pages)
    
    text_content = []
    for page_num in range(start_idx, end_idx):
        page = doc[page_num]
        text_content.append(f"\n--- Page {page_num + 1} ---\n")
        text_content.append(page.get_text())
    
    doc.close()
    
    global _progress_tracker
    if _progress_tracker is not None:
        _progress_tracker.mark_pages_read(start_page, end_idx)
    
    return "".join(text_content)


def get_pdf_info(file_path: str) -> Dict[str, Any]:
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF not installed")
    
    doc = fitz.open(file_path)
    info = {
        "total_pages": len(doc),
        "metadata": doc.metadata,
        "file_size": os.path.getsize(file_path),
        "is_encrypted": doc.is_encrypted,
    }
    doc.close()
    
    global _progress_tracker
    if _progress_tracker is not None:
        _progress_tracker.total_pages = info["total_pages"]
    
    return info


def search_in_pdf(file_path: str, pattern: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF not installed")
    
    doc = fitz.open(file_path)
    matches = []
    search_pattern = pattern if case_sensitive else pattern.lower()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        search_text = text if case_sensitive else text.lower()
        
        if search_pattern in search_text:
            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                search_line = line if case_sensitive else line.lower()
                if search_pattern in search_line:
                    matches.append({
                        "page": page_num + 1,
                        "line_in_page": line_num + 1,
                        "text": line.strip()
                    })
    
    doc.close()
    return matches


def read_lines_tracked(file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> List[str]:
    result = read_lines(file_path, start_line, end_line)
    
    global _progress_tracker
    if _progress_tracker is not None:
        actual_end = start_line + len(result) - 1 if result else start_line
        _progress_tracker.mark_lines_read(start_line, actual_end)
    
    return result


# ============================================================================
# FUNCTION SCHEMAS
# ============================================================================

PROGRESS_FUNCTION_SCHEMAS = [
    {
        "name": "get_progress",
        "description": "Get current processing progress. CALL THIS FREQUENTLY.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {},
            "required": []
        }
    },
    {
        "name": "update_progress",
        "description": "Update processing progress after actions.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "pages_read_start": {"type": ["integer", "null"]},
                "pages_read_end": {"type": ["integer", "null"]},
                "lines_read_start": {"type": ["integer", "null"]},
                "lines_read_end": {"type": ["integer", "null"]},
                "template_created": {"type": ["boolean", "null"]},
                "sections_identified": {"type": ["array", "null"], "items": {"type": "string"}},
                "section_filled": {"type": ["string", "null"]},
                "section_summary": {"type": ["string", "null"]},
                "current_phase": {"type": ["string", "null"], "enum": ["INIT", "TEMPLATE", "EXTRACTION", "FINALIZE", "DONE", None]}
            },
            "required": []
        }
    }
]

PDF_FUNCTION_SCHEMAS = [
    {
        "name": "read_pdf_file",
        "description": "Reads complete PDF. Use ONLY for small PDFs (< 20 pages).",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {"type": "string"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "read_pdf_pages",
        "description": "Reads specific pages from PDF. Progress auto-tracked.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {"type": "string"},
                "start_page": {"type": "integer", "minimum": 1},
                "end_page": {"type": ["integer", "null"], "minimum": 1}
            },
            "required": ["file_path", "start_page"]
        }
    },
    {
        "name": "get_pdf_info",
        "description": "Gets PDF info including total page count.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {"type": "string"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "search_in_pdf",
        "description": "Searches for pattern in PDF.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {"type": "string"},
                "pattern": {"type": "string"},
                "case_sensitive": {"type": "boolean"}
            },
            "required": ["file_path", "pattern"]
        }
    }
]


# ============================================================================
# CONFIGURATION
# ============================================================================

class SummarizationConfig:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        model: str = "gpt-4.1",
        max_iterations: int = 250,
        chunk_size_pages: int = 10,
        chunk_size_lines: int = 300,
        initial_scan_pages: int = 10,
        initial_scan_lines: int = 500,
        checkpoint_dir: str = ".checkpoints",
        max_context_tokens: int = 180000,  # Conservative
        auto_checkpoint_interval: int = 10,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.model = model
        self.max_iterations = max_iterations
        self.chunk_size_pages = chunk_size_pages
        self.chunk_size_lines = chunk_size_lines
        self.initial_scan_pages = initial_scan_pages
        self.initial_scan_lines = initial_scan_lines
        self.checkpoint_dir = checkpoint_dir
        self.max_context_tokens = max_context_tokens
        self.auto_checkpoint_interval = auto_checkpoint_interval
        
        self.is_pdf = input_path.lower().endswith('.pdf')
        self.file_type = "PDF" if self.is_pdf else "TEXT"


# ============================================================================
# FUNCTION MAP
# ============================================================================

FUNCTION_MAP = {
    "get_progress": get_progress,
    "update_progress": update_progress,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "create_directory": create_directory,
    "list_directory": list_directory,
    "copy_file": copy_file,
    "edit_file": edit_file,
    "edit_file_batch": edit_file_batch,
    "file_exists": file_exists,
    "get_file_info": get_file_info,
    "read_lines": read_lines_tracked,
    "search_in_file": search_in_file,
    "read_pdf_file": read_pdf_file,
    "read_pdf_pages": read_pdf_pages,
    "get_pdf_info": get_pdf_info,
    "search_in_pdf": search_in_pdf,
}


# ============================================================================
# PROMPTS
# ============================================================================

def get_system_prompt(config: SummarizationConfig) -> str:
    return f"""You are an expert document analyst. Create a COMPLETE summary ensuring EVERY page is read.

## CRITICAL RULES:

### 1. TRACK PROGRESS
- Call `get_progress()` at START of every iteration
- Cannot finish until `reading_complete` is TRUE

### 2. READ ALL PAGES
- Read {config.chunk_size_pages} pages at a time
- `next_chunk_to_read` tells you exactly what to read next
- Continue until `reading_complete: true`

### 3. UPDATE AFTER ACTIONS
- After filling section: `update_progress(section_filled="NAME", section_summary="brief")`
- After template: `update_progress(template_created=true, sections_identified=[...])`

## PHASES:
1. INIT: Get doc info
2. TEMPLATE: Read first {config.initial_scan_pages} pages, create template
3. EXTRACTION: Read ALL pages, fill template
4. FINALIZE: Verify, write summary
5. DONE: Only when reading_complete=true

## INPUT: {config.input_path}
## OUTPUT: {config.output_path}

START with get_pdf_info, then get_progress."""


def get_user_prompt(config: SummarizationConfig) -> str:
    return f"""Create comprehensive summary. Read ALL pages.

Input: {config.input_path}
Output: {config.output_path}

Call get_progress() frequently. Do NOT finish until ALL pages read.

BEGIN with get_pdf_info("{config.input_path}")"""


def get_resume_prompt(config: SummarizationConfig, checkpoint: Checkpoint) -> str:
    progress = checkpoint.progress_tracker
    next_chunk = progress.get_next_chunk()
    
    return f"""⚠️ RESUMING FROM CHECKPOINT

Progress: {progress.get_progress_percentage():.1f}%
Pages: {len(progress.pages_read)}/{progress.total_pages}
Phase: {progress.current_phase}
Unfilled: {progress.get_unfilled_sections()[:5]}

NEXT: {"Read pages " + str(next_chunk[0]) + "-" + str(next_chunk[1]) if next_chunk else "Finalize"}

Call get_progress() first, then continue."""


# ============================================================================
# MAIN AGENT
# ============================================================================

class DocumentSummarizationAgent:
    """Agent with fixed context management."""
    
    def __init__(self, config: SummarizationConfig, resume: bool = True):
        self.config = config
        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.context_manager = ContextManager(max_tokens=config.max_context_tokens)
        self.start_iteration = 0
        
        global _progress_tracker
        
        checkpoint = None
        if resume:
            checkpoint = self.checkpoint_manager.load(config.input_path)
        
        if checkpoint:
            print(f"\n📁 Checkpoint found: {checkpoint.timestamp}")
            print(f"   Progress: {checkpoint.progress_tracker.get_progress_percentage():.1f}%")
            
            _progress_tracker = checkpoint.progress_tracker
            self.start_iteration = checkpoint.iteration
            
            if checkpoint.output_file_content and not file_exists(config.output_path):
                write_file(config.output_path, checkpoint.output_file_content)
            
            self._setup_tools()
            self._setup_messages_resume(checkpoint)
        else:
            _progress_tracker = ProgressTracker(is_pdf=config.is_pdf)
            self._setup_tools()
            self._setup_messages()
    
    def _setup_tools(self):
        for schema in PROGRESS_FUNCTION_SCHEMAS:
            self.tools.append({"type": "function", "function": schema})
        for schema in FUNCTION_SCHEMAS:
            self.tools.append({"type": "function", "function": schema})
        if PDF_SUPPORT:
            for schema in PDF_FUNCTION_SCHEMAS:
                self.tools.append({"type": "function", "function": schema})
    
    def _setup_messages(self):
        self.messages = [
            {"role": "system", "content": get_system_prompt(self.config)},
            {"role": "user", "content": get_user_prompt(self.config)}
        ]
    
    def _setup_messages_resume(self, checkpoint: Checkpoint):
        self.messages = [
            {"role": "system", "content": get_system_prompt(self.config)},
            {"role": "user", "content": get_resume_prompt(self.config, checkpoint)}
        ]
    
    def _execute_function(self, function_name: str, function_args: Dict[str, Any]) -> str:
        func = FUNCTION_MAP.get(function_name)
        
        if not func:
            return f"Error: Function '{function_name}' not recognized."
        
        try:
            result = func(**function_args)
            
            if result is None:
                return "Success"
            elif isinstance(result, str):
                return result
            elif isinstance(result, list):
                return "\n".join(str(item) for item in result) if result else "(empty)"
            else:
                return json.dumps(result, indent=2, default=str)
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _log_tool_call(self, function_name: str, function_args: Dict[str, Any], response: str):
        print(f"\n  📌 {function_name}", end="")
        
        if function_name == "get_progress":
            try:
                p = json.loads(response)
                print(f" → {p.get('progress_percentage', 0):.1f}%")
            except:
                print()
        elif function_name == "read_pdf_pages":
            print(f" → pages {function_args.get('start_page')}-{function_args.get('end_page')}")
        elif function_name == "edit_file":
            print(f" → edit '{function_args.get('search_text', '')[:25]}...'")
        else:
            print()
    
    def _save_checkpoint(self, iteration: int):
        global _progress_tracker
        
        output_content = None
        if file_exists(self.config.output_path):
            try:
                output_content = read_file(self.config.output_path)
            except:
                pass
        
        checkpoint = Checkpoint.create(
            self.config,
            deepcopy(_progress_tracker),
            iteration,
            output_content
        )
        
        path = self.checkpoint_manager.save(checkpoint)
        print(f"\n  💾 Checkpoint saved")
    
    def _check_completion_allowed(self) -> tuple[bool, str]:
        global _progress_tracker
        
        if _progress_tracker is None:
            return False, "Tracker not initialized"
        
        if not _progress_tracker.is_reading_complete():
            unread = _progress_tracker.get_unread_pages()[:3]
            return False, f"Unread pages: {unread}..."
        
        return True, "OK"
    
    def run(self) -> bool:
        global _progress_tracker
        
        print("=" * 70)
        print("DOCUMENT SUMMARIZATION AGENT")
        print("=" * 70)
        print(f"Input:  {self.config.input_path}")
        print(f"Output: {self.config.output_path}")
        print("=" * 70)
        
        stall_count = 0
        last_progress = 0.0
        
        for iteration in range(self.start_iteration, self.config.max_iterations):
            print(f"\n{'─' * 50}")
            print(f"Iteration {iteration + 1}/{self.config.max_iterations}", end="")
            
            if _progress_tracker:
                current = _progress_tracker.get_progress_percentage()
                print(f" | {current:.1f}% | {_progress_tracker.current_phase}")
                
                if current == last_progress and current > 0:
                    stall_count += 1
                else:
                    stall_count = 0
                last_progress = current
                
                if stall_count >= 3:
                    next_chunk = _progress_tracker.get_next_chunk()
                    if next_chunk:
                        self.messages.append({
                            "role": "user",
                            "content": f"Read pages {next_chunk[0]}-{next_chunk[1]} now."
                        })
                        stall_count = 0
            else:
                print()
            
            # Auto-checkpoint
            if iteration > 0 and iteration % self.config.auto_checkpoint_interval == 0:
                self._save_checkpoint(iteration)
            
            # Truncate context if needed
            self.messages = self.context_manager.truncate_if_needed(
                self.messages, _progress_tracker
            )
            
            # LLM call with error handling
            try:
                response = litellm.completion(
                    model=self.config.model,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
            except (ContextWindowExceededError, BadRequestError) as e:
                error_msg = str(e)
                print(f"\n⚠️ API Error: {error_msg[:100]}")
                
                if "tool" in error_msg.lower() or "context" in error_msg.lower():
                    # More aggressive truncation
                    print("  Applying aggressive truncation...")
                    self.context_manager.max_tokens = int(self.context_manager.max_tokens * 0.6)
                    self.messages = self.context_manager.truncate_if_needed(
                        self.messages, _progress_tracker
                    )
                    
                    self._save_checkpoint(iteration)
                    
                    try:
                        response = litellm.completion(
                            model=self.config.model,
                            messages=self.messages,
                            tools=self.tools,
                            tool_choice="auto"
                        )
                    except Exception as e2:
                        print(f"❌ Retry failed: {e2}")
                        print("Run again to resume from checkpoint.")
                        return False
                else:
                    self._save_checkpoint(iteration)
                    raise
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            self.messages.append(response_message)
            
            if response_message.content:
                print(f"\nAssistant: {response_message.content[:200]}...")
            
            if not tool_calls:
                can_complete, reason = self._check_completion_allowed()
                
                if can_complete:
                    self.checkpoint_manager.delete(self.config.input_path)
                    print("\n" + "=" * 70)
                    print("✅ COMPLETE")
                    print("=" * 70)
                    return True
                else:
                    print(f"\n  ⛔ {reason}")
                    self.messages.append({
                        "role": "user",
                        "content": f"Cannot finish: {reason}. Call get_progress() and continue."
                    })
                    continue
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                function_response = self._execute_function(function_name, function_args)
                self._log_tool_call(function_name, function_args, function_response)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_response
                })
        
        self._save_checkpoint(self.config.max_iterations)
        print(f"\n⚠️ Max iterations. Run again to resume.")
        return False
    
    def verify_output(self) -> Dict[str, Any]:
        global _progress_tracker
        
        result = {
            "file_exists": False,
            "file_size": 0,
            "unfilled_placeholders": [],
            "sections_found": 0,
            "pages_processed": 0,
            "total_pages": 0,
            "complete": False
        }
        
        if _progress_tracker:
            result["pages_processed"] = len(_progress_tracker.pages_read)
            result["total_pages"] = _progress_tracker.total_pages
            result["complete"] = _progress_tracker.is_reading_complete()
        
        if not file_exists(self.config.output_path):
            return result
        
        result["file_exists"] = True
        result["file_size"] = get_file_info(self.config.output_path)["size"]
        
        content = read_file(self.config.output_path)
        result["unfilled_placeholders"] = list(set(re.findall(r'\{\{[A-Z_]+\}\}', content)))
        result["sections_found"] = len(re.findall(r'^#{1,3}\s+.+$', content, re.MULTILINE))
        
        return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = SummarizationConfig(
        input_path="/home/prasa/projects/negd/parivesh-poc/docs/eia_report.pdf",
        output_path="/home/prasa/projects/negd/parivesh-poc/docs/summary_1.md",
        model="gpt-5.2",
        max_iterations=250,
        chunk_size_pages=10,
        auto_checkpoint_interval=10,
        max_context_tokens=150000,  # More conservative
    )
    
    agent = DocumentSummarizationAgent(config, resume=True)
    success = agent.run()
    
    v = agent.verify_output()
    print(f"\nPages: {v['pages_processed']}/{v['total_pages']} | Complete: {v['complete']}")
    if v["unfilled_placeholders"]:
        print(f"Unfilled: {v['unfilled_placeholders'][:5]}")