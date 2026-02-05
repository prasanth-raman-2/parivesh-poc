import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any


# Function schemas for LLM integration
FUNCTION_SCHEMAS = [
    {
        "name": "read_file",
        "description": "Reads the complete content of a text file and returns it as a string. Use this when you need to read the entire file content.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read."
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "write_file",
        "description": "Writes content to a file, creating parent directories if needed. Overwrites existing file content.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file."
                }
            },
            "required": ["file_path", "content"]
        }
    },
    {
        "name": "append_file",
        "description": "Appends content to the end of a file. Creates the file if it doesn't exist.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file."
                },
                "content": {
                    "type": "string",
                    "description": "Content to append to the file."
                }
            },
            "required": ["file_path", "content"]
        }
    },
    {
        "name": "list_directory",
        "description": "Lists all files and directories in the specified directory. Can optionally list recursively.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory to list."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, lists all files recursively in subdirectories. Default is false."
                }
            },
            "required": ["directory"]
        }
    },
    {
        "name": "create_directory",
        "description": "Creates a directory and all necessary parent directories.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory to create."
                }
            },
            "required": ["directory"]
        }
    },
    {
        "name": "file_exists",
        "description": "Checks if a file or directory exists at the specified path.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to check for existence."
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "get_file_info",
        "description": "Gets detailed information about a file including size, timestamps, and type.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to get information about."
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "copy_file",
        "description": "Copies a file from source to destination, creating parent directories if needed.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source file path to copy from."
                },
                "destination": {
                    "type": "string",
                    "description": "Destination file path to copy to."
                }
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "move_file",
        "description": "Moves or renames a file from source to destination, creating parent directories if needed.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source file path to move from."
                },
                "destination": {
                    "type": "string",
                    "description": "Destination file path to move to."
                }
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "read_lines",
        "description": "Reads specific line ranges from a file. Useful for reading only parts of large files.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read."
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed). Default is 1.",
                    "minimum": 1
                },
                "end_line": {
                    "type": ["integer", "null"],
                    "description": "Ending line number (1-indexed, inclusive). Null means read to end of file.",
                    "minimum": 1
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "search_in_file",
        "description": "Searches for a pattern in a file and returns all matching lines with line numbers.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to search in."
                },
                "pattern": {
                    "type": "string",
                    "description": "Text pattern to search for."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case-sensitive. Default is false."
                }
            },
            "required": ["file_path", "pattern"]
        }
    },
    {
        "name": "edit_file",
        "description": "Edits a file by replacing the first occurrence of search_text with replace_text. Uses exact string matching.",
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit."
                },
                "search_text": {
                    "type": "string",
                    "description": "Exact text to search for (must match exactly including whitespace)."
                },
                "replace_text": {
                    "type": "string",
                    "description": "Text to replace the search_text with."
                }
            },
            "required": ["file_path", "search_text", "replace_text"]
        }
    },
    {
        "name": "edit_file_batch",
        "description": (
            "Edits a file using SEARCH/REPLACE block format. Supports multiple replacements in a single call. "
            "Format: file_path\\n<<<<<<< SEARCH\\nold content\\n=======\\nnew content\\n>>>>>>> REPLACE"
        ),
        "parameters": {
            "type": "object",
            "strict": True,
            "properties": {
                "changes": {
                    "type": "string",
                    "description": "Multi-line string with file path on first line followed by SEARCH/REPLACE blocks."
                }
            },
            "required": ["changes"]
        }
    }
]


def read_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def write_file(file_path: str, content: str) -> None:
    """
    Writes the given content to a file.
    Creates parent directories if they don't exist.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def append_file(file_path: str, content: str) -> None:
    """
    Appends content to the end of a file.
    Creates the file if it doesn't exist.
    
    Args:
        file_path: Path to the file
        content: Content to append
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content)


# def delete_file(file_path: str) -> bool:
#     """
#     Deletes a file or directory.
    
#     Args:
#         file_path: Path to the file or directory to delete
        
#     Returns:
#         True if deletion was successful, False otherwise
#     """
#     try:
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#         return True
#     except Exception as e:
#         print(f"Error deleting {file_path}: {e}")
#         return False


def list_directory(directory: str, recursive: bool = False) -> List[str]:
    """
    Lists files and directories in the specified directory.
    
    Args:
        directory: Path to the directory
        recursive: If True, lists all files recursively
        
    Returns:
        List of file/directory paths
    """
    if not os.path.exists(directory):
        return []
    
    if recursive:
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files
    else:
        return [os.path.join(directory, item) for item in os.listdir(directory)]


def create_directory(directory: str) -> None:
    """
    Creates a directory and all parent directories.
    
    Args:
        directory: Path to the directory to create
    """
    os.makedirs(directory, exist_ok=True)


def file_exists(file_path: str) -> bool:
    """
    Checks if a file or directory exists.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file/directory exists, False otherwise
    """
    return os.path.exists(file_path)


def get_file_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Gets information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file info (size, created, modified) or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None
    
    stats = os.stat(file_path)
    return {
        'size': stats.st_size,
        'created': stats.st_ctime,
        'modified': stats.st_mtime,
        'accessed': stats.st_atime,
        'is_file': os.path.isfile(file_path),
        'is_directory': os.path.isdir(file_path)
    }


def copy_file(source: str, destination: str) -> bool:
    """
    Copies a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if copy was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"Error copying {source} to {destination}: {e}")
        return False


def move_file(source: str, destination: str) -> bool:
    """
    Moves a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if move was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
        return True
    except Exception as e:
        print(f"Error moving {source} to {destination}: {e}")
        return False


def read_lines(file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> List[str]:
    """
    Reads specific lines from a file.
    
    Args:
        file_path: Path to the file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive). None means read to end.
        
    Returns:
        List of lines
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    start_idx = max(0, start_line - 1)
    end_idx = len(lines) if end_line is None else min(end_line, len(lines))
    
    return lines[start_idx:end_idx]


def search_in_file(file_path: str, pattern: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Searches for a pattern in a file and returns matching lines.
    
    Args:
        file_path: Path to the file
        pattern: Pattern to search for
        case_sensitive: Whether search should be case-sensitive
        
    Returns:
        List of dictionaries with line number and line content
    """
    matches = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            search_line = line if case_sensitive else line.lower()
            search_pattern = pattern if case_sensitive else pattern.lower()
            
            if search_pattern in search_line:
                matches.append({
                    'line_number': line_num,
                    'line': line.rstrip('\n')
                })
    
    return matches


def edit_file(file_path: str, search_text: str, replace_text: str) -> bool:
    """
    Edits a file by replacing search_text with replace_text.
    Uses exact string matching for the replacement.
    
    Args:
        file_path: Path to the file to edit
        search_text: Text to search for (must match exactly)
        replace_text: Text to replace with
        
    Returns:
        True if replacement was successful, False if search text not found
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If search_text is empty
    """
    if not search_text:
        raise ValueError("search_text cannot be empty")
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Check if search text exists in the file
    if search_text not in content:
        return False
    
    # Replace the text (only first occurrence to match container tools behavior)
    new_content = content.replace(search_text, replace_text, 1)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    return True


def edit_file_batch(changes: str) -> Dict[str, Any]:
    """
    Edits a file using SEARCH/REPLACE block format.
    Supports multiple SEARCH/REPLACE blocks in a single call.
    
    Format:
        file_path
        <<<<<<< SEARCH
        old content
        =======
        new content
        >>>>>>> REPLACE
    
    Args:
        changes: String containing file path and SEARCH/REPLACE blocks
        
    Returns:
        Dictionary with status and message
        
    Example:
        changes = '''
        /path/to/file.py
        <<<<<<< SEARCH
        def old_function():
            pass
        =======
        def new_function():
            return True
        >>>>>>> REPLACE
        '''
    """
    try:
        lines = changes.strip().splitlines()
        if not lines:
            return {"status": "ERROR", "message": "No changes provided"}
        
        file_path = lines[0].strip()
        
        if not os.path.exists(file_path):
            return {"status": "ERROR", "message": f"File not found: {file_path}"}
        
        # Read original file content
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        index = 1
        replacements_made = 0
        
        # Process all SEARCH/REPLACE blocks
        while index < len(lines):
            if not lines[index].startswith("<<<<<<< SEARCH"):
                index += 1
                continue
            
            search_start = index
            
            # Find the separator and end markers
            try:
                separator_index = lines.index("=======", search_start)
                end_index = lines.index(">>>>>>> REPLACE", separator_index)
            except ValueError:
                return {"status": "ERROR", "message": "Malformed SEARCH/REPLACE block"}
            
            # Extract search and replace content
            search_content = "\n".join(lines[search_start + 1:separator_index])
            replace_content = "\n".join(lines[separator_index + 1:end_index])
            
            # Check if search content exists
            if search_content not in file_content:
                return {
                    "status": "ERROR",
                    "message": f"Search content not found in file. Search block starting at line {search_start + 1}"
                }
            
            # Apply the replacement (only first occurrence)
            file_content = file_content.replace(search_content, replace_content, 1)
            replacements_made += 1
            
            index = end_index + 1
        
        # Write the updated content
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_content)
        
        return {
            "status": "SUCCESS",
            "message": f"Successfully applied {replacements_made} replacement(s) to {file_path}"
        }
        
    except Exception as e:
        return {"status": "ERROR", "message": f"Error editing file: {str(e)}"}