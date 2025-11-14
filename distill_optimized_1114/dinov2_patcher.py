"""
DINOv2 Python 3.9 Compatibility Patcher
Patches DINOv2 code to work with Python 3.9 by replacing | type hints with Union
"""
import re
import sys
from pathlib import Path
import torch.hub


def patch_dinov2_file(file_path):
    """
    Patch a single Python file to replace | type hints with Union/Optional
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        patched_lines = []
        has_union = 'Union' in content
        has_optional = 'Optional' in content
        needs_union_import = False
        needs_optional_import = False
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Skip if already has Union/Optional
            if 'Union[' in line or 'Optional[' in line:
                patched_lines.append(line)
                continue
            
            # Check if line has type hints with |
            if '|' in line and (':' in line or '->' in line):
                # Pattern 1: Type | None or None | Type (handle first)
                if '| None' in line or 'None |' in line:
                    # Replace Type | None with Optional[Type]
                    line = re.sub(
                        r'(\w+(?:\[[^\]]+\])?)\s*\|\s*None\b',
                        r'Optional[\1]',
                        line
                    )
                    line = re.sub(
                        r'None\s*\|\s*(\w+(?:\[[^\]]+\])?)',
                        r'Optional[\1]',
                        line
                    )
                    if line != original_line:
                        needs_optional_import = True
                
                # Pattern 2: Type1 | Type2 (not None) - handle after Optional
                if '|' in line and 'Optional[' not in line and 'Union[' not in line:
                    # Replace Type1 | Type2 with Union[Type1, Type2]
                    # Use a more careful regex that handles word boundaries
                    line = re.sub(
                        r'(\w+(?:\[[^\]]+\])?)\s*\|\s*(\w+(?:\[[^\]]+\])?)',
                        r'Union[\1, \2]',
                        line
                    )
                    if line != original_line:
                        needs_union_import = True
            
            patched_lines.append(line)
        
        # Add imports if needed
        if needs_union_import or needs_optional_import:
            import_line_idx = None
            typing_import_idx = None
            
            # Find where to insert imports
            for i, line in enumerate(patched_lines):
                if line.startswith('import ') or line.startswith('from '):
                    if import_line_idx is None:
                        import_line_idx = i
                    if 'from typing import' in line:
                        typing_import_idx = i
                        break
            
            if typing_import_idx is not None:
                # Add to existing typing import
                existing_line = patched_lines[typing_import_idx]
                imports_to_add = []
                if needs_union_import and 'Union' not in existing_line:
                    imports_to_add.append('Union')
                if needs_optional_import and 'Optional' not in existing_line:
                    imports_to_add.append('Optional')
                
                if imports_to_add:
                    # Extract existing imports
                    if 'from typing import' in existing_line:
                        existing_imports = existing_line.replace('from typing import', '').strip()
                        all_imports = existing_imports.split(',') + imports_to_add
                        all_imports = [imp.strip() for imp in all_imports]
                        patched_lines[typing_import_idx] = f"from typing import {', '.join(sorted(set(all_imports)))}"
            elif import_line_idx is not None:
                # Add new typing import
                imports = []
                if needs_union_import:
                    imports.append('Union')
                if needs_optional_import:
                    imports.append('Optional')
                patched_lines.insert(import_line_idx, f"from typing import {', '.join(imports)}")
        
        patched_content = '\n'.join(patched_lines)
        
        # Write if changed
        if patched_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(patched_content)
            return True
        
        return False
    
    except Exception as e:
        # Silently skip files that can't be patched
        return False


def patch_dinov2_for_python39():
    """
    Patch all DINOv2 Python files to be compatible with Python 3.9
    """
    # Get the DINOv2 cache directory
    hub_dir = Path(torch.hub.get_dir())
    dinov2_path = hub_dir / "facebookresearch_dinov2_main"
    
    if not dinov2_path.exists():
        return dinov2_path, 0
    
    # Find all Python files
    python_files = list(dinov2_path.rglob("*.py"))
    patched_count = 0
    
    for py_file in python_files:
        if patch_dinov2_file(py_file):
            patched_count += 1
    
    return dinov2_path, patched_count


def ensure_dinov2_patched():
    """
    Ensure DINOv2 is patched for Python 3.9 compatibility.
    This should be called before loading DINOv2 models.
    """
    import sys
    
    if sys.version_info >= (3, 10):
        # No patching needed for Python 3.10+
        return
    
    # Check if already patched (look for a marker file)
    hub_dir = Path(torch.hub.get_dir())
    dinov2_path = hub_dir / "facebookresearch_dinov2_main"
    marker_file = dinov2_path / ".python39_patched" if dinov2_path.exists() else None
    
    if marker_file and marker_file.exists():
        # Already patched
        return
    
    # Try to trigger download first (if not already downloaded)
    try:
        torch.hub.list("facebookresearch/dinov2", force_reload=False)
    except:
        pass
    
    # Patch the files
    if dinov2_path.exists():
        print("Patching DINOv2 for Python 3.9 compatibility...")
        _, patched_count = patch_dinov2_for_python39()
        if patched_count > 0:
            print(f"  ✓ Patched {patched_count} files")
            # Create marker file
            marker_file = dinov2_path / ".python39_patched"
            marker_file.touch()
        else:
            print("  ✓ DINOv2 files already compatible (or no changes needed)")


def load_dinov2_with_patch(model_name="dinov2_vitb14", verbose=False):
    """
    Load DINOv2 model with automatic Python 3.9 patching
    
    Args:
        model_name: Name of the DINOv2 model to load
        verbose: Whether to show verbose output
    
    Returns:
        DINOv2 model
    """
    import warnings
    
    # Ensure DINOv2 is patched
    ensure_dinov2_patched()
    
    # Load the model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*xFormers.*")
        try:
            # Try loading from local cache first (patched version)
            model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=verbose, source='local')
        except:
            # Fallback to default source
            model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=verbose)
    
    return model

