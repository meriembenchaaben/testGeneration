#!/usr/bin/env python3
"""
Module for writing test files to Maven project structure.
"""

import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def get_test_class_fqn(test_content: str, fallback_filename: Optional[str] = None) -> str:
    """
    Extract the fully qualified test class name from test file content.  
    Args:
        test_content: The Java test file content as string
        fallback_filename: Optional filename to use if class name can't be found
    Returns:
        Fully qualified class name
    """
    package = ""
    class_name = None
    for line in test_content.split('\n'):
        line = line.strip()
        if line.startswith('package '):
            package = line.replace('package ', '').replace(';', '').strip()
            break
    # Find class name from class declaration
    # Matches: public class ClassName, class ClassName, public final class ClassName, etc.
    class_pattern = r'\b(?:public\s+)?(?:final\s+)?(?:abstract\s+)?class\s+(\w+)'
    class_match = re.search(class_pattern, test_content)
    if class_match:
        class_name = class_match.group(1)
    elif fallback_filename:
        class_name = fallback_filename.replace('.java', '')
        logger.warning(f"Could not find class declaration, using filename: {class_name}")
    else:
        raise ValueError("Could not determine class name from test content")
    
    if package:
        return f"{package}.{class_name}"
    return class_name


def get_test_destination_path(repo_root: Path, test_class_fqn: str) -> Path:
    """
    Determine the destination path for a test file in the Maven project structure.
    Args:
        repo_root: Root directory of the Maven project
        test_class_fqn: Fully qualified test class name
    Returns:
        Path to the destination directory where the test file should be placed
    """
    print(repo_root)
    if 'immutables' in str(repo_root):
        test_base = repo_root / "test"
    else:
        test_base = repo_root / "src" / "test" / "java"
    if not test_base.exists():
        raise ValueError(f"Maven test directory not found: {test_base}")
    # The idea here is to put the file in the correct package if a package name exists, otherwise directly under test_base
    if '.' in test_class_fqn:
        package = test_class_fqn.rsplit('.', 1)[0]
        package_path = package.replace('.', '/')
        destination_dir = test_base / package_path
    else:
        destination_dir = test_base
    
    return destination_dir


def write_test_file(repo_root: Path, rel_path: str, content: str) -> Path:
    """
    Write the generated Java test file to the Maven repository.
    Args:
        repo_root: Root directory of the Maven project
        rel_path: Relative path hint (e.g., "TestClass.java") - used as filename
        content: Complete Java test file content as string
    Returns:
        Path: Absolute path to the written test file
    Raises:
        ValueError: If Maven test directory structure doesn't exist or class name can't be determined
    """
    filename = Path(rel_path).name
    test_class_fqn = get_test_class_fqn(content, fallback_filename=filename)
    logger.info(f"Determined test class: {test_class_fqn}")
    destination_dir = get_test_destination_path(repo_root, test_class_fqn)
    destination_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory structure: {destination_dir}")
    abs_path = destination_dir / filename
    abs_path.write_text(content, encoding="utf-8")
    logger.info(f"Wrote test file to: {abs_path}")
    return abs_path


def cleanup_test_file(test_file_path: Path) -> None:
    """
    Remove a test file and clean up empty parent directories.
    Args:
        test_file_path: Path to the test file to remove
    """
    if test_file_path.exists():
        test_file_path.unlink()
        logger.info(f"Removed test file: {test_file_path}")
        try:
            parent = test_file_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                logger.info(f"Removed empty directory: {parent}")
        except:
            pass
