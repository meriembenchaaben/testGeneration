#!/usr/bin/env python3
"""
Module for parsing JaCoCo coverage reports, and check if a given third-party methods are covered by tests
"""

import logging
import re
from pathlib import Path
from html.parser import HTMLParser
from typing import Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CoverageResult:
    """Result of coverage analysis for a specific method"""
    method_covered: bool
    method_class: str
    target_method: str
    jacoco_report_path: Optional[Path]
    error: Optional[str] = None
    total_covered_lines: int = 0


class JaCoCoHTMLParser(HTMLParser):
    """An HTML parser to extract covered code lines from JaCoCo reports"""
    def __init__(self):
        super().__init__()
        self.covered_lines = set()
        self.in_covered_span = False
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        if tag == 'span':
            attr_dict = dict(attrs)
            # Check if this span has an id starting with 'L' and class contains 'fc'
            # 'fc' means "fully covered" in JaCoCo
            span_id = attr_dict.get('id', '')
            span_class = attr_dict.get('class', '')
            if span_id.startswith('L') and 'fc' in span_class:
                self.in_covered_span = True
                self.current_text = []
    def handle_endtag(self, tag):
        if tag == 'span' and self.in_covered_span:
            code_line = ''.join(self.current_text).strip()
            if code_line:
                self.covered_lines.add(code_line)
            self.in_covered_span = False
            self.current_text = []
    def handle_data(self, data):
        if self.in_covered_span:
            self.current_text.append(data)


def find_jacoco_report(repo_root: Path) -> Optional[Path]:
    """
    Find the JaCoCo HTML report directory.
    Args:
        repo_root: Root directory of the Maven project
    Returns:
        Path to the JaCoCo HTML report directory, or None if not found
    """
    jacoco_dir = repo_root / "target" / "site" / "jacoco"
    if jacoco_dir.exists() and jacoco_dir.is_dir():
        logger.info(f"Found JaCoCo report at: {jacoco_dir}")
        return jacoco_dir
    logger.warning(f"JaCoCo report directory not found at: {jacoco_dir}")
    return None


def get_html_file_path(jacoco_dir: Path, method_class: str) -> Optional[Path]:
    """
    Get the path to the HTML file for a specific class in the JaCoCo report.
    Args:
        jacoco_dir: Path to JaCoCo report directory
        method_class: Fully qualified class name (e.g., "com.example.MyClass")
    Returns:
        Path to the HTML file, or None if not found
    """
    if '.' in method_class:
        package_name = method_class.rsplit('.', 1)[0]
        simple_class_name = method_class.rsplit('.', 1)[1]
    else:
        package_name = ""
        simple_class_name = method_class
    if package_name:
        html_file = jacoco_dir / package_name / f"{simple_class_name}.java.html"
    else:
        html_file = jacoco_dir / f"{simple_class_name}.java.html"
    if html_file.exists():
        logger.info(f"Found coverage HTML file: {html_file}")
        return html_file
    logger.warning(f"Coverage HTML file not found: {html_file}")
    return None


def parse_covered_lines(html_file: Path) -> Set[str]:
    """
    Parse JaCoCo HTML file and extract all covered code lines.
    Args:
        html_file: Path to the JaCoCo HTML report file
    Returns:
        Set of covered code lines
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        parser = JaCoCoHTMLParser()
        parser.feed(html_content)
        logger.info(f"Parsed {len(parser.covered_lines)} covered code lines")
        return parser.covered_lines
    except Exception as e:
        logger.error(f"Failed to parse HTML file {html_file}: {e}")
        return set()


def check_extends_relationship(repo_root: Path, method_class: str, target_class_fqn: str) -> bool:
    """
    Check if method_class extends target_class by examining the source file.
    Args:
        repo_root: Root directory of the Maven project
        method_class: Fully qualified name of the subclass
        target_class_fqn: Fully qualified name of the potential superclass
    Returns:
        bool: True if method_class extends target_class_fqn
    """
    try:
        # Find the source file for method_class
        if '.' in method_class:
            package_name = method_class.rsplit('.', 1)[0]
            simple_class_name = method_class.rsplit('.', 1)[1]
        else:
            package_name = ""
            simple_class_name = method_class
        
        # Try to find the Java source file
        src_dirs = [
            repo_root / "src" / "main" / "java",
            repo_root / "src" / "test" / "java"
        ]
        
        for src_dir in src_dirs:
            if package_name:
                java_file = src_dir / package_name.replace('.', '/') / f"{simple_class_name}.java"
            else:
                java_file = src_dir / f"{simple_class_name}.java"
            
            if java_file.exists():
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract the simple name of the target class
                target_simple_name = target_class_fqn.rsplit('.', 1)[1] if '.' in target_class_fqn else target_class_fqn
                
                # Look for extends clause
                # Pattern: class ClassName extends SuperClassName
                extends_pattern = rf'class\s+{re.escape(simple_class_name)}\s+extends\s+(\w+)'
                match = re.search(extends_pattern, content)
                
                if match:
                    extended_class = match.group(1)
                    # Check if the extended class matches the target (by simple name)
                    if extended_class == target_simple_name:
                        logger.info(f"Found extends relationship: {method_class} extends {target_class_fqn}")
                        return True
                    # Also check if it matches the fully qualified name
                    if extended_class == target_class_fqn:
                        logger.info(f"Found extends relationship: {method_class} extends {target_class_fqn}")
                        return True
                
                logger.debug(f"No extends relationship found for {method_class} and {target_class_fqn}")
                return False
        
        logger.warning(f"Source file not found for class: {method_class}")
        return False
        
    except Exception as e:
        logger.error(f"Error checking extends relationship: {e}")
        return False


def check_method_in_lines(covered_lines: Set[str], target_class: str, method_name: str, check_super_call: bool = False) -> bool:
    """
    Check if a specific method appears in the covered lines.
    Args:
        covered_lines: Set of covered code lines
        target_class: Short class name of the target 
        method_name: Method name to search for
        check_super_call: If True, look for super(...) calls instead of new ClassName(...)
    Returns:
        bool: True if method is found in covered lines
    """
    if method_name == "<init>":
        if check_super_call:
            # Look for super(...) constructor calls
            pattern = "super("
            for line in covered_lines:
                if pattern in line:
                    logger.debug(f"Found super constructor call: {line}")
                    return True
        else:
            # Look for regular constructor calls
            pattern = f"new {target_class}("
            for line in covered_lines:
                if pattern in line:
                    logger.debug(f"Found constructor call: {line}")
                    return True
    elif method_name == "<clinit>":
        for line in covered_lines:
            if target_class in line:
                logger.debug(f"Found static initializer usage: {line}")
                return True
    else:
        # Regular method - look for method name
        # Be more specific: look for method name followed by '('
        pattern1 = f"{method_name}("
        pattern2 = f".{method_name}("
        for line in covered_lines:
            if pattern1 in line or pattern2 in line:
                logger.debug(f"Found method call: {line}")
                return True
    return False


def get_coverage_result(
    repo_root: Path,
    method_class: str,
    target_method: str
) -> CoverageResult:
    """
    Check if a target third-party method is covered by tests using JaCoCo report.
    Args:
        repo_root: Root directory of the Maven project
        method_class: Fully qualified name of the class being tested
        target_method: Fully qualified third-party method to check
    Returns:
        CoverageResult with details about whether the method is covered
    """
    # Find JaCoCo report directory
    jacoco_dir = find_jacoco_report(repo_root)
    if not jacoco_dir:
        return CoverageResult(
            method_covered=False,
            method_class=method_class,
            target_method=target_method,
            jacoco_report_path=None,
            error="Method is not covered: JaCoCo report not generated"
        )
    html_file = get_html_file_path(jacoco_dir, method_class)
    if not html_file:
        return CoverageResult(
            method_covered=False,
            method_class=method_class,
            target_method=target_method,
            jacoco_report_path=jacoco_dir,
            error=f"Method is not covered: Jacoco HTML report for class not generated"
        )
    covered_lines = parse_covered_lines(html_file)
    if not covered_lines:
        return CoverageResult(
            method_covered=False,
            method_class=method_class,
            target_method=target_method,
            jacoco_report_path=jacoco_dir,
            error="Method is not covered: No covered lines found in report",
            total_covered_lines=0
        )
    try:
        target_class_fqn = target_method.rsplit('.', 1)[0]
        target_short_class = target_class_fqn.rsplit('.', 1)[1] if '.' in target_class_fqn else target_class_fqn
        target_method_name = target_method.rsplit('.', 1)[1]
    except Exception as e:
        return CoverageResult(
            method_covered=False,
            method_class=method_class,
            target_method=target_method,
            jacoco_report_path=jacoco_dir,
            error=f"Failed to parse target method: {e}",
            total_covered_lines=len(covered_lines)
        )
    
    # Corner case: If target is a constructor (<init>), check if it's a superclass constructor
    check_super_call = False
    if target_method_name == "<init>":
        # Check if method_class extends target_class_fqn
        if check_extends_relationship(repo_root, method_class, target_class_fqn):
            logger.info(f"Detected superclass constructor call: {method_class} extends {target_class_fqn}")
            check_super_call = True
    
    is_covered = check_method_in_lines(
        covered_lines,
        target_short_class,
        target_method_name,
        check_super_call=check_super_call
    )
    return CoverageResult(
        method_covered=is_covered,
        method_class=method_class,
        target_method=target_method,
        jacoco_report_path=jacoco_dir,
        error=None,
        total_covered_lines=len(covered_lines)
    )

def print_coverage_summary(result: CoverageResult) -> None:
    if result.error:
        print(f"Error: {result.error}")
        print("Status: COVERAGE CHECK FAILED")
    elif result.method_covered:
        print("Status: METHOD IS COVERED")
    else:
        print("Status: METHOD IS NOT COVERED")
