#!/usr/bin/env python3
"""
Module for parsing JaCoCo coverage reports, and check if a given third-party methods are covered by tests
"""

import logging
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


def check_method_in_lines(covered_lines: Set[str], target_class: str, method_name: str) -> bool:
    """
    Check if a specific method appears in the covered lines.
    Args:
        covered_lines: Set of covered code lines
        target_class: Short class name of the target 
        method_name: Method name to search for 
    Returns:
        bool: True if method is found in covered lines
    """
    if method_name == "<init>":
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
            error="JaCoCo report directory not found"
        )
    html_file = get_html_file_path(jacoco_dir, method_class)
    if not html_file:
        return CoverageResult(
            method_covered=False,
            method_class=method_class,
            target_method=target_method,
            jacoco_report_path=jacoco_dir,
            error=f"Coverage HTML file not found for class: {method_class}"
        )
    covered_lines = parse_covered_lines(html_file)
    if not covered_lines:
        return CoverageResult(
            method_covered=False,
            method_class=method_class,
            target_method=target_method,
            jacoco_report_path=jacoco_dir,
            error="No covered lines found in HTML report",
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
    is_covered = check_method_in_lines(
        covered_lines,
        target_short_class,
        target_method_name
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
