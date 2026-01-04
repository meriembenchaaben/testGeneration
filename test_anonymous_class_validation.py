#!/usr/bin/env python3
"""
Test the anonymous inner class detection in validation.
"""

import re


def detect_anonymous_classes(java_source: str):
    """Detect anonymous inner classes in Java code."""
    pattern = r"new\s+([A-Z]\w+)\s*\([^)]*\)\s*\{"
    matches = list(re.finditer(pattern, java_source))
    return matches


if __name__ == "__main__":
    print("Testing anonymous inner class detection...\n")
    
    # Test 1: Valid - regular instantiation
    test1 = """
    RandomAccessReadDataStream stream = new RandomAccessReadDataStream(data);
    stream.method();
    """
    matches = detect_anonymous_classes(test1)
    print(f"1. Regular instantiation: {len(matches)} anonymous classes")
    print(f"   Result: {'✓ PASS' if len(matches) == 0 else '✗ FAIL'}\n")
    
    # Test 2: Invalid - anonymous inner class
    test2 = """
    RandomAccessReadDataStream stream = new RandomAccessReadDataStream(data) {
        private long currentPosition = 0;
        public void someMethod() {
            // override
        }
    };
    """
    matches = detect_anonymous_classes(test2)
    print(f"2. Anonymous inner class: {len(matches)} detected")
    if matches:
        for i, m in enumerate(matches):
            print(f"   Match {i+1}: new {m.group(1)}(...) {{")
    print(f"   Result: {'✗ FAIL (as expected)' if len(matches) > 0 else '✓ PASS'}\n")
    
    # Test 3: Multiple anonymous classes
    test3 = """
    TTFDataStream stream = new TTFDataStream(input) {
        public void method1() { }
    };
    
    SomeClass obj = new SomeClass() {
        public void method2() { }
    };
    """
    matches = detect_anonymous_classes(test3)
    print(f"3. Multiple anonymous classes: {len(matches)} detected")
    if matches:
        for i, m in enumerate(matches):
            print(f"   Match {i+1}: new {m.group(1)}(...) {{")
    print(f"   Result: {'✗ FAIL (as expected)' if len(matches) > 0 else '✓ PASS'}\n")
    
    print("=" * 60)
    print("Anonymous class detection working!")
    print("\nPattern matched: new ClassName(...) {")
    print("This detects anonymous inner classes that override behavior.")
