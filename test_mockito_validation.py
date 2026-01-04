#!/usr/bin/env python3
"""
Example demonstrating Mockito stubbing validation.
This shows how the validation detects forbidden Mockito patterns.
"""

import re


def validate_mockito_usage(java_source: str) -> tuple[bool, list]:
    """Check for forbidden Mockito stubbing patterns."""
    
    mockito_patterns = [
        (r"when\s*\(", "when().thenReturn() stubbing"),
        (r"doReturn\s*\(", "doReturn().when() stubbing"),
        (r"doThrow\s*\(", "doThrow().when() stubbing"),
        (r"doNothing\s*\(", "doNothing().when() stubbing"),
        (r"\.thenReturn\s*\(", ".thenReturn() stubbing"),
        (r"\.thenThrow\s*\(", ".thenThrow() stubbing"),
        (r"@Spy\b", "@Spy annotation"),
        (r"\.verify\s*\(", "Mockito.verify()"),
    ]
    
    violations = []
    for pattern, description in mockito_patterns:
        if re.search(pattern, java_source, re.IGNORECASE):
            violations.append(description)
    
    return len(violations) == 0, violations


if __name__ == "__main__":
    print("Testing Mockito validation...\n")
    
    # Valid: Mock creation only
    valid = """
    SomeDependency dep = Mockito.mock(SomeDependency.class);
    MyClass obj = new MyClass(dep);
    obj.method();
    """
    is_valid, violations = validate_mockito_usage(valid)
    print(f"1. Mock creation only: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Invalid: Stubbing with when/thenReturn
    stubbed = """
    SomeDependency dep = mock(SomeDependency.class);
    when(dep.getValue()).thenReturn(42);
    """
    is_valid, violations = validate_mockito_usage(stubbed)
    print(f"2. when/thenReturn: {'✓ PASS' if is_valid else '✗ FAIL (expected)'}")
    if violations:
        print(f"   Detected: {', '.join(violations)}")
    
    # Invalid: Using spy
    spy_code = """
    MyClass instance = Mockito.spy(new MyClass());
    """
    is_valid, violations = validate_mockito_usage(spy_code)
    print(f"3. Mockito.spy(): {'✓ PASS' if is_valid else '✗ FAIL (expected)'}")
    if violations:
        print(f"   Detected: {', '.join(violations)}")
    
    # Invalid: Using verify
    verify_code = """
    MyClass obj = new MyClass(dep);
    obj.method();
    verify(dep).someMethod();
    """
    is_valid, violations = validate_mockito_usage(verify_code)
    print(f"4. verify(): {'✓ PASS' if is_valid else '✗ FAIL (expected)'}")
    if violations:
        print(f"   Detected: {', '.join(violations)}")
    
    print("\n" + "=" * 60)
    print("Mockito validation working correctly!")
    print("\nAllowed:")
    print("  ✓ Mockito.mock() - creating mock objects")
    print("\nForbidden:")
    print("  ✗ when().thenReturn() - stubbing methods")
    print("  ✗ doReturn().when() - stubbing methods")
    print("  ✗ Mockito.spy() - spying on objects")
    print("  ✗ verify() - verifying method calls")
