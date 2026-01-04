#!/usr/bin/env python3
"""
Example script demonstrating the hard constraint validation.
This shows how the validation catches extended classes and overridden methods.
"""

import re


def _validate_hard_constraints(java_source: str, test_class_name: str) -> tuple[bool, str]:
    """
    Validate hard constraints: no extended classes or overridden methods.
    
    Args:
        java_source: The generated Java test source code
        test_class_name: Expected test class name
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all constraints pass, False otherwise
        - error_message: Empty string if valid, detailed error message otherwise
    """
    violations = []
    
    # Check for class extension (class TestName extends SomeClass)
    extends_pattern = rf"class\s+{re.escape(test_class_name)}\s+extends\s+\w+"
    if re.search(extends_pattern, java_source):
        match = re.search(extends_pattern, java_source)
        violations.append(
            f"HARD CONSTRAINT VIOLATION: Test class extends another class.\n"
            f"  Found: {match.group(0)}\n"
            f"  Requirement: Test class must not extend any other class.\n"
            f"  Fix: Remove the 'extends' clause from the test class declaration."
        )
    
    # Check for @Override annotations
    override_pattern = r"@Override\s*\n\s*(?:public|protected|private)?\s*\w+\s+\w+\s*\("
    override_matches = list(re.finditer(override_pattern, java_source, re.MULTILINE))
    if override_matches:
        violations.append(
            f"HARD CONSTRAINT VIOLATION: Test contains overridden method(s).\n"
            f"  Found {len(override_matches)} @Override annotation(s).\n"
            f"  Requirement: Test methods must not override any methods.\n"
            f"  Fix: Remove all @Override annotations and ensure methods are not overriding parent methods."
        )
        
        # Show examples of violations (up to 3)
        for i, match in enumerate(override_matches[:3]):
            start = max(0, match.start() - 50)
            end = min(len(java_source), match.end() + 100)
            context = java_source[start:end].strip()
            violations.append(f"  Example {i+1}:\n    {context[:150]}...")
    
    if violations:
        error_msg = "\n" + "=" * 80 + "\n"
        error_msg += "HARD CONSTRAINT VALIDATION FAILED\n"
        error_msg += "=" * 80 + "\n\n"
        error_msg += "\n\n".join(violations)
        error_msg += "\n\n" + "=" * 80 + "\n"
        error_msg += "Please regenerate the test without these violations.\n"
        error_msg += "Remember: The test class must be a standalone class that does not extend\n"
        error_msg += "any other class and does not override any methods.\n"
        error_msg += "=" * 80 + "\n"
        return False, error_msg
    
    return True, ""


# Test cases
if __name__ == "__main__":
    print("Testing hard constraint validation...\n")
    
    # Test 1: Valid test (no violations)
    valid_test = """
package generated;

import org.junit.jupiter.api.Test;

public class GeneratedReachabilityTest {
    @Test
    public void testMethod() {
        // Test code here
    }
}
"""
    
    is_valid, error = _validate_hard_constraints(valid_test, "GeneratedReachabilityTest")
    print("Test 1 - Valid test:")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if not is_valid:
        print(f"  Error:\n{error}")
    print()
    
    # Test 2: Test with extended class
    extended_test = """
package generated;

import org.junit.jupiter.api.Test;

public class GeneratedReachabilityTest extends BaseTestClass {
    @Test
    public void testMethod() {
        // Test code here
    }
}
"""
    
    is_valid, error = _validate_hard_constraints(extended_test, "GeneratedReachabilityTest")
    print("Test 2 - Test with extended class:")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL (as expected)'}")
    if not is_valid:
        print(f"  Error message preview:\n{error[:300]}...")
    print()
    
    # Test 3: Test with @Override annotation
    override_test = """
package generated;

import org.junit.jupiter.api.Test;

public class GeneratedReachabilityTest {
    @Override
    public void setUp() {
        // Setup code
    }
    
    @Test
    public void testMethod() {
        // Test code here
    }
}
"""
    
    is_valid, error = _validate_hard_constraints(override_test, "GeneratedReachabilityTest")
    print("Test 3 - Test with @Override:")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL (as expected)'}")
    if not is_valid:
        print(f"  Error message preview:\n{error[:300]}...")
    print()
    
    # Test 4: Test with both violations
    both_violations_test = """
package generated;

import org.junit.jupiter.api.Test;

public class GeneratedReachabilityTest extends BaseTestClass {
    @Override
    public void setUp() {
        // Setup code
    }
    
    @Override
    public void tearDown() {
        // Teardown code
    }
    
    @Test
    public void testMethod() {
        // Test code here
    }
}
"""
    
    is_valid, error = _validate_hard_constraints(both_violations_test, "GeneratedReachabilityTest")
    print("Test 4 - Test with both violations:")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL (as expected)'}")
    if not is_valid:
        print(f"  Error message preview:\n{error[:400]}...")
    print()
    
    print("=" * 80)
    print("All validation tests completed!")
