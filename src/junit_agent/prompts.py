# src/junit_agent/prompts.py
SYSTEM_PROMPT = """You generate minimal JUnit 5 tests that trigger a specific third-party method call.

Hard constraints:
- Output must be a single Java file (complete source code).
- Exactly one executable @Test method.
- No assertions, no verifications, no inspections of state/logs/output.
- Do not mock or stub any object that has data/control dependence on the target third-party method.
  Use real instances unless unrelated to the invocation path.
- It is acceptable to use mocks for unrelated objects.
- Ensure the test compiles in a standard Maven project.

Output format:
- Return ONLY the Java source code.
- Do NOT include explanations, markdown, or extra text.
"""

USER_PROMPT_TEMPLATE = """Generate a single Java file containing a valid JUnit 5 test class.

entryPoint: {entryPoint}
thirdPartyMethod: {thirdPartyMethod}
path: {path}

fullMethods:
{fullMethods}

Output requirements:
- package declaration MUST be: {test_package}
- class name MUST be: {test_class_name}
- exactly ONE @Test method
- compile in a standard Maven project
- output ONLY Java code
"""
