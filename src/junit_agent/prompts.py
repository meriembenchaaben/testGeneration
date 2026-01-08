
SYSTEM_PROMPT = """You are an expert Java developer who writes good unit tests in order to maximize code coverage. You are specialized in Maven and JUnit 5 testing framework."""

USER_PROMPT_TEMPLATE = """Generate a JUnit 5 test that executes the full chain of method calls starting from the specified entry point method and reaching the specified third-party method. No assertions, inspections, or verifications are required. The test’s only goal is to try invoking the given third-party method during execution. 

You are provided with:
entryPoint: Fully qualified public method where execution must begin.
thirdPartyMethod: Fully qualified third-party method that must be invoked.
path: Ordered list of method calls that must be traversed during execution.
methodSources: Complete and exact source code for all methods in the call chain.
constructors: All constructors of the class containing the entryPoint. Use one of these constructors to instantiate the class in the test. Then use that instance to call the entryPoint method.
fieldDeclarations: Instance variables and class variables of the class containing the entryPoint. These are the fields that can be accessed or set when creating the test.
setters: All setters of the entryPoint class that can modify the declared fields.
imports: All non-core-java imports that may be required by the test.

entryPoint: {entryPoint}
thirdPartyMethod: {thirdPartyMethod}
path: {path}
methodSources:
{methodSources}
constructors:
{constructors}
fieldDeclarations:
{fieldDeclarations}
setters:
{setters}
imports: {imports}

Hard constraints:
- The test class MUST NOT extend any other class (no 'extends' keyword).
- The test class MUST NOT override any methods (no @Override annotations).
- Do NOT create anonymous inner classes (e.g., new ClassName() {{ ... }}).
- Use real objects whenever possible.
- Use mocks only when necessary, and ONLY for objects required as constructor parameters when instantiating the class that contains the entry-point method, provided those objects are not directly related to the target method call.
- Do NOT mock or spy on the class under test.
- Do NOT alter, stub, or control the behavior of ANY method in the call path.
- Use only the following test-related libraries: junit-jupiter, mockito-core, and mockito-junit-jupiter. Do not use any other testing or mocking libraries.
- Do NOT add any assertions, verifications, or inspections of state/logs/output.
- package declaration MUST be: {test_package}
- class name MUST be: {test_class_name}
- exactly ONE @Test method
- Return ONLY the complete Java source code.
- Do NOT include explanations, markdown, or extra text.
"""
