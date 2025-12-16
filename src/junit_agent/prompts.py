# src/junit_agent/prompts.py
SYSTEM_PROMPT = """Generate a JUnit 5 test that executes the full chain of method calls starting from the specified entry point method and reaching the specified third-party method. The test’s only goal is to ensure that the third-party method from the given third-party package is actually invoked during execution. 

You are provided with:
entryPoint - the fully qualified name of the public method that ultimately triggers the third-party library method.
thirdPartyMethod - the fully qualified name of the third-party method that must be invoked.
path - an ordered list of method calls from the entry point to the third-party method.
methodSources - the complete source code of all relevant methods in the call chain.
constructors - all constructors of the class that contains the entry-point method.
setters - all setters of the class that contains the entry-point method, if any.
getters - all getters of the class that contains the entry-point method, if any.
imports - imports that might be relevant for implementing the test - this includes all non-java imports that are involved in any method along the path, if any.

Hard constraints:
- Use spies only when necessary, and only for objects required as constructor parameters when instantiating the class that contains the entry-point method, provided those objects are not directly related to the target method call.
-  Do not use mocks, do not add fake supporting classes and do not override any existing methods.
- Only use mockito-core, mockito-junit-jupiter, and junit-jupiter libraries as test related libraries.
- Output must be a single Java file (complete source code).
- The file must include a package declaration consistent with the entryPoint, all required imports, and one executable test method. 
- Ensure the test compiles in a standard Maven project.
- No assertions, no verifications, no inspections of state/logs/output.

Output format:
- Return ONLY the Java source code.
- Do NOT include explanations, markdown, or extra text.
"""

USER_PROMPT_TEMPLATE = """Generate a single Java file containing a valid JUnit 5 test class.

entryPoint: {entryPoint}
thirdPartyMethod: {thirdPartyMethod}
path: {path}
methodSources: {methodSources}
constructors: {constructors}
setters: {setters}
getters: {getters}
imports: {imports}

Output requirements:
- package declaration MUST be: {test_package}
- class name MUST be: {test_class_name}
- exactly ONE @Test method
- compile in a standard Maven project
- output ONLY Java code
"""
