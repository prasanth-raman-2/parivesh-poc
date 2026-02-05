import json
import litellm
from app.summarize.tools.summarize_tools import (
    FUNCTION_SCHEMAS,
    read_file, write_file, create_directory, append_file, 
    list_directory, copy_file, edit_file, edit_file_batch, 
    file_exists, get_file_info, read_lines, search_in_file
)

file_path = "/home/prasa/projects/negd/parivesh-poc/docs/eia_report.md"
output_path = "/home/prasa/projects/negd/parivesh-poc/docs/summary.md"

EIA_SUMMARY_TEMPLATE = """# Environmental Impact Assessment (EIA) Summary

## Document Information
- **Source File:** {file_path}
- **Total Lines:** {{TOTAL_LINES}}
- **Generated On:** {{GENERATION_DATE}}

---

## 1. Executive Summary
{{EXECUTIVE_SUMMARY}}

---

## 2. Project Overview
### 2.1 Project Description
{{PROJECT_DESCRIPTION}}

### 2.2 Project Location
{{PROJECT_LOCATION}}

### 2.3 Project Objectives
{{PROJECT_OBJECTIVES}}

---

## 3. Environmental Baseline
### 3.1 Physical Environment
{{PHYSICAL_ENVIRONMENT}}

### 3.2 Biological Environment
{{BIOLOGICAL_ENVIRONMENT}}

### 3.3 Socio-Economic Environment
{{SOCIO_ECONOMIC_ENVIRONMENT}}

---

## 4. Impact Assessment
### 4.1 Air Quality Impacts
{{AIR_QUALITY_IMPACTS}}

### 4.2 Water Resource Impacts
{{WATER_RESOURCE_IMPACTS}}

### 4.3 Noise and Vibration Impacts
{{NOISE_VIBRATION_IMPACTS}}

### 4.4 Land Use and Soil Impacts
{{LAND_SOIL_IMPACTS}}

### 4.5 Biodiversity Impacts
{{BIODIVERSITY_IMPACTS}}

### 4.6 Socio-Economic Impacts
{{SOCIO_ECONOMIC_IMPACTS}}

---

## 5. Mitigation Measures
### 5.1 Environmental Mitigation
{{ENVIRONMENTAL_MITIGATION}}

### 5.2 Social Mitigation
{{SOCIAL_MITIGATION}}

---

## 6. Environmental Management Plan (EMP)
### 6.1 Monitoring Framework
{{MONITORING_FRAMEWORK}}

### 6.2 Implementation Schedule
{{IMPLEMENTATION_SCHEDULE}}

### 6.3 Budget Allocation
{{BUDGET_ALLOCATION}}

---

## 7. Public Consultation
{{PUBLIC_CONSULTATION}}

---

## 8. Risk Assessment
### 8.1 Identified Risks
{{IDENTIFIED_RISKS}}

### 8.2 Emergency Response Plan
{{EMERGENCY_RESPONSE}}

---

## 9. Regulatory Compliance
{{REGULATORY_COMPLIANCE}}

---

## 10. Conclusions and Recommendations
### 10.1 Key Findings
{{KEY_FINDINGS}}

### 10.2 Recommendations
{{RECOMMENDATIONS}}

---

## Appendix: Key Data Points
{{KEY_DATA_POINTS}}
""".format(file_path=file_path)

SYSTEM_PROMPT = """You are an expert Environmental Impact Assessment (EIA) analyst. Your task is to thoroughly analyze and summarize an EIA report, ensuring NO key information is missed.

## CRITICAL INSTRUCTIONS:

### File Reading Strategy (MANDATORY for large files):
1. FIRST: Use `get_file_info` to check file size
2. THEN: Use `read_lines` to read the file in chunks of 200-300 lines at a time
3. Process the ENTIRE file systematically - DO NOT skip any sections
4. Track your progress: which lines you've read and summarized

### Workflow:
1. **Initialize**: 
   - Get file info to determine total lines
   - Create the summary template at: {output_path}
   
2. **Systematic Reading** (repeat until entire file is processed):
   - Read lines in chunks (e.g., lines 1-300, then 301-600, etc.)
   - For each chunk, identify which template sections the content belongs to
   - Extract key information, data points, statistics, and findings
   
3. **Progressive Template Filling**:
   - After processing each chunk, use `edit_file` to replace the placeholder (e.g., {{{{EXECUTIVE_SUMMARY}}}}) with actual content
   - Use `search_in_file` to find specific data if needed
   - Each placeholder should be replaced with comprehensive content

### Template Placeholders to Fill:
- {{{{TOTAL_LINES}}}} - Total line count
- {{{{GENERATION_DATE}}}} - Current date
- {{{{EXECUTIVE_SUMMARY}}}} - High-level overview (3-5 paragraphs)
- {{{{PROJECT_DESCRIPTION}}}} - What the project is about
- {{{{PROJECT_LOCATION}}}} - Geographic details, coordinates, region
- {{{{PROJECT_OBJECTIVES}}}} - Goals and purposes
- {{{{PHYSICAL_ENVIRONMENT}}}} - Climate, topography, geology, hydrology
- {{{{BIOLOGICAL_ENVIRONMENT}}}} - Flora, fauna, ecosystems
- {{{{SOCIO_ECONOMIC_ENVIRONMENT}}}} - Demographics, economy, land use
- {{{{AIR_QUALITY_IMPACTS}}}} - Emissions, pollutants, air quality changes
- {{{{WATER_RESOURCE_IMPACTS}}}} - Water usage, discharge, quality impacts
- {{{{NOISE_VIBRATION_IMPACTS}}}} - Noise levels, affected areas
- {{{{LAND_SOIL_IMPACTS}}}} - Soil erosion, contamination, land use change
- {{{{BIODIVERSITY_IMPACTS}}}} - Species affected, habitat loss
- {{{{SOCIO_ECONOMIC_IMPACTS}}}} - Employment, displacement, community effects
- {{{{ENVIRONMENTAL_MITIGATION}}}} - Measures to reduce environmental harm
- {{{{SOCIAL_MITIGATION}}}} - Measures to address social concerns
- {{{{MONITORING_FRAMEWORK}}}} - How impacts will be monitored
- {{{{IMPLEMENTATION_SCHEDULE}}}} - Timeline for EMP implementation
- {{{{BUDGET_ALLOCATION}}}} - Financial provisions for environmental management
- {{{{PUBLIC_CONSULTATION}}}} - Stakeholder engagement, public hearings
- {{{{IDENTIFIED_RISKS}}}} - Environmental and social risks
- {{{{EMERGENCY_RESPONSE}}}} - Contingency plans
- {{{{REGULATORY_COMPLIANCE}}}} - Laws, permits, clearances required
- {{{{KEY_FINDINGS}}}} - Main conclusions from the assessment
- {{{{RECOMMENDATIONS}}}} - Suggested actions and conditions
- {{{{KEY_DATA_POINTS}}}} - Important statistics, measurements, thresholds

### Quality Requirements:
- Include SPECIFIC numbers, statistics, and measurements when available
- Preserve technical terminology and scientific data
- Note any gaps or missing information in the original report
- Each section should be detailed (minimum 2-3 sentences, more for complex sections)
- If information for a section is not found, write "Information not available in the source document"

### Output Path: {output_path}

Begin by getting the file info, then create the template, then systematically read and process the entire document.""".format(output_path=output_path)

MAX_ITERATION = 50  # Increased for large file processing

# Map function names to their implementations
FUNCTION_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "create_directory": create_directory,
    "list_directory": list_directory,
    "copy_file": copy_file,
    "edit_file": edit_file,
    "edit_file_batch": edit_file_batch,
    "file_exists": file_exists,
    "get_file_info": get_file_info,
    "read_lines": read_lines,
    "search_in_file": search_in_file,
}

# Build tools list
tools = [
    {"type": "function", "function": schema}
    for schema in FUNCTION_SCHEMAS
]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user", 
        "content": f"""Please create a comprehensive summary of the EIA report.

**Source file:** {file_path}
**Output file:** {output_path}

**Summary Template to use:**
```markdown
{EIA_SUMMARY_TEMPLATE}
```

Instructions:
1. First, get file info to know the total lines
2. Create the template file at the output path
3. Read the source file in chunks of 200-300 lines using read_lines
4. For each chunk, identify relevant information for each template section
5. Use edit_file to replace each placeholder with extracted content
6. Continue until the ENTIRE source file has been processed
7. Ensure all placeholders are replaced with actual content or "Information not available"

Start now by checking the file info."""
    }
]

print("=" * 60)
print("EIA Report Summarization Agent")
print("=" * 60)
print(f"Source: {file_path}")
print(f"Output: {output_path}")
print("=" * 60)

for i in range(MAX_ITERATION):
    response = litellm.completion(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    print(f"\n{'─' * 40}")
    print(f"Iteration {i + 1}/{MAX_ITERATION}")
    print('─' * 40)
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # Always append assistant message to conversation
    messages.append(response_message)
    
    # Print assistant's reasoning if any
    if response_message.content:
        print(f"\nAssistant: {response_message.content[:500]}{'...' if len(response_message.content or '') > 500 else ''}")
    
    if not tool_calls:
        print("\n" + "=" * 60)
        print("SUMMARIZATION COMPLETE")
        print("=" * 60)
        print(f"\nFinal Response:\n{response_message.content}")
        print(f"\nSummary saved to: {output_path}")
        break
    
    # Process each tool call
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Execute the function using the map
        func = FUNCTION_MAP.get(function_name)
        if func:
            try:
                function_response = func(**function_args)
            except Exception as e:
                function_response = f"Error executing {function_name}: {str(e)}"
        else:
            function_response = f"Function {function_name} not recognized."
        
        # Convert response to string for the API
        if function_response is None:
            function_response = "Success (no return value)"
        elif not isinstance(function_response, str):
            function_response = json.dumps(function_response, indent=2, default=str)
        
        # Log tool execution
        print(f"\n  📌 Tool: {function_name}")
        if function_name == "read_lines":
            start = function_args.get('start_line', 1)
            end = function_args.get('end_line', 'EOF')
            print(f"     Reading lines {start} to {end}")
            print(f"     Retrieved {len(function_response)} characters")
        elif function_name == "edit_file":
            search_preview = function_args.get('search_text', '')[:50]
            print(f"     Replacing: '{search_preview}...'")
            print(f"     Result: {function_response}")
        elif function_name == "write_file":
            print(f"     Writing to: {function_args.get('file_path', 'unknown')}")
            print(f"     Content length: {len(function_args.get('content', ''))} chars")
        elif function_name == "get_file_info":
            print(f"     File: {function_args.get('file_path', 'unknown')}")
            if isinstance(function_response, str):
                try:
                    info = json.loads(function_response)
                    print(f"     Size: {info.get('size', 'N/A')} bytes")
                except:
                    pass
        else:
            print(f"     Args: {json.dumps(function_args)[:100]}")
        
        # Append tool response with required tool_call_id
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": function_response
        })

else:
    print(f"\n⚠️  Max iterations ({MAX_ITERATION}) reached.")
    print("The summarization may be incomplete. Consider increasing MAX_ITERATION.")

# Final verification
print("\n" + "=" * 60)
print("POST-PROCESSING CHECK")
print("=" * 60)

if file_exists(output_path):
    info = get_file_info(output_path)
    print(f"✓ Output file created: {output_path}")
    print(f"  Size: {info['size']} bytes")
    
    # Check for unfilled placeholders
    content = read_file(output_path)
    import re
    unfilled = re.findall(r'\{\{[A-Z_]+\}\}', content)
    if unfilled:
        print(f"\n⚠️  Unfilled placeholders found ({len(unfilled)}):")
        for placeholder in set(unfilled):
            print(f"   - {placeholder}")
    else:
        print("✓ All placeholders filled")
else:
    print(f"✗ Output file not found: {output_path}")