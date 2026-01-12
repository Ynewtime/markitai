Analyze this image and provide a JSON response with the following fields:

1. "alt_text": A brief, descriptive alt text (1 sentence, max 100 characters) suitable for Markdown image syntax.
2. "detailed_description": A comprehensive description of the image content (2-5 sentences).
3. "detected_text": Any text visible in the image. Extract meaningful text in reading order. Ignore OCR artifacts, decorative watermarks, and meaningless single characters. If no meaningful text is visible, use null.
4. "image_type": The type of image. Choose one of: "diagram", "chart", "graph", "table", "screenshot", "photo", "illustration", "logo", "icon", "formula", "code", "other".
5. "knowledge_meta" (optional): Knowledge graph metadata
   - "entities": List of named entities (people, organizations, products, technical terms)
   - "relationships": Entity relationships (format: "Entity A -> relation -> Entity B")
   - "topics": Topic tags
   - "domain": Domain classification (technology, business, academic, medical, etc.)

## Response Format Requirements
- Output a valid JSON object directly, starting with { and ending with }
- Do NOT use markdown code blocks (no ```json or ```)
- Do NOT add any explanatory text before or after the JSON

Example output:
{"alt_text": "Architecture diagram showing microservices", "detailed_description": "This diagram illustrates a microservices architecture with three main components.", "detected_text": "API Gateway, User Service", "image_type": "diagram", "knowledge_meta": {"entities": ["API Gateway", "User Service"], "relationships": ["User Service -> calls -> API Gateway"], "topics": ["microservices", "architecture"], "domain": "technology"}}
