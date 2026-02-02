"""
AI Agent for X-Ray Analysis using multiple Vision LLMs.
Priority: Kimi AI -> Groq LLaVA -> Gemini -> OpenAI
"""
import os
import base64
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert radiologist AI assistant. Analyze the provided chest X-ray image and identify ALL visible abnormalities.

For each finding, provide:
1. **Condition Name** (e.g., Pneumonia, Pleural Effusion, Cardiomegaly)
2. **Confidence** (High/Medium/Low)
3. **Location** (e.g., Right Lower Lobe, Bilateral)
4. **Brief Explanation** (1-2 sentences)

IMPORTANT:
- List ALL conditions you observe, not just the primary one.
- If you see signs of BOTH Pneumonia AND Effusion, report BOTH.
- Be thorough but concise.
- If the image appears normal, state "No significant abnormalities detected."

Output your findings in the following JSON format:
{
    "findings": [
        {
            "condition": "Condition Name",
            "confidence": "High/Medium/Low",
            "location": "Location Description",
            "explanation": "Brief explanation"
        }
    ],
    "overall_impression": "One sentence summary of the X-ray",
    "recommendation": "Suggested next steps (e.g., consult physician, no action needed)"
}
"""

def encode_image_to_base64(image_path):
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_with_kimi(image_path):
    """Use Kimi AI (Moonshot) vision model."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("KIMI_API_KEY"),
        base_url="https://api.moonshot.cn/v1"
    )
    
    base64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    
    response = client.chat.completions.create(
        model="moonshot-v1-8k-vision-preview",  # Kimi vision model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT + "\n\nPlease analyze this chest X-ray and provide your findings in the specified JSON format."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def analyze_with_groq(image_path):
    """Use Groq with LLaVA vision model (FREE)."""
    from groq import Groq
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    base64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    
    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT + "\n\nPlease analyze this chest X-ray and provide your findings in the specified JSON format."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def analyze_with_gemini(image_path):
    """Use Google Gemini Vision API."""
    import google.generativeai as genai
    import PIL.Image
    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    img = PIL.Image.open(image_path)
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = SYSTEM_PROMPT + "\n\nPlease analyze this chest X-ray and provide your findings in the specified JSON format."
    response = model.generate_content([prompt, img])
    return response.text

def analyze_with_openai(image_path):
    """Use OpenAI GPT-4 Vision API."""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze this chest X-ray and provide your findings in the specified JSON format."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "high"}}
            ]}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content

def analyze_xray_with_agent(image_path):
    """
    Analyze an X-ray image using AI Vision.
    Priority: Kimi -> Groq (free) -> Gemini -> OpenAI
    """
    import json
    import re
    
    content = None
    provider = None
    last_error = None
    
    # Priority order of providers
    providers = [
        ("KIMI_API_KEY", analyze_with_kimi, "Kimi AI"),
        ("GROQ_API_KEY", analyze_with_groq, "Llama 3.2 Vision (Groq)"),
        ("GEMINI_API_KEY", analyze_with_gemini, "Gemini"),
        ("OPENAI_API_KEY", analyze_with_openai, "GPT-4"),
    ]
    
    for key_name, analyze_fn, name in providers:
        api_key = os.getenv(key_name)
        if api_key:
            try:
                content = analyze_fn(image_path)
                provider = name
                break
            except Exception as e:
                last_error = str(e)
                print(f"{name} failed: {e}")
                continue
    
    if content is None:
        return {"success": False, "error": last_error or "No API keys configured", "data": None}
    
    # Parse JSON from response
    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            findings = json.loads(json_match.group())
            return {"success": True, "data": findings, "provider": provider, "raw_response": content}
        else:
            return {"success": True, "data": {"raw_analysis": content}, "provider": provider, "raw_response": content}
    except json.JSONDecodeError:
        return {"success": True, "data": {"raw_analysis": content}, "provider": provider, "raw_response": content}

def get_agent_findings_summary(agent_result):
    """Convert agent findings to a user-friendly summary."""
    if not agent_result.get("success"):
        return f"AI Agent Error: {agent_result.get('error', 'Unknown error')}"
    
    data = agent_result.get("data", {})
    provider = agent_result.get("provider", "AI")
    
    if "raw_analysis" in data:
        return f"**{provider} Analysis:**\n{data['raw_analysis']}"
    
    findings = data.get("findings", [])
    if not findings:
        return f"**{provider}:** No significant abnormalities detected."
    
    summary_lines = [f"**{provider} Agent Findings:**"]
    for f in findings:
        condition = f.get("condition", "Unknown")
        confidence = f.get("confidence", "Unknown")
        location = f.get("location", "")
        explanation = f.get("explanation", "")
        
        line = f"- **{condition}** ({confidence} confidence)"
        if location:
            line += f" - {location}"
        if explanation:
            line += f"\n  _{explanation}_"
        summary_lines.append(line)
    
    if data.get("overall_impression"):
        summary_lines.append(f"\n**Overall:** {data['overall_impression']}")
    
    if data.get("recommendation"):
        summary_lines.append(f"**Recommendation:** {data['recommendation']}")
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Analyzing: {test_path}")
        result = analyze_xray_with_agent(test_path)
        print(get_agent_findings_summary(result))
    else:
        print("Usage: python ai_agent.py <path_to_xray_image>")
