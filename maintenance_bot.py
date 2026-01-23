import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BLUE = "\033[94m"
RED = "\033[91m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

# --- OpenAI Client Setup ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=openai_api_key)
MODEL = "gpt-4o-mini"

# --- Sub-function: Generate Repair Instructions for NG Items ---
def generate_repair_instructions(diagnostics_ng):
    """
    Generate actionable repair advice for failed/issue components.
    
    Args:
        diagnostics_ng (list): List of issue messages (e.g., ["❌ mirrors: missing/broken"])
        model (str): OpenAI model to use
        
    Returns:
        str: Plain-text repair instructions or fallback message
    """
    if not diagnostics_ng:
        logger.info("No NG items provided — returning default success message.")
        return "No issues detected — all critical components are functional."

    ng_list_str = "\n".join(diagnostics_ng)
    fix_prompt = f"""You are a certified commercial truck maintenance expert specializing in fleet safety and DOT compliance.
    The following issues were detected during a visual inspection of a heavy-duty truck:

    {ng_list_str}

    For each issue:
    - Identify the specific component mentioned (e.g., 'front lights', 'wipers', 'plate number').
    - Provide clear, actionable, and safety-compliant steps to resolve it.
    - Include part replacement tips, cleaning procedures, or regulatory considerations if relevant.
    - Keep advice practical for technicians or fleet managers.

    Respond in plain text with bullet points starting with '-'. Do not use markdown."""

    try:
        logger.info("Sending NG diagnostics to LLM for repair recommendations...")
        fix_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.2,
            max_tokens=500
        )
        result = fix_response.choices[0].message.content.strip()
        logger.info("Successfully generated repair instructions.")
        return result
    except Exception as e:
        error_msg = f"⚠️ Error generating fixes: {str(e)}"
        logger.error(f"LLM call failed for repair instructions: {e}")
        return error_msg

# --- Sub-function: Generate Maintenance Tips for OK Items ---
def generate_maintenance_tips(diagnostics_ok):
    """
    Generate preventive maintenance advice for passing components.
    
    Args:
        diagnostics_ok (list): List of OK messages (e.g., ["✅ wipers: ok"])
        model (str): OpenAI model to use
        
    Returns:
        str: Plain-text maintenance tips or fallback message
    """
    if not diagnostics_ok:
        logger.info("No OK items provided — returning default maintenance message.")
        return "No passing components reported — unable to provide maintenance advice."

    ok_list_str = "\n".join(diagnostics_ok)
    maintain_prompt = f"""You are a certified commercial truck maintenance expert focused on preventive care and longevity of vehicle systems.
    The following components passed inspection on a commercial truck:

    {ok_list_str}

    For each component:
    - Suggest best practices to maintain it in optimal condition.
    - Recommend inspection frequency (e.g., daily, weekly, every 10k miles).
    - Mention signs of early wear to watch for.
    - Include cleaning, lubrication, or alignment tips if applicable.

    Respond in plain text with bullet points starting with '-'. Do not use markdown."""

    try:
        logger.info("Sending OK diagnostics to LLM for maintenance tips...")
        maintain_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": maintain_prompt}],
            temperature=0.5,
            max_tokens=500
        )
        result = maintain_response.choices[0].message.content.strip()
        logger.info("Successfully generated maintenance tips.")
        return result
    except Exception as e:
        error_msg = f"⚠️ Error generating maintenance tips: {str(e)}"
        logger.error(f"LLM call failed for maintenance tips: {e}")
        return error_msg
    
# --- Main Orchestrator Function ---
def main(diagnostics_ok, diagnostics_ng, model=MODEL):
    """
    Generate repair instructions and maintenance tips.
    
    Args:
        diagnostics_ok (list): Passing component messages
        diagnostics_ng (list): Issue/failure messages
        model (str): OpenAI model to use
        
    Returns:
        dict: {
            "fixes": str,
            "maintenance": str
        }
    """
    logger.info("Starting main maintenance analysis...")
    try:
        fixes = generate_repair_instructions(diagnostics_ng)
        maintenance = generate_maintenance_tips(diagnostics_ok)
        logger.info("Maintenance analysis completed successfully.")
        return {
            "fixes": fixes,
            "maintenance": maintenance
        }
    except Exception as e:
        logger.error(f"❌ Unexpected error in main maintenance analysis: {e}", exc_info=True)
        return {
            "fixes": "⚠️ Unable to generate repair instructions due to an internal error.",
            "maintenance": "⚠️ Unable to generate maintenance tips due to an internal error."
        }

# --- Main Execution ---
if __name__ == "__main__":
    # --- Dummy Input Data ---
    diagnostics_ng = [
        "❌ front lights: missing/broken",
        "⚠️ plate number: visible but could not be read",
        "❌ wipers: missing/broken"
    ]

    diagnostics_ok = [
        "✅ mirrors: ok",
        "✅ top mirror: ok"
    ]

    print("🚛 Truck Maintenance Advisor (LLM-Powered)\n")

    # Generate recommendations
    recommendations = main(diagnostics_ok, diagnostics_ng)

    # Print Repair Fixes
    print(f"\n🔧 {ORANGE}Recommended Fixes for Issues:{RESET}")
    print("-" * 40)
    print(recommendations["fixes"])
    print("\n")

    # Print Maintenance Tips
    print(f"\n✅ {CYAN}Maintenance Tips for Good Components:{RESET}")
    print("-" * 40)
    print(recommendations["maintenance"])
    print("\n✨ Advice generated successfully.")