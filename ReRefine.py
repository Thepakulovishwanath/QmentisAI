from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.schema import SystemMessage, HumanMessage
import json
import re

load_dotenv()

def sanitize_json_string(json_str):
    """Sanitize a JSON string by removing or replacing control characters."""
    json_str = re.sub(r'[\x00-\x09\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', json_str)
    def clean_string_value(match):
        value = match.group(1)
        value = value.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        return f'"{value}"'
    json_str = re.sub(r'"((?:\\.|[^"\\])*)"', clean_string_value, json_str)
    return json_str

class UserStoryInvestAnalyzer:
    def __init__(self, chat_model=None):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL")
        self.chat_model = chat_model
        if not self.api_key or not self.model:
            raise ValueError("GROQ_API_KEY or GROQ_MODEL environment variables not set")
        
    def initialize_chat_model(self):
        """Initialize and return the Groq chat model."""
        if not self.chat_model:
            self.chat_model = ChatGroq(model=self.model, api_key=self.api_key)
        return self.chat_model
        
    def create_analysis_prompt(self, user_story, aspects_to_enhance, additional_context, input_score):
        """Create the prompt messages for user story extraction and INVEST analysis."""
        if not isinstance(user_story, str):
            user_story = json.dumps(user_story)
            
        messages = [
            SystemMessage(content="""You are an expert agile coach specializing in analyzing user stories using the INVEST criteria. 
Your task is twofold:
1. Analyze the original user story and calculate its INVEST score.
2. Create an improved version and provide a detailed refinement summary, considering the provided refinement guidance.

Follow this structured approach:
- Extract the original components (Title, Description, AcceptanceCriteria, AdditionalInformation).
- Score the original story against each INVEST criterion (1-5 scale), considering all provided details accurately.
- Identify specific weaknesses in the original story.
- Create an improved version addressing those weaknesses, incorporating the aspects to enhance and additional context provided.
- If the aspects to enhance or additional context indicate "No specific aspects provided." or "No additional context provided.", perform a general refinement of the user story based on the INVEST criteria, focusing on common areas of improvement such as clarity, testability, and estimability.
- Re-score each INVEST criterion for the improved user story (1-5 scale).
- Calculate the improved INVEST score by summing the improved scores.
- Generate a detailed refinement summary comparing the two versions."""),
            HumanMessage(content=f"""
            # User Story: {user_story}

            ## Refinement Guidance

            ### Aspects of the user story to enhance:
            {aspects_to_enhance}

            ### Additional information or context to consider:
            {additional_context}

            ## Task Overview

            Perform a complete INVEST analysis on the provided user story with these steps:

            ### Step 1: Analyze the Original User Story
            - Extract or identify all components (Title, Description, AcceptanceCriteria, AdditionalInformation).
            - Score each INVEST criterion (1-5 scale) for the ORIGINAL story AS IS, using all provided details (e.g., evaluate existing acceptance criteria accurately).
            - The user has provided an input score of {input_score}/30 for the original story. Use this as the baseline for comparison, but still provide your own scoring for each criterion based on your analysis.

            ### Step 2: Create an Improved Version
            - Generate an improved user story (Title, Description, AcceptanceCriteria, AdditionalInformation) addressing each weakness.
            - Consider the aspects to enhance and additional context provided to guide the refinement.
            - Re-score each INVEST criterion for the IMPROVED version (1-5 scale).
            - Calculate the new total INVEST score for the improved version by summing the improved scores.

            ### Step 3: Generate Analysis Output
            - Include both original and improved user story components.
            - For each INVEST criterion, provide the original score, the improved score, an explanation of the original score, and specific recommendations for improvement.
            - Ensure explanations reflect the actual content (e.g., don’t claim missing acceptance criteria if they’re present).

            ### Step 4: Create a Dynamic Refinement Summary
            - List specific improvements as bullet points (using '*' on new lines).
            - Include concrete examples of changes between versions.
            - End with "INVEST Score improved from {input_score}/30 to Y/30", where Y is the total improved score.

            ## Response Format:

            Return a structured JSON:

            {{
              "OriginalUserStory": {{
                "Title": "string",
                "Description": "string",
                "AcceptanceCriteria": ["string", ...],
                "AdditionalInformation": "string"
              }},
              "ImprovedUserStory": {{
                "Title": "string",
                "Description": "string",
                "AcceptanceCriteria": ["string", ...],
                "AdditionalInformation": "string"
              }},
              "Independent": {{
                "score": number,
                "improved_score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Negotiable": {{
                "score": number,
                "improved_score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Valuable": {{
                "score": number,
                "improved_score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Estimable": {{
                "score": number,
                "improved_score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Small": {{
                "score": number,
                "improved_score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Testable": {{
                "score": number,
                "improved_score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "overall": {{
                "input_score": number,
                "improved_score": number,
                "summary": "string",
                "refinement_summary": "string with '*' bullets on new lines"
              }}
            }}

            IMPORTANT:
            - Return ONLY raw JSON without markdown or backticks.
            - Ensure scores are integers (1-5), overall scores sum correctly (max 30).
            - Use the provided input_score ({input_score}/30) as the overall.input_score in the response.
            - Use simple '*' bullets on new lines in refinement_summary.
            - Accurately reflect provided acceptance criteria in scoring.
            """)
        ]
        return messages
                
    def analyze_user_story(self, user_story, aspects_to_enhance="", additional_context="", input_score=0):
        """Extract components and perform INVEST analysis with refinement guidance."""
        try:
            chat_model = self.initialize_chat_model()
            analysis_prompt = self.create_analysis_prompt(user_story, aspects_to_enhance, additional_context, input_score)
            response = chat_model.invoke(analysis_prompt)
            content = response.content.strip()
            
            json_content = sanitize_json_string(content)
            result = json.loads(json_content)
            
            # Validate required fields
            required_sections = ["OriginalUserStory", "ImprovedUserStory", "Independent", "Negotiable", 
                                "Valuable", "Estimable", "Small", "Testable", "overall"]
            for section in required_sections:
                if section not in result:
                    if section in ["OriginalUserStory", "ImprovedUserStory"]:
                        result[section] = {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""}
                    elif section == "overall":
                        result[section] = {"input_score": input_score, "improved_score": 0, "summary": "", "refinement_summary": ""}
                    else:
                        result[section] = {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""}
            
            # Validate scores for each criterion (original and improved)
            for criterion in ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]:
                # Validate original score
                if "score" not in result[criterion] or not isinstance(result[criterion]["score"], (int, float)):
                    result[criterion]["score"] = 0
                else:
                    try:
                        score = int(result[criterion]["score"])
                        result[criterion]["score"] = max(1, min(5, score))
                    except (ValueError, TypeError):
                        result[criterion]["score"] = 0
                
                # Validate improved score
                if "improved_score" not in result[criterion] or not isinstance(result[criterion]["improved_score"], (int, float)):
                    result[criterion]["improved_score"] = 0
                else:
                    try:
                        improved_score = int(result[criterion]["improved_score"])
                        result[criterion]["improved_score"] = max(1, min(5, improved_score))
                    except (ValueError, TypeError):
                        result[criterion]["improved_score"] = 0
            
            # Calculate the total improved score
            calculated_improved_score = sum(result[c]["improved_score"] for c in ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"])
            
            # Validate overall section
            if "input_score" not in result["overall"] or not isinstance(result["overall"]["input_score"], (int, float)):
                result["overall"]["input_score"] = input_score
            result["overall"]["input_score"] = max(0, min(30, int(result["overall"]["input_score"])))
            
            if "improved_score" not in result["overall"] or not isinstance(result["overall"]["improved_score"], (int, float)):
                result["overall"]["improved_score"] = calculated_improved_score
            result["overall"]["improved_score"] = max(0, min(30, int(result["overall"]["improved_score"])))
            
            # If the AI's improved_score doesn't match the calculated sum, update it
            if result["overall"]["improved_score"] != calculated_improved_score:
                result["overall"]["improved_score"] = calculated_improved_score
                # Update the refinement_summary to reflect the correct improved score
                current_refinement_summary = result["overall"].get("refinement_summary", "")
                score_pattern = r"INVEST Score improved from \d+/30 to \d+/30"
                updated_refinement_summary = re.sub(score_pattern, f"INVEST Score improved from {result['overall']['input_score']}/30 to {calculated_improved_score}/30", current_refinement_summary)
                result["overall"]["refinement_summary"] = updated_refinement_summary if updated_refinement_summary else f"INVEST Score improved from {result['overall']['input_score']}/30 to {calculated_improved_score}/30"
            
            return result
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "OriginalUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
                "ImprovedUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
                "Independent": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Negotiable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Valuable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Estimable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Small": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Testable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "overall": {"input_score": input_score, "improved_score": 0, "summary": "Error in analysis", "refinement_summary": ""}
            }

def preprocess_input(user_input):
    """Preprocess user input to ensure valid JSON structure and extract refinement guidance."""
    try:
        # Parse the input JSON
        data = json.loads(user_input)
        
        # Extract the UserStory portion
        if "UserStory" not in data:
            raise ValueError("Input JSON must contain a 'UserStory' field with Title, Description, AcceptanceCriteria, and AdditionalInformation.")
        
        user_story = data["UserStory"]
        required_fields = ["Title", "Description", "AcceptanceCriteria", "AdditionalInformation"]
        for field in required_fields:
            if field not in user_story:
                raise ValueError(f"UserStory must contain the field: {field}")
        
        # Extract aspects_to_enhance (expected to be a string)
        aspects_to_enhance = data.get("aspects_to_enhance", "")
        if not isinstance(aspects_to_enhance, str):
            raise ValueError("'aspects_to_enhance' must be a string.")
        aspects_to_enhance_str = aspects_to_enhance if aspects_to_enhance else "No specific aspects provided."

        # Extract additional_context (expected to be a string)
        additional_context = data.get("additional_context", "")
        if not isinstance(additional_context, str):
            raise ValueError("'additional_context' must be a string.")
        additional_context_str = additional_context if additional_context else "No additional context provided."

        # Extract the user's input score
        input_score = data.get("overall", {}).get("score", 0)
        if not isinstance(input_score, (int, float)):
            input_score = 0
        input_score = max(0, min(30, int(input_score)))

        return user_story, aspects_to_enhance_str, additional_context_str, input_score

    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format. Please ensure the input is a properly structured JSON object.")
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

if __name__ == "__main__":
    analyzer = UserStoryInvestAnalyzer()
    user_input = input("Enter the complete user story JSON (including aspects_to_enhance and additional_context): ")
    
    try:
        # Preprocess the input to extract user story, refinement guidance, and input score
        user_story, aspects_to_enhance, additional_context, input_score = preprocess_input(user_input)
        result = analyzer.analyze_user_story(user_story, aspects_to_enhance, additional_context, input_score)
        final_results = json.dumps(result, indent=2)
        print(final_results)
        with open("invest_analysis.json", 'w') as f:
            f.write(final_results)
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"Error processing user story: {str(e)}")