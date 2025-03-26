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
        
    def create_analysis_prompt(self, user_story):
        """Create the prompt messages for user story extraction and INVEST analysis."""
        if not isinstance(user_story, str):
            user_story = json.dumps(user_story)
            
        messages = [
            SystemMessage(content="""You are an expert agile coach specializing in analyzing user stories using the INVEST criteria. 
Your task is twofold:
1. Analyze the original user story and calculate its INVEST score.
2. Create an improved version and provide a detailed refinement summary.

Follow this structured approach:
- Extract the original components (Title, Description, AcceptanceCriteria, AdditionalInformation).
- Score the original story against each INVEST criterion (1-5 scale), considering all provided details accurately.
- Identify specific weaknesses in the original story.
- Create an improved version addressing those weaknesses, including all components.
- Calculate the improved INVEST score.
- Generate a detailed refinement summary comparing the two versions."""),
            HumanMessage(content=f"""
            # User Story: {user_story}

            ## Task Overview

            Perform a complete INVEST analysis on the provided user story with these steps:

            ### Step 1: Analyze the Original User Story
            - Extract or identify all components (Title, Description, AcceptanceCriteria, AdditionalInformation).
            - Score each INVEST criterion (1-5 scale) for the ORIGINAL story AS IS, using all provided details (e.g., evaluate existing acceptance criteria accurately).
            - Calculate the total INVEST score for the original story.
            - Identify specific weaknesses and areas for improvement.

            ### Step 2: Create an Improved Version
            - Generate an improved user story (Title, Description, AcceptanceCriteria, AdditionalInformation) addressing each weakness.
            - Re-score each INVEST criterion for the IMPROVED version.
            - Calculate the new total INVEST score.

            ### Step 3: Generate Analysis Output
            - Include both original and improved user story components.
            - For each INVEST criterion, explain the original score and provide specific recommendations.
            - Ensure explanations reflect the actual content (e.g., don’t claim missing acceptance criteria if they’re present).

            ### Step 4: Create a Dynamic Refinement Summary
            - List specific improvements as bullet points (using '*' on new lines).
            - Include concrete examples of changes between versions.
            - End with "INVEST Score improved from X/30 to Y/30".

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
                "explanation": "string",
                "recommendation": "string"
              }},
              "Negotiable": {{
                "score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Valuable": {{
                "score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Estimable": {{
                "score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Small": {{
                "score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "Testable": {{
                "score": number,
                "explanation": "string",
                "recommendation": "string"
              }},
              "overall": {{
                "score": number,
                "improved_score": number,
                "summary": "string",
                "refinement_summary": "string with '*' bullets on new lines"
              }}
            }}

            IMPORTANT:
            - Return ONLY raw JSON without markdown or backticks.
            - Ensure scores are integers (1-5), overall scores sum correctly (max 30).
            - Use simple '*' bullets on new lines in refinement_summary.
            - Accurately reflect provided acceptance criteria in scoring.
            """)
        ]
        return messages
                
    def analyze_user_story(self, user_story):
        """Extract components and perform INVEST analysis."""
        try:
            chat_model = self.initialize_chat_model()
            analysis_prompt = self.create_analysis_prompt(user_story)
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
                        result[section] = {"score": 0, "improved_score": 0, "summary": "", "refinement_summary": ""}
                    else:
                        result[section] = {"score": 0, "explanation": "", "recommendation": ""}
                        
            # Validate scores
            for criterion in ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]:
                result[criterion]["score"] = max(1, min(5, int(result[criterion]["score"])))
            result["overall"]["score"] = sum(result[c]["score"] for c in ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"])
            result["overall"]["improved_score"] = min(30, int(result["overall"].get("improved_score", 0)))
            
            return result
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "OriginalUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
                "ImprovedUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
                "Independent": {"score": 0, "explanation": "", "recommendation": ""},
                "Negotiable": {"score": 0, "explanation": "", "recommendation": ""},
                "Valuable": {"score": 0, "explanation": "", "recommendation": ""},
                "Estimable": {"score": 0, "explanation": "", "recommendation": ""},
                "Small": {"score": 0, "explanation": "", "recommendation": ""},
                "Testable": {"score": 0, "explanation": "", "recommendation": ""},
                "overall": {"score": 0, "improved_score": 0, "summary": "Error in analysis", "refinement_summary": ""}
            }

def preprocess_input(user_input):
    """Preprocess user input to ensure valid JSON structure."""
    try:
        # If it’s already valid JSON, return it
        return json.loads(user_input)
    except json.JSONDecodeError:
        # Attempt to fix common issues
        cleaned_input = user_input.strip().replace('\n', '').replace('\r', '')
        # Check if it starts with "UserStory" without proper nesting
        if '"UserStory":' in cleaned_input and '"Title":' in cleaned_input:
            # Reconstruct the JSON by wrapping the UserStory content properly
            try:
                # Extract the UserStory part up to AdditionalInformation
                story_match = re.search(r'"UserStory":\s*"Title":\s*"([^"]+)",\s*"Description":\s*"([^"]+)",\s*"AcceptanceCriteria":\s*\[(.*?)\],\s*"AdditionalInformation":\s*"([^"]+)"', cleaned_input)
                if story_match:
                    title, desc, ac, add_info = story_match.groups()
                    ac_list = [item.strip().strip('"') for item in ac.split(',')]
                    fixed_story = {
                        "UserStory": {
                            "Title": title,
                            "Description": desc,
                            "AcceptanceCriteria": ac_list,
                            "AdditionalInformation": add_info
                        }
                    }
                    # Append remaining fields if present
                    for section in ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable", "overall"]:
                        section_match = re.search(rf'"{section}":\s*{{(.*?)}}', cleaned_input)
                        if section_match:
                            section_content = '{' + section_match.group(1) + '}'
                            fixed_story[section] = json.loads(section_content)
                    return fixed_story
            except Exception:
                pass
        # If we can’t fix it, raise an error with guidance
        raise ValueError("Invalid JSON format. Please ensure the input is a properly structured JSON object, e.g., {\"UserStory\": {\"Title\": \"...\", ...}}")

if __name__ == "__main__":
    analyzer = UserStoryInvestAnalyzer()
    user_input = input("Enter user story to analyze: ")
    
    try:
        # Preprocess the input to handle malformed JSON
        user_story = preprocess_input(user_input)
        result = analyzer.analyze_user_story(user_story)
        final_results = json.dumps(result, indent=2)
        print(final_results)
        with open("invest_analysis.json", 'w') as f:
            f.write(final_results)
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"Error processing user story: {str(e)}")