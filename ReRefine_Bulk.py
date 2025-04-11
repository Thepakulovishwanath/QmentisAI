from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.schema import SystemMessage, HumanMessage
import json
import re
import glob

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
            user_story = json.dumps(user_story, ensure_ascii=False)
        else:
            try:
                json.loads(user_story)
            except json.JSONDecodeError:
                raise ValueError("user_story string is not valid JSON")
        
        user_story_escaped = json.dumps(user_story)[1:-1]  # Escape for prompt
        
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
            - If the aspects to enhance or additional context indicate "No specific aspects provided." or "No additional context provided.", perform a general refinement based on INVEST criteria, focusing on clarity, testability, and estimability.
            - Re-score each INVEST criterion for the improved user story (1-5 scale).
            - Calculate the improved INVEST score by summing the improved scores.
            - Generate a detailed refinement summary comparing the two versions.

            Return ONLY raw JSON without markdown or backticks."""),
            HumanMessage(content=f"""
            # User Story: {user_story_escaped}

            ## Refinement Guidance

            ### Aspects of the user story to enhance:
            {aspects_to_enhance}

            ### Additional information or context to consider:
            {additional_context}

            ## Task Overview

            Perform a complete INVEST analysis on the provided user story with these steps:

            ### Step 1: Analyze the Original User Story
            - Extract all components (Title, Description, AcceptanceCriteria, AdditionalInformation).
            - Score each INVEST criterion (1-5 scale) for the ORIGINAL story AS IS.
            - Use the provided input score of {input_score}/30 as the baseline for comparison.

            ### Step 2: Create an Improved Version
            - Generate an improved user story addressing each weakness.
            - Consider the aspects to enhance and additional context to guide the refinement.
            - Re-score each INVEST criterion for the IMPROVED version (1-5 scale).
            - Calculate the new total INVEST score.

            ### Step 3: Generate Analysis Output
            - Include both original and improved user story components.
            - For each INVEST criterion, provide the original score, improved score, explanation, and recommendation.
            - Ensure explanations reflect the actual content.

            ### Step 4: Create a Refinement Summary
            - List improvements as bullet points (using '*' on new lines).
            - Include examples of changes.
            - End with "INVEST Score improved from {input_score}/30 to Y/30", where Y is the total improved score.

            ## Response Format:
            {{
              "OriginalUserStory": {{"Title": "string", "Description": "string", "AcceptanceCriteria": ["string", ...], "AdditionalInformation": "string"}},
              "ImprovedUserStory": {{"Title": "string", "Description": "string", "AcceptanceCriteria": ["string", ...], "AdditionalInformation": "string"}},
              "Independent": {{"score": number, "improved_score": number, "explanation": "string", "recommendation": "string"}},
              "Negotiable": {{"score": number, "improved_score": number, "explanation": "string", "recommendation": "string"}},
              "Valuable": {{"score": number, "improved_score": number, "explanation": "string", "recommendation": "string"}},
              "Estimable": {{"score": number, "improved_score": number, "explanation": "string", "recommendation": "string"}},
              "Small": {{"score": number, "improved_score": number, "explanation": "string", "recommendation": "string"}},
              "Testable": {{"score": number, "improved_score": number, "explanation": "string", "recommendation": "string"}},
              "overall": {{"input_score": number, "improved_score": number, "summary": "string", "refinement_summary": "string with '*' bullets"}}
            }}
            """)
        ]
        return messages
                
    def analyze_user_story(self, user_story, aspects_to_enhance="", additional_context="", input_score=0):
        """Extract components and perform INVEST analysis with refinement guidance."""
        try:
            chat_model = self.initialize_chat_model()
            analysis_prompt = self.create_analysis_prompt(user_story, aspects_to_enhance, additional_context, input_score)
            response = chat_model.invoke(analysis_prompt)
            
            if not isinstance(response.content, str):
                raise ValueError(f"LLM response.content is not a string: {type(response.content)}")
            content = response.content.strip()
            if not content:
                raise ValueError("LLM returned empty content")
            
            json_content = sanitize_json_string(content)
            try:
                result = json.loads(json_content)
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM returned invalid JSON: {json_content[:100]}... Error: {str(e)}")
            
            # Filter result to match desired structure
            filtered_result = {
                "OriginalUserStory": result.get("OriginalUserStory", {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""}),
                "ImprovedUserStory": result.get("ImprovedUserStory", {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""}),
                "Independent": result.get("Independent", {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""}),
                "Negotiable": result.get("Negotiable", {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""}),
                "Valuable": result.get("Valuable", {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""}),
                "Estimable": result.get("Estimable", {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""}),
                "Small": result.get("Small", {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""}),
                "Testable": result.get("Testable", {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""}),
                "overall": result.get("overall", {"input_score": input_score, "improved_score": 0, "summary": "", "refinement_summary": ""})
            }
            
            # Validate scores
            for criterion in ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]:
                filtered_result[criterion]["score"] = max(1, min(5, int(filtered_result[criterion].get("score", 0))))
                filtered_result[criterion]["improved_score"] = max(1, min(5, int(filtered_result[criterion].get("improved_score", 0))))
            
            # Calculate improved score
            calculated_improved_score = sum(filtered_result[c]["improved_score"] for c in ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"])
            filtered_result["overall"]["input_score"] = max(0, min(30, int(input_score)))
            filtered_result["overall"]["improved_score"] = calculated_improved_score
            if filtered_result["overall"]["refinement_summary"]:
                filtered_result["overall"]["refinement_summary"] = re.sub(
                    r"INVEST Score improved from \d+/30 to \d+/30",
                    f"INVEST Score improved from {input_score}/30 to {calculated_improved_score}/30",
                    filtered_result["overall"]["refinement_summary"]
                )
            
            return filtered_result
            
        except Exception as e:
            return {
                "OriginalUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
                "ImprovedUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
                "Independent": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Negotiable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Valuable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Estimable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Small": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "Testable": {"score": 0, "improved_score": 0, "explanation": "", "recommendation": ""},
                "overall": {"input_score": input_score, "improved_score": 0, "summary": f"Error in analysis: {str(e)}", "refinement_summary": ""}
            }

def preprocess_input(user_input):
    """Preprocess user input to ensure valid JSON structure and extract refinement guidance."""
    try:
        data = json.loads(user_input)
        if "input" not in data:
            raise ValueError("Input JSON must contain an 'input' field with Title, Description, AcceptanceCriteria, and AdditionalInformation.")
        
        user_story = data["input"]
        required_fields = ["title", "description", "acceptance_criteria", "additional_information"]
        for field in required_fields:
            if field not in user_story:
                raise ValueError(f"input must contain the field: {field}")

        aspects_to_enhance = data.get("aspects_to_enhance", "No specific aspects provided.")
        if not isinstance(aspects_to_enhance, str):
            raise ValueError("'aspects_to_enhance' must be a string.")
        
        additional_context = data.get("additional_context", "No additional context provided.")
        if not isinstance(additional_context, str):
            raise ValueError("'additional_context' must be a string.")

        input_score = data.get("evaluation", {}).get("overall", {}).get("score", 0)
        if not isinstance(input_score, (int, float)):
            input_score = 0
        input_score = max(0, min(30, int(input_score)))

        return user_story, aspects_to_enhance, additional_context, input_score

    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format. Please ensure the input is a properly structured JSON object.")
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

def process_input_folder(input_folder, output_folder):
    """Process all JSON files in the input folder and save results in the output folder."""
    # Use the script's directory as the base path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_input_path = os.path.join(script_dir, input_folder)
    full_output_path = os.path.join(script_dir, output_folder)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for JSON files in: {full_input_path}")
    
    # Check if input folder exists
    if not os.path.exists(full_input_path):
        print(f"Input folder '{input_folder}' does not exist at {full_input_path}. Please create it and add your JSON files (e.g., 'user_story_1.json').")
        # Suggest existing folders
        existing_folders = [d for d in os.listdir(script_dir) if os.path.isdir(os.path.join(script_dir, d))]
        if existing_folders:
            print(f"Existing folders in {script_dir}: {', '.join(existing_folders)}")
        return
    
    # Ensure output folder exists
    os.makedirs(full_output_path, exist_ok=True)
    
    # Get all JSON files in the input folder
    json_files = glob.glob(os.path.join(full_input_path, "*.json"))
    print(f"Found files: {json_files}")
    if not json_files:
        print(f"No JSON files found in '{full_input_path}'. Please add files like 'user_story_1.json', 'user_story_2.json', etc.")
        return
    
    analyzer = UserStoryInvestAnalyzer()
    
    for input_file in json_files:
        print(f"Processing {input_file}...")
        try:
            # Read the JSON file
            with open(input_file, 'r', encoding='utf-8') as f:
                input_json = f.read()
            
            # Preprocess and analyze
            user_story, aspects, context, score = preprocess_input(input_json)
            result = analyzer.analyze_user_story(user_story, aspects, context, score)
            
            # Prepare output file path
            filename = os.path.basename(input_file)
            output_file = os.path.join(full_output_path, filename)
            
            # Save filtered result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Saved result to {output_file}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    # Define input and output folders relative to script location
    input_folder = "user_story_evaluations"
    output_folder = "output"
    
    # Process all files in the input folder
    process_input_folder(input_folder, output_folder)