import os
import requests
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
class OpenWebUIResponse(BaseModel):
    """Schema to enforce JSON output structure for DeepEval."""
    answer: str

class OpenWebUIModel(DeepEvalBaseLLM):
    def __init__(self, api_url="http://localhost:3000/api/chat/completions", 
                 model="deepseek_r1_reasoner.Reasoning_Effort_4/deepseek-r1:8b", #"deepseek_r1_reasoner.Reasoning_Effort_4/deepseek-r1:8b",
                 extraction_model="llama3.1:latest", enable_cot=True):
        """
        Initialize OpenWebUIModel with two models:
        - `model`: The primary model that generates full reasoning + response.
        - `extraction_model`: The secondary model that extracts the final answer.
        """
        self.api_url = api_url
        self.model = model
        self.extraction_model = extraction_model
        self.enable_cot = enable_cot
        self.api_key = os.getenv("OPENWEBUI_API_KEY") # Load API key from environment variable
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.load_model()

    def load_model(self):
        """Required method for DeepEvalBaseLLM."""
        print(f"Using primary model: {self.model} and extraction model: {self.extraction_model}")
        return self.model
    

    def generate(self, prompt: str, schema: BaseModel = OpenWebUIResponse) -> BaseModel:
        if self.enable_cot: # Direct prompting 
            return self._generate_CoT_promting( prompt, schema)
        else:
            return self._generate_direct_promting( prompt, schema)

    def _generate_CoT_promting(self, prompt: str, schema: BaseModel = OpenWebUIResponse) -> BaseModel:
        """
        Step 1: Primary model generates a full thought process.
        Step 2: Secondary model extracts the actual response.
        Step 3: Return only the final answer, ensuring it adheres to the schema.
        """
        print("".join(['*'] * 100))
        # print(f"schema: {schema.schema_json(indent=2)}")
        

        # Extract schema properties dynamically
        schema_dict = schema.model_json_schema()
        answer_properties = schema_dict["properties"]["answer"]
        answer_type = answer_properties["type"]  # Can be "string", "integer", etc.
        valid_choices = answer_properties.get("enum", None)  # Only present for multiple-choice

        if valid_choices:
            prompt = f"Ensure the output includes one of the following: {', '.join(valid_choices)}.\n" + prompt
            
        print(prompt)

        # Step 1: Generate full thought process
        thought_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
 
        thought_response = self._query_openwebui(thought_payload)
        # print(f"Raw Thought Process:\n{thought_response}\n")
        model_response = self._extract_model_answer_from_thought_response(thought_response)
        print( "model_response:", model_response)

        # Step 2: Construct extraction prompt based on type
        if valid_choices:
            # Multiple-choice case
            extraction_prompt = (
                # f"Prompt {prompt}"
                f"Extract only the final answer from this response to the Prompt."
                f"Ensure the output is exactly one of the following: {', '.join(valid_choices)}. "
                f"Do not add any extra text:\n\n{model_response}"
            )
        elif answer_type == "integer":
            # Numerical answer case
            extraction_prompt = (
                # f"Prompt {prompt}"
                "Extract only the final numerical answer from this response to the Prompt "
                "Return it as a single integer value without any extra text:\n\n"
                f"{model_response}"
            )
        else:
            # Fallback for unknown types
            extraction_prompt = (
                # f"Prompt {prompt}"
                "Extract only the final answer from this response to the Prompt "
                "Ensure it matches the expected format:\n\n"
                f"Examples and question: {prompt}"
                f"Reponse: {model_response}"
            )

        # Step 3: Query the model for extraction
        extraction_payload = {
            "model": self.extraction_model,
            "messages": [{"role": "user", "content": extraction_prompt}],
            "temperature": 0.0,
            "max_tokens": 100  # Allow room for numbers or short responses
        }
        final_answer = self._query_openwebui(extraction_payload).strip()

        # Step 4: Validate extracted output based on expected type
        if valid_choices and final_answer not in valid_choices:
            print(f"⚠️ Invalid extracted answer: {final_answer}. Defaulting to {next(iter(valid_choices))}.")
            final_answer = next(iter(valid_choices))  # Default to the first valid choice
        elif answer_type == "integer":
            try:
                final_answer = int(final_answer)  # Ensure it’s a valid integer
            except ValueError:
                print(f"⚠️ Invalid integer extracted: {final_answer}. Defaulting to 0.")
                final_answer = 0  # Default fallback
        
        print("final_answer", final_answer)
        return schema(answer=final_answer)
    



    def _extract_model_answer_from_thought_response(self, thought_response: str) -> str:
        """
        Extracts the model's final answer from the thought response.
        
        It looks for the marker '</think>' and returns only the text that follows it.
        If the marker is not found, it returns an empty string.
        """
        marker = "</think>"
        
        # Find the position of the marker
        marker_index = thought_response.find(marker)

        if marker_index == -1:
            print("⚠️ Marker '</think>' not found in response. Returning original CoT.")
            return thought_response

        # Extract everything after the marker
        model_answer = thought_response[marker_index + len(marker):].strip()
        return model_answer

    def _generate_direct_promting(self, prompt: str, schema: BaseModel = OpenWebUIResponse) -> BaseModel:
        """
        Direct prompting: Ensure the model returns only the final response with no additional text.
        """
        print("".join(['*'] * 100))
        print(prompt)

        # Extract schema properties dynamically
        schema_dict = schema.model_json_schema()
        answer_properties = schema_dict["properties"]["answer"]
        answer_type = answer_properties["type"]  # Can be "string", "integer", etc.
        valid_choices = answer_properties.get("enum", None)  # Only present for multiple-choice


        # Step 1: Construct strict direct prompt
        if valid_choices:
            final_prompt = (
                f"Respond with only the correct answer. "
                f"Your response must be exactly one of the following: {', '.join(valid_choices)}. "
                f"Do not add any explanations, formatting, or extra text.\n\nQuestion: {prompt}"
            )
        elif answer_type == "integer":
            final_prompt = (
                "Provide only the final numerical answer. "
                "Return it as a single integer with no extra text, formatting, or words.\n\n"
                f"Question: {prompt}"
            )
        else:
            final_prompt = (
                "Provide only the final answer in the correct format. "
                "Do not include any explanations, extra text, or formatting.\n\n"
                f"Question: {prompt}"
            )

        # Step 2: Query the model
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": 0.0,
            "max_tokens": 100  # Ensure response is concise
        }
        final_answer = self._query_openwebui(payload).strip()

        # Step 3: Validate extracted output
        if valid_choices and final_answer not in valid_choices:
            print(f"⚠️ Invalid extracted answer: {final_answer}. Defaulting to {next(iter(valid_choices))}.")
            final_answer = next(iter(valid_choices))  # Default to the first valid choice
        elif answer_type == "integer":
            try:
                final_answer = int(final_answer)  # Ensure it’s a valid integer
            except ValueError:
                print(f"⚠️ Invalid integer extracted: {final_answer}. Defaulting to 0.")
                final_answer = 0  # Default fallback

        print("final_answer", final_answer)
        return schema(answer=final_answer)

    async def a_generate(self, prompt: str, schema: BaseModel = OpenWebUIResponse) -> BaseModel:
        """Asynchronous version of generate() for DeepEval compatibility."""
        return self.generate(prompt, schema)

    def get_model_name(self):
        """Returns the model name for logging/debugging."""
        return f"{self.model}"

    def _query_openwebui(self, payload):
        """Helper function to send requests to OpenWebUI API."""
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error querying OpenWebUI: {e}")
            return "Error"