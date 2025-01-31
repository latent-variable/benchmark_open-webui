import os
import json
import requests
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM

class OpenWebUIResponse(BaseModel):
    """Schema to enforce JSON output structure for DeepEval."""
    answer: str

class OpenWebUIModel(DeepEvalBaseLLM):
    def __init__(self, api_url="http://localhost:3000/api/chat/completions", 
                 model="llama3.2:latest",
                 extraction_model="llama3.2:latest"):
        """
        Initialize OpenWebUIModel with two models:
        - `model`: The primary model that generates full reasoning + response.
        - `extraction_model`: The secondary model that extracts the final answer.
        """
        self.api_url = api_url
        self.model = model
        self.extraction_model = extraction_model
        self.api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImZjZjYzNDU2LTI2NTYtNGJhOC1iMGU5LWRjNzg0YzBmMDUxMiJ9.C598t2d7Jrb-59SIgzsd2eLJnvKy_IvPqSqIg8ZIa60"  # Load API key from environment variable
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
        """
        Step 1: Primary model generates a full thought process.
        Step 2: Secondary model extracts the actual response.
        Step 3: Return only the final answer.
        """
        print(f"Prompt: {prompt}")

        # Step 1: Generate full thought process
        thought_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # "temperature": 0.7,
            # "max_tokens": 512
        }
        thought_response = self._query_openwebui(thought_payload)
        print(f"Raw Thought Process:\n{thought_response}\n")

        # Step 2: Extract final answer using the secondary model
        extraction_payload = {
            "model": self.extraction_model,
            "messages": [{"role": "user", "content": f"Extract only the final answer from this response only return 'A', 'B', 'C', 'D','No', 'Yes', 'True' or 'False':\n\n{thought_response}"}],
            "temperature": 0.0,
            "max_tokens": 5  # Keep it short to only return "A", "B", "C", or "D"
        }
        final_answer = self._query_openwebui(extraction_payload)

        # Ensure final answer is valid (A, B, C, or D)
        valid_choices = {"A", "B", "C", "D", "True", "False", "No", "Yes"}
        extracted_answer = final_answer.strip().split()[0]  # Extract first valid word
        print('extracted_answer', extracted_answer)
        if extracted_answer not in valid_choices:
            print('****************ERROR**************************')
            extracted_answer = "A"  # Default fallback in case of errors

        

        return schema(answer=extracted_answer)

    async def a_generate(self, prompt: str, schema: BaseModel = OpenWebUIResponse) -> BaseModel:
        """Asynchronous version of generate() for DeepEval compatibility."""
        return self.generate(prompt, schema)

    def get_model_name(self):
        """Returns the model name for logging/debugging."""
        return f"{self.model} + {self.extraction_model}"

    def _query_openwebui(self, payload):
        """Helper function to send requests to OpenWebUI API."""
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error querying OpenWebUI: {e}")
            return "Error"