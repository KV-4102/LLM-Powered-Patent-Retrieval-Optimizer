import os
from transformers import AutoModel, AutoTokenizer
import torch
import json
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
import time
import re

# Load pre-trained model and tokenizer
model_name = "AI-Growth-Lab/PatentSBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

qdrant_url = "http://192.168.10.142:6333/"
collection_name = "test_sbert"

client = QdrantClient(qdrant_url)

# Function to obtain embeddings using CLS pooling
def cls_pooling(model_output, attention_mask):
    return model_output.last_hidden_state[:, 0]
# Path for the output JSON file
output_json_file_path = r"C:\Users\GraphemeLabs\Desktop\Graphemelabs\Codes\test\output.json"

# Read the original JSON file for input data
with open(r"C:\Users\GraphemeLabs\Desktop\Graphemelabs\Dataset\dummyjson.json", "r", encoding="utf-8") as file:
    search_data = json.load(file)

# Initialize list to store output data
output_data = []

# Get the data for project_1
project_data_list = search_data["project_15"]

for project_data in search_data["project_15"]:
    project_id = 15
    patent_numbers = project_data["test_case"]["patent_number_ddb_ddb"]
    num_patent_numbers = len(patent_numbers)

    for criterion_idx, search_criterion in enumerate(project_data["search_criteria"], start=1):
        criterion_data = {
            "project": project_id,
            "model": "sbert",
            "query_no": criterion_idx,
            "Search_Criteria": search_criterion,
            "Patent_no": [f"{patent} - NA" for patent in patent_numbers],
            "MRRFirstdoc": "",
            "Precision": {}, 
            "Recall": {}     
        }
        
        snippet_inputs = tokenizer(
                search_criterion,
                return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to("cuda")
        # Forward pass to obtain embeddings
        with torch.no_grad():
            outputs = model(**snippet_inputs)

        # Perform CLS pooling using the last_hidden_state[:, 0]
        sentence_embeddings = cls_pooling(outputs, snippet_inputs['attention_mask'])

         # Move sentence_embeddings to CPU and convert to numpy array
        embedding_array = sentence_embeddings.squeeze().detach().cpu().numpy()
        # Get the limit from the project data
        limit_list = project_data["limit"]

        # Initialize variables for tracking relevant and retrieved documents
        relevant_documents = set(project_data["test_case"]["patent_number_ddb_ddb"])

        # Initialize variables for tracking retrieved documents for each limit
        retrieved_documents_dict = {limit: set() for limit in limit_list}
        
        # Search for similar embeddings in Qdrant for each limit
        for limit in limit_list:
            search_results = client.search(
                collection_name=collection_name,
                query_vector=embedding_array,
                limit=limit
                        )

            first_position = None
            highest_rank = float('inf')  # Initialize highest rank as infinity
            latest_occurrence_rank = float('-inf')  # Initialize latest occurrence rank as negative infinity

            # Iterate over the search results to find the positions
            for j, result in enumerate(search_results):
                for patent_number_ddb in project_data["test_case"].get("patent_number_ddb_ddb", []):
                    if first_position is None and (patent_number_ddb in result.payload.get("patent_number_ddb_ddb", []) or patent_number_ddb in result.payload.get("patent_number_original_national_office", [])):
                        first_position = j + 1
                    if patent_number_ddb in result.payload.get("patent_number_ddb_ddb", []) or patent_number_ddb in result.payload.get("patent_number_original_national_office", []):
                        position = j + 1  # Get the position of the current matching patent number
                        highest_rank = min(highest_rank, position)  # Update highest_rank with the lowest position found so far
                        latest_occurrence_rank = max(latest_occurrence_rank, position)  # Update latest_occurrence_rank with the highest position found so far

            # Calculate MRR upper and lower bounds
            MRR_upper = 1 / first_position if first_position is not None else 0
            MRR_lower = 1 / latest_occurrence_rank if latest_occurrence_rank != float('-inf') else 0

            for i, patent_number_ddb in enumerate(project_data["test_case"].get("patent_number_ddb_ddb", []), start=1):
                found = False
                for j, result in enumerate(search_results):
                    if patent_number_ddb in result.payload.get("patent_number_ddb_ddb", []) or patent_number_ddb in result.payload.get("patent_number_original_national_office", []):
                        position = j + 1
                        criterion_data["Patent_no"][patent_numbers.index(patent_number_ddb)] = f"{patent_number_ddb} - {position}"
                        print(f"Patent number {patent_number_ddb} for search criterion {criterion_idx} was found at position {position} in top {limit} results.")
                        found = True
                        break
                if not found:
                    criterion_data["Patent_no"][patent_numbers.index(patent_number_ddb)] = f"{patent_number_ddb} - NA"               
                if found:
                    retrieved_documents_dict[limit].add(patent_number_ddb)
            # Similarly, iterate over the other patent number field
            for i, patent_number_original in enumerate(project_data["test_case"].get("patent_number_original_national_office", []), start=1):
                found = False
                # Check if the patent number exists in the search results
                for j, result in enumerate(search_results):
                    if patent_number_original in result.payload.get("patent_number_ddb_ddb", []) or patent_number_original in result.payload.get("patent_number_original_national_office", []):
                        position = j + 1
                        criterion_data["Patent_no"][patent_numbers.index(patent_number_ddb)] = f"{patent_number_ddb} - {position}"
                        print(f"Patent number {patent_number_ddb} for search criterion {criterion_idx} was found at position {position} in top {limit} results.")
                        found = True
                        break
                
                if not found:
                    criterion_data["Patent_no"][patent_numbers.index(patent_number_ddb)] = f"{patent_number_ddb} - {position}"
                    print(f"Patent number {patent_number_ddb} for search criterion {criterion_idx} was found at position {position} in top {limit} results.")
                if found:
                    # Increment R_found if patent number is found in search results
                    retrieved_documents_dict[limit].add(patent_number_original)
            # Calculate Precision and Recall
            R_found = len(retrieved_documents_dict[limit])

            # Calculate Precision and Recall
            precision = R_found / limit if limit > 0 else 0
            recall = R_found / len(project_data["test_case"]["patent_number_ddb_ddb"]) if len(relevant_documents) > 0 else 0
            
            print(f"Results for Limit {limit} and search criterion {criterion_idx}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print(f"Mean Reciprocal Rank upper:" ,round(MRR_upper,8))
            # Update MRR, Precision, and Recall fields in criterion data
            criterion_data[f"MRRFirstdoc"] = f"{MRR_upper:.8f}" if MRR_upper > 0 else ""

            # Store precision and recall at the specified limit
            criterion_data["Precision"][limit] = f"{precision:.2%}"
            criterion_data["Recall"][limit] = f"{recall:.2%}"

        # Add criterion data to output data
        output_data.append(criterion_data)

# Read the existing data from the output JSON file if it exists
existing_output_data = []
if os.path.exists(output_json_file_path):
    with open(output_json_file_path, "r") as existing_output_file:
        existing_output_data = json.load(existing_output_file)

# Append new data to existing data
existing_output_data.extend(output_data)

# Write updated data (existing + new) back to the JSON file
with open(output_json_file_path, "w") as output_file:
    json.dump(existing_output_data, output_file, indent=4)
