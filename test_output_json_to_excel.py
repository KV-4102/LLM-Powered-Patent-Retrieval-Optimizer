import json
import pandas as pd

# Path for the input JSON file
input_json_file_path = r"C:\Users\GraphemeLabs\Desktop\Graphemelabs\Codes\test\output.json"

# Path for the output Excel file
output_excel_file_path = r"C:\Users\GraphemeLabs\Desktop\Graphemelabs\Codes\test\output_product.xlsx"

# Read data from the JSON file
with open(input_json_file_path, "r") as json_file:
    data = json.load(json_file)

df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
df.to_excel(output_excel_file_path, index=False)

print(f"Conversion from JSON to Excel is completed. Excel file saved at: {output_excel_file_path}")
