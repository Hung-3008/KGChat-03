import sys
import os
import json

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.schema import ExtractedEdges

try:
    print("Generating schema for ExtractedEdges...")
    schema = ExtractedEdges.model_json_schema()
    print(json.dumps(schema, indent=2))
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
