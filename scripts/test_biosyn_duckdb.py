import sys
import os
import duckdb
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend/BioSyn to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
biosyn_path = os.path.join(project_root, 'backend', 'BioSyn')
if biosyn_path not in sys.path:
    sys.path.append(biosyn_path)

try:
    from inference import BioSynInference
except ImportError as e:
    logger.error(f"Failed to import BioSynInference: {e}")
    sys.exit(1)

def main():
    mention = "myocardial infarction"
    logger.info(f"Running BioSyn inference for mention: '{mention}'")

    # Initialize BioSynInference
    try:
        inference_engine = BioSynInference()
        result = inference_engine.predict(mention)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)
        
    logger.info("Inference result obtained.")
    
    if 'predictions' not in result or not result['predictions']:
        logger.warning("No predictions found.")
        return

    top_prediction = result['predictions'][0]
    cui = top_prediction['id']
    name = top_prediction['name']
    
    logger.info(f"Top prediction: Name='{name}', CUI='{cui}'")
    
    # Query DuckDB
    db_path = os.path.join(project_root, 'data', 'umls.duckdb')
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return

    logger.info(f"Connecting to DuckDB at {db_path}...")
    try:
        con = duckdb.connect(db_path)
        
        ids_to_check = [cui]
        if '|' in cui:
            ids_to_check.extend(cui.split('|'))
            
        found_any = False
        for id_val in ids_to_check:
            # Check CUI column
            query_cui = "SELECT STR FROM mrconso WHERE CUI = ? LIMIT 5"
            logger.info(f"Checking CUI='{id_val}'...")
            results_cui = con.execute(query_cui, [id_val]).fetchall()
            if results_cui:
                logger.info(f" -> Found match in CUI column for '{id_val}': {results_cui[0][0]}")
                found_any = True
                continue
                
            # Check CODE column
            query_code = "SELECT STR FROM mrconso WHERE CODE = ? LIMIT 5"
            logger.info(f"Checking CODE='{id_val}'...")
            results_code = con.execute(query_code, [id_val]).fetchall()
            if results_code:
                logger.info(f" -> Found match in CODE column for '{id_val}': {results_code[0][0]}")
                found_any = True
                continue
                
        if not found_any:
            logger.warning(f"No matches found in mrconso for ID {cui} (checked CUI and CODE columns)")
            
        con.close()
        
    except Exception as e:
        logger.error(f"Database query failed: {e}")

if __name__ == "__main__":
    main()
