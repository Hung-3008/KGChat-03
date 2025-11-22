import logging
from backend.query.baseline import RetrievalManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("retrieval_example")

def main():
    logger.info("Initializing RetrievalManager...")
    manager = RetrievalManager(config_path="backend/configs/configs.yml")
    
    questions = [
        "What are the treatments for diabetes?",
        #"What are the symptoms of hypertension?",
        #"How is cancer diagnosed?",
    ]
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {i}/{len(questions)}: {question}")
        logger.info(f"{'='*80}\n")
        
        try:
            answer = manager.run(question)
            
            print(f"\n{'='*80}")
            print(f"QUESTION: {question}")
            print(f"{'='*80}")
            print(f"ANSWER:\n{answer}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Failed to process question '{question}': {e}")
            continue
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
