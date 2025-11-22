KEYWORD_EXTRACTION_PROMPT = """Extract useful keywords from the following question for information retrieval. Return only a list of keywords.
Question: {question}"""

ANSWER_GENERATION_PROMPT = """Answer the question based on the provided context.

Question: {question}

Graph Context:
{graph_context}

Google Grounding Context:
{google_context}

Answer:
"""
