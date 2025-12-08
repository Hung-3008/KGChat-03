---
description: Optimize Qdrant collection and run graph creation
---

Step 1: Qdrant Optimization (CRITICAL)
This step merges the 23 file segments into 2, which is required to prevent search timeouts on the 15M vector dataset.
**NOTE**: This process may take 10-20 minutes. Please **do not interrupt** it until it says "✅ Optimization Complete!".

```bash
# Run optimization script
python backend/utils/force_optimize_qdrant.py
```

Step 2: Run Graph Creation
Once Step 1 is complete (Status: green, Segments: <= 2), you can run the graph creation script.

```bash
# Run create_graph.py
python create_graph.py
```
