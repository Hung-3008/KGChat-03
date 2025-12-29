#!/usr/bin/env python3
"""
Test Qdrant concurrent query capacity after optimization

This script tests whether Qdrant can handle high concurrent load
after applying the performance tuning configuration.
"""

import sys
import time
import numpy as np
from pathlib import Path  
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

def test_concurrent_queries(num_queries=100, max_workers=50):
    """
    Test Qdrant with concurrent queries
    
    Args:
        num_queries: Total number of queries to execute
        max_workers: Maximum concurrent workers
    """
    print(f"\n{'='*60}")
    print(f"Testing Qdrant Concurrent Query Capacity")
    print(f"{'='*60}\n")
    
    # Connect to Qdrant with gRPC
    client = QdrantClient(
        url="http://localhost:6333",
        grpc_port=6334,
        prefer_grpc=True,
        timeout=300
    )
    
    print(f"✓ Connected to Qdrant with gRPC")
    
    # Get collection info
    try:
        collection_info = client.get_collection("kg_lv2_nodes")
        print(f"✓ Collection 'kg_lv2_nodes': {collection_info.points_count} points")
    except Exception as e:
        print(f"✗ Failed to get collection info: {e}")
        return
    
    def single_query(query_id):
        """Execute a single search query"""
        start_time = time.time()
        
        try:
            # Generate random vector (768 dimensions for biobert)
            query_vector = np.random.rand(768).tolist()
            
            # Execute search with grouping (like entity linking)
            result = client.query_points_groups(
                collection_name="kg_lv2_nodes",
                query=query_vector,
                group_by="cui",
                limit=5,
                group_size=1,
                with_payload=["cui", "name"],
                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=False
                )
            )
            
            elapsed = time.time() - start_time
            return {
                'id': query_id,
                'status': 'OK',
                'elapsed': elapsed,
                'results': len(result.groups) if result else 0,
                'error': None
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                'id': query_id,
                'status': 'FAIL',
                'elapsed': elapsed,
                'results': 0,
                'error': str(e)
            }
    
    # Run concurrent queries
    print(f"\nRunning {num_queries} queries with {max_workers} concurrent workers...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_query, i) for i in range(num_queries)]
        results = [f.result() for f in futures]
    
    total_time = time.time() - start_time
    
    # Analyze results
    success = [r for r in results if r['status'] == 'OK']
    failed = [r for r in results if r['status'] == 'FAIL']
    
    if success:
        avg_latency = np.mean([r['elapsed'] for r in success])
        median_latency = np.median([r['elapsed'] for r in success])
        p95_latency = np.percentile([r['elapsed'] for r in success], 95)
        max_latency = max([r['elapsed'] for r in success])
    else:
        avg_latency = median_latency =p95_latency = max_latency = 0
    
    # Print results
    print(f"Results:")
    print(f"{'='*60}")
    print(f"✓ Successful queries: {len(success)}/{num_queries} ({len(success)/num_queries*100:.1f}%)")
    print(f"✗ Failed queries: {len(failed)}/{num_queries} ({len(failed)/num_queries*100:.1f}%)")
    print(f"\nLatency Statistics (successful queries):")
    print(f"  Average: {avg_latency*1000:.1f}ms")
    print(f"  Median:  {median_latency*1000:.1f}ms")
    print(f"  P95:     {p95_latency*1000:.1f}ms")
    print(f"  Max:     {max_latency*1000:.1f}ms")
    print(f"\nThroughput:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  QPS: {num_queries/total_time:.1f} queries/second")
    
    # Error analysis
    if failed:
        print(f"\nError Summary:")
        error_types = {}
        for f in failed:
            error_msg = str(f['error'])[:50]  # First 50 chars
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  [{count}x] {error}")
    
    # Pass/Fail criteria
    success_rate = len(success) / num_queries
    print(f"\n{'='*60}")
    if success_rate >= 0.95:  # 95% success rate
        print(f"✅ TEST PASSED (Success rate: {success_rate*100:.1f}%)")
        print(f"   Qdrant can handle {max_workers} concurrent queries!")
    elif success_rate >= 0.80:
        print(f"⚠️  TEST PARTIAL (Success rate: {success_rate*100:.1f}%)")
        print(f"   Some errors occurred, but acceptable for high load")
    else:
        print(f"❌ TEST FAILED (Success rate: {success_rate*100:.1f}%)")
        print(f"   Too many errors, capacity insufficient")
    print(f"{'='*60}\n")
    
    return success_rate >= 0.80

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qdrant concurrent capacity")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--workers", type=int, default=50, help="Concurrent workers")
    args = parser.parse_args()
    
    try:
        success = test_concurrent_queries(args.queries, args.workers)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
