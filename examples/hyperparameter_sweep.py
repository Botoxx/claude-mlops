#!/usr/bin/env python3
"""
Hyperparameter Sweep Example
============================

Demonstrates parallel hyperparameter optimization using Claude orchestration.
This example trains multiple RandomForest models with different parameters
simultaneously and compares their performance.

Usage:
    python examples/hyperparameter_sweep.py

Expected output:
    - 16 parallel training jobs
    - Results ranked by accuracy
    - Total execution time
    - Success/failure statistics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import subprocess
import concurrent.futures
import itertools
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from orchestrator.core import MLOpsOrchestrator
from utils.logging import setup_logger

logger = setup_logger(__name__)

def run_training_job(params: Tuple[int, int], script_path: str) -> Dict[str, Any]:
    """
    Execute a single training job with given hyperparameters.
    
    Args:
        params: Tuple of (n_estimators, max_depth)
        script_path: Path to the training script
        
    Returns:
        Dictionary containing job results and metadata
    """
    n_estimators, max_depth = params
    job_id = f"job_{n_estimators}_{max_depth}"
    
    logger.info(f"🚀 Starting {job_id}: n_estimators={n_estimators}, max_depth={max_depth}")
    
    start_time = time.time()
    
    # Prepare command with parameters
    cmd = [
        'python', script_path,
        '--n_estimators', str(n_estimators),
        '--max_depth', str(max_depth) if max_depth is not None else 'None'
    ]
    
    try:
        # Execute training job
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path(__file__).parent
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Parse accuracy from output
            for line in result.stdout.strip().split('\n'):
                if 'Training completed with accuracy:' in line:
                    accuracy = float(line.split(':')[-1].strip())
                    
                    logger.info(f"✅ {job_id} completed: {accuracy:.4f} accuracy ({duration:.1f}s)")
                    
                    return {
                        'job_id': job_id,
                        'params': {'n_estimators': n_estimators, 'max_depth': max_depth},
                        'accuracy': accuracy,
                        'duration': duration,
                        'status': 'success',
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
            
            # Could not parse accuracy
            logger.warning(f"⚠️ {job_id} completed but could not parse accuracy")
            return {
                'job_id': job_id,
                'params': {'n_estimators': n_estimators, 'max_depth': max_depth},
                'accuracy': 0.0,
                'duration': duration,
                'status': 'parse_error',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            # Job failed
            logger.error(f"❌ {job_id} failed: {result.stderr}")
            return {
                'job_id': job_id,
                'params': {'n_estimators': n_estimators, 'max_depth': max_depth},
                'accuracy': 0.0,
                'duration': time.time() - start_time,
                'status': 'failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {job_id} timeout after 5 minutes")
        return {
            'job_id': job_id,
            'params': {'n_estimators': n_estimators, 'max_depth': max_depth},
            'accuracy': 0.0,
            'duration': 300,
            'status': 'timeout',
            'stdout': '',
            'stderr': 'Job timeout'
        }
    except Exception as e:
        logger.error(f"💥 {job_id} exception: {e}")
        return {
            'job_id': job_id,
            'params': {'n_estimators': n_estimators, 'max_depth': max_depth},
            'accuracy': 0.0,
            'duration': time.time() - start_time,
            'status': 'exception',
            'stdout': '',
            'stderr': str(e)
        }

def orchestrate_hyperparameter_sweep() -> List[Dict[str, Any]]:
    """
    Orchestrate a complete hyperparameter sweep using parallel execution.
    
    Returns:
        List of job results with performance metrics
    """
    logger.info("🎯 Starting Claude-powered hyperparameter sweep")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 5, 7, None]  # None = unlimited depth
    }
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        param_grid['n_estimators'],
        param_grid['max_depth']
    ))
    
    logger.info(f"📊 Testing {len(param_combinations)} parameter combinations")
    logger.info(f"🔧 Parameter grid: {param_grid}")
    
    # Training script path
    script_path = Path(__file__).parent / 'train_model.py'
    
    # Parallel execution configuration
    max_workers = min(8, len(param_combinations))  # Adjust based on system
    logger.info(f"⚡ Using {max_workers} parallel workers")
    
    start_time = time.time()
    results = []
    
    # Execute jobs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {
            executor.submit(run_training_job, params, str(script_path)): params
            for params in param_combinations
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"💥 Exception processing {params}: {e}")
                results.append({
                    'job_id': f"job_{params[0]}_{params[1]}",
                    'params': {'n_estimators': params[0], 'max_depth': params[1]},
                    'accuracy': 0.0,
                    'duration': 0.0,
                    'status': 'executor_exception',
                    'stdout': '',
                    'stderr': str(e)
                })
    
    total_time = time.time() - start_time
    logger.info(f"⏱️ Total sweep completed in {total_time:.1f} seconds")
    
    return results

def analyze_results(results: List[Dict[str, Any]]) -> None:
    """
    Analyze and display hyperparameter sweep results.
    
    Args:
        results: List of job results from the sweep
    """
    # Filter successful runs
    successful_runs = [r for r in results if r['status'] == 'success']
    failed_runs = [r for r in results if r['status'] != 'success']
    
    logger.info(f"\n📈 HYPERPARAMETER SWEEP ANALYSIS")
    logger.info("=" * 60)
    
    if not successful_runs:
        logger.error("❌ No successful runs! Check your setup and try again.")
        return
    
    # Sort by accuracy (descending)
    successful_runs.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Display top results
    logger.info(f"🏆 TOP 10 RESULTS:")
    logger.info(f"{'Rank':<6} {'n_est':<8} {'depth':<8} {'Accuracy':<12} {'Time(s)':<8}")
    logger.info("-" * 50)
    
    for i, result in enumerate(successful_runs[:10]):
        params = result['params']
        depth_str = str(params['max_depth']) if params['max_depth'] is not None else "None"
        logger.info(
            f"{i+1:<6} {params['n_estimators']:<8} {depth_str:<8} "
            f"{result['accuracy']:<12.4f} {result['duration']:<8.1f}"
        )
    
    # Best model analysis
    best_result = successful_runs[0]
    logger.info(f"\n🎯 BEST CONFIGURATION:")
    logger.info(f"   Parameters: {best_result['params']}")
    logger.info(f"   Accuracy: {best_result['accuracy']:.4f}")
    logger.info(f"   Training time: {best_result['duration']:.1f}s")
    
    # Statistics
    total_jobs = len(results)
    success_rate = len(successful_runs) / total_jobs * 100
    avg_accuracy = sum(r['accuracy'] for r in successful_runs) / len(successful_runs)
    avg_duration = sum(r['duration'] for r in successful_runs) / len(successful_runs)
    
    logger.info(f"\n📊 STATISTICS:")
    logger.info(f"   Total jobs: {total_jobs}")
    logger.info(f"   Successful: {len(successful_runs)} ({success_rate:.1f}%)")
    logger.info(f"   Failed: {len(failed_runs)}")
    logger.info(f"   Average accuracy: {avg_accuracy:.4f}")
    logger.info(f"   Average duration: {avg_duration:.1f}s")
    
    # Failure analysis
    if failed_runs:
        logger.info(f"\n⚠️ FAILURE ANALYSIS:")
        failure_types = {}
        for result in failed_runs:
            status = result['status']
            failure_types[status] = failure_types.get(status, 0) + 1
        
        for status, count in failure_types.items():
            logger.info(f"   {status}: {count} jobs")

def save_results(results: List[Dict[str, Any]], output_file: str = "sweep_results.json") -> None:
    """
    Save sweep results to JSON file for further analysis.
    
    Args:
        results: Job results to save
        output_file: Output filename
    """
    output_path = Path(__file__).parent / output_file
    
    # Prepare data for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        # Ensure all values are JSON serializable
        for key, value in serializable_result.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                continue
            else:
                serializable_result[key] = str(value)
        serializable_results.append(serializable_result)
    
    # Add metadata
    output_data = {
        'metadata': {
            'timestamp': time.time(),
            'total_jobs': len(results),
            'successful_jobs': len([r for r in results if r['status'] == 'success']),
            'sweep_type': 'hyperparameter_optimization'
        },
        'results': serializable_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"💾 Results saved to {output_path}")

def main():
    """Main execution function for the hyperparameter sweep example."""
    
    logger.info("🔄 Claude MLOps Orchestrator - Hyperparameter Sweep Example")
    logger.info("=" * 70)
    
    try:
        # Run the hyperparameter sweep
        results = orchestrate_hyperparameter_sweep()
        
        # Analyze and display results  
        analyze_results(results)
        
        # Save results for further analysis
        save_results(results)
        
        logger.info("\n✅ Hyperparameter sweep completed successfully!")
        logger.info("🚀 Ready for production model deployment!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Sweep interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Sweep failed with error: {e}")
        raise

if __name__ == "__main__":
    main()