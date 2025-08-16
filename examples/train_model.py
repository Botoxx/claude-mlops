#!/usr/bin/env python3
"""
Training Script for ML Models
=============================

Simple training script that can be orchestrated by the Claude MLOps system.
Supports various hyperparameters and provides structured output for orchestration.

This script demonstrates:
- Command-line argument parsing
- Model training with scikit-learn
- Structured output for orchestration
- Error handling and logging

Usage:
    python train_model.py --n_estimators 100 --max_depth 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ML dependencies
try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required ML dependencies: {e}")
    print("Install with: pip install scikit-learn")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ML model with specified hyperparameters"
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--n_estimators', 
        type=int, 
        default=100,
        help='Number of trees in the random forest (default: 100)'
    )
    parser.add_argument(
        '--max_depth', 
        type=str, 
        default='None',
        help='Maximum depth of trees (default: None for unlimited)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Data configuration
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./model_outputs',
        help='Directory to save model artifacts (default: ./model_outputs)'
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save trained model to disk'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def load_and_prepare_data(test_size=0.2, random_state=42, verbose=False):
    """
    Load and prepare the Iris dataset for training.
    
    Args:
        test_size: Fraction of data for testing
        random_state: Random seed for split
        verbose: Enable detailed output
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, class_names)
    """
    if verbose:
        print("📊 Loading Iris dataset...")
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    if verbose:
        print(f"   Dataset shape: {X.shape}")
        print(f"   Number of classes: {len(iris.target_names)}")
        print(f"   Class names: {iris.target_names.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    if verbose:
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names

def create_model(n_estimators, max_depth, random_state=42, verbose=False):
    """
    Create and configure the Random Forest model.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth (None for unlimited)
        random_state: Random seed
        verbose: Enable detailed output
        
    Returns:
        Configured RandomForestClassifier
    """
    # Handle max_depth parameter
    if isinstance(max_depth, str) and max_depth.lower() == 'none':
        max_depth = None
    elif isinstance(max_depth, str):
        try:
            max_depth = int(max_depth)
        except ValueError:
            print(f"Warning: Invalid max_depth '{max_depth}', using None")
            max_depth = None
    
    if verbose:
        print(f"🌳 Creating Random Forest model:")
        print(f"   n_estimators: {n_estimators}")
        print(f"   max_depth: {max_depth}")
        print(f"   random_state: {random_state}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1  # Use single core for consistent timing
    )
    
    return model

def train_model(model, X_train, y_train, verbose=False):
    """
    Train the model on the training data.
    
    Args:
        model: Scikit-learn model to train
        X_train: Training features
        y_train: Training labels
        verbose: Enable detailed output
        
    Returns:
        Trained model and training time
    """
    if verbose:
        print("🚀 Training model...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"   Training completed in {training_time:.3f} seconds")
    
    return model, training_time

def evaluate_model(model, X_test, y_test, class_names, verbose=False):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Names of target classes
        verbose: Enable detailed output
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if verbose:
        print("📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Detailed classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    if verbose:
        print(f"   Test accuracy: {accuracy:.4f}")
        print(f"   Predictions shape: {y_pred.shape}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            print(f"   Top feature importance: {max(importances):.3f}")
    
    # Compile evaluation results
    evaluation = {
        'accuracy': float(accuracy),
        'predictions': y_pred.tolist(),
        'prediction_probabilities': y_pred_proba.tolist(),
        'classification_report': class_report,
        'feature_importances': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
        'test_samples': len(X_test)
    }
    
    return evaluation

def save_model_artifacts(model, evaluation, hyperparams, output_dir, verbose=False):
    """
    Save model and evaluation results to disk.
    
    Args:
        model: Trained model
        evaluation: Evaluation metrics
        hyperparams: Model hyperparameters  
        output_dir: Directory to save artifacts
        verbose: Enable detailed output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"💾 Saving model artifacts to {output_path}")
    
    # Save model
    model_path = output_path / "model.pkl"
    joblib.dump(model, model_path)
    
    # Save evaluation results
    results = {
        'hyperparameters': hyperparams,
        'evaluation': evaluation,
        'model_type': type(model).__name__,
        'timestamp': time.time()
    }
    
    results_path = output_path / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"   Model saved: {model_path}")
        print(f"   Results saved: {results_path}")

def main():
    """Main training function."""
    args = parse_arguments()
    
    if args.verbose:
        print("🤖 Claude MLOps Training Script")
        print("=" * 40)
        print(f"Configuration:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names, class_names = load_and_prepare_data(
            test_size=args.test_size,
            random_state=args.random_state,
            verbose=args.verbose
        )
        
        # Create model
        model = create_model(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            verbose=args.verbose
        )
        
        # Train model
        trained_model, training_time = train_model(
            model, X_train, y_train, 
            verbose=args.verbose
        )
        
        # Evaluate model
        evaluation = evaluate_model(
            trained_model, X_test, y_test, class_names,
            verbose=args.verbose
        )
        
        # Save artifacts if requested
        if args.save_model:
            hyperparams = {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth,
                'random_state': args.random_state
            }
            save_model_artifacts(
                trained_model, evaluation, hyperparams, args.output_dir,
                verbose=args.verbose
            )
        
        # Output results (for orchestration parsing)
        accuracy = evaluation['accuracy']
        print(f"Training completed with accuracy: {accuracy:.4f}")
        
        if args.verbose:
            print(f"\n✅ Training successful!")
            print(f"   Final accuracy: {accuracy:.4f}")
            print(f"   Training time: {training_time:.3f}s")
            print(f"   Model type: {type(trained_model).__name__}")
        
        # Return success
        return 0
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)