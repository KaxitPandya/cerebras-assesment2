"""
plot_expert_histogram.py - Generate expert usage histogram from MoE routing log

Reads moe_routes.jsonl and produces expert_hist.png with analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sys


def load_routing_data(jsonl_path: str):
    """Load routing data from JSONL file."""
    metadata = None
    routes = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if record['type'] == 'meta':
                metadata = record
            elif record['type'] == 'route':
                routes.append(record)
    
    return metadata, routes


def compute_statistics(routes, num_experts=60):
    """Compute expert usage statistics."""
    # Count expert selections
    expert_counts = Counter()
    weighted_counts = Counter()
    
    for route in routes:
        for expert_id, weight in zip(route['topk_ids'], route['topk_weights']):
            expert_counts[expert_id] += 1
            weighted_counts[expert_id] += weight
    
    # Ensure all experts are represented
    for i in range(num_experts):
        if i not in expert_counts:
            expert_counts[i] = 0
            weighted_counts[i] = 0.0
    
    # Convert to arrays
    experts = list(range(num_experts))
    counts = [expert_counts[i] for i in experts]
    weights = [weighted_counts[i] for i in experts]
    
    # Normalize
    total_selections = sum(counts)
    normalized = [c / total_selections if total_selections > 0 else 0 for c in counts]
    
    # Compute entropy
    probs = np.array(normalized)
    probs = probs[probs > 0]  # Remove zeros for log
    entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
    max_entropy = np.log2(num_experts)
    normalized_entropy = entropy / max_entropy
    
    # Top-K experts
    top_3 = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        'experts': experts,
        'counts': counts,
        'normalized': normalized,
        'weighted': weights,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': normalized_entropy,
        'top_3': top_3,
        'total_tokens': len(routes),
        'total_selections': total_selections,
        'num_experts': num_experts
    }


def plot_histogram(stats, metadata, output_path='expert_hist.png'):
    """Generate and save expert usage histogram."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Color bars by usage (hot = more used)
    colors = plt.cm.RdYlBu_r(np.array(stats['normalized']) / max(stats['normalized']) if max(stats['normalized']) > 0 else np.zeros(len(stats['normalized'])))
    
    # Plot 1: Raw counts
    ax1 = axes[0]
    bars1 = ax1.bar(stats['experts'], stats['counts'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Expert ID', fontsize=12)
    ax1.set_ylabel('Selection Count', fontsize=12)
    ax1.set_title(f'MoE Expert Usage Distribution (Layer 0)\n{metadata["model_id"]}', fontsize=14)
    ax1.set_xlim(-1, stats['num_experts'])
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight top 3
    for expert_id, count in stats['top_3']:
        ax1.annotate(f'#{expert_id}\n({count})', 
                    xy=(expert_id, count), 
                    xytext=(expert_id, count + max(stats['counts'])*0.05),
                    ha='center', fontsize=9, fontweight='bold', color='red')
    
    # Plot 2: Normalized distribution
    ax2 = axes[1]
    bars2 = ax2.bar(stats['experts'], stats['normalized'], color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1/stats['num_experts'], color='red', linestyle='--', linewidth=2, label=f'Uniform ({1/stats["num_experts"]:.4f})')
    ax2.set_xlabel('Expert ID', fontsize=12)
    ax2.set_ylabel('Selection Probability', fontsize=12)
    ax2.set_title('Normalized Expert Selection Distribution', fontsize=14)
    ax2.set_xlim(-1, stats['num_experts'])
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f"Total Tokens: {stats['total_tokens']}\n"
        f"Total Selections: {stats['total_selections']}\n"
        f"Top-K per token: {metadata.get('top_k', 4)}\n"
        f"Entropy: {stats['entropy']:.3f} bits\n"
        f"Normalized Entropy: {stats['normalized_entropy']:.3f}\n"
        f"Top-3 Experts: {', '.join([f'#{e}({c})' for e,c in stats['top_3']])}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to {output_path}")


def main():
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else 'moe_routes.jsonl'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'expert_hist.png'
    
    print(f"Loading routing data from {jsonl_path}...")
    metadata, routes = load_routing_data(jsonl_path)
    
    if not routes:
        print("Error: No routing records found!")
        return
    
    print(f"Loaded {len(routes)} routing records")
    print(f"Model: {metadata.get('model_id', 'Unknown')}")
    print(f"Device: {metadata.get('device', 'Unknown')}")
    
    # Determine number of experts (Qwen1.5-MoE has 60 experts)
    num_experts = 60
    
    print("\nComputing statistics...")
    stats = compute_statistics(routes, num_experts)
    
    print(f"\n=== Expert Usage Analysis ===")
    print(f"Total tokens processed: {stats['total_tokens']}")
    print(f"Total expert selections: {stats['total_selections']}")
    print(f"\nTop-3 Most Used Experts:")
    for rank, (expert_id, count) in enumerate(stats['top_3'], 1):
        pct = count / stats['total_selections'] * 100
        print(f"  {rank}. Expert #{expert_id}: {count} selections ({pct:.2f}%)")
    
    print(f"\nEntropy Analysis:")
    print(f"  Entropy: {stats['entropy']:.3f} bits")
    print(f"  Max Entropy (uniform): {stats['max_entropy']:.3f} bits")
    print(f"  Normalized Entropy: {stats['normalized_entropy']:.3f}")
    
    if stats['normalized_entropy'] > 0.9:
        interpretation = "Expert usage is highly uniform - good load balancing."
    elif stats['normalized_entropy'] > 0.7:
        interpretation = "Expert usage is moderately balanced with some specialization."
    else:
        interpretation = "Expert usage is concentrated - some experts dominate."
    
    print(f"  Interpretation: {interpretation}")
    
    print(f"\nGenerating histogram...")
    plot_histogram(stats, metadata, output_path)
    
    # Save analysis to JSON
    analysis = {
        'total_tokens': stats['total_tokens'],
        'total_selections': stats['total_selections'],
        'top_3_experts': [{'expert_id': e, 'count': c, 'percentage': c/stats['total_selections']*100} for e, c in stats['top_3']],
        'entropy_bits': stats['entropy'],
        'max_entropy_bits': stats['max_entropy'],
        'normalized_entropy': stats['normalized_entropy'],
        'interpretation': interpretation
    }
    
    with open('analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("Analysis saved to analysis.json")


if __name__ == '__main__':
    main()
