from typing import Dict, List, Any

# Standard thresholds for the 5 tasks (Absolute score values or relative multipliers)
# If a competition has an AUC metric (e.g. TPS May 2022), score ranges [0.5, 1.0].
# If it has classification error (e.g. leaf), lower is better or binned.
# To make it universally robust for our preliminary ablation harness, we binned based on
# a relative threshold against the initial baseline score.

def compute_medal(score: float, baseline_score: float, maximize: bool = True) -> str:
    """
    Categorizes a score into Gold, Silver, Bronze, or None relative to the baseline score.
    """
    if score is None:
        return "None"
        
    if maximize:
        if baseline_score == 0:
            pct_diff = 0.0
        else:
            pct_diff = (score - baseline_score) / baseline_score
            
        if pct_diff >= 0.05:
            return "Gold"
        elif pct_diff >= 0.02:
            return "Silver"
        elif pct_diff >= 0.005:
            return "Bronze"
    else:
        # Lower is better (e.g. loss or classification error)
        if baseline_score == 0:
            pct_diff = 0.0
        else:
            pct_diff = (baseline_score - score) / baseline_score
            
        if pct_diff >= 0.05:
            return "Gold"
        elif pct_diff >= 0.02:
            return "Silver"
        elif pct_diff >= 0.005:
            return "Bronze"
            
    return "None"

def calculate_ablation_metrics(
    nodes_history: List[dict],
    baseline_score: float,
    maximize: bool = True
) -> Dict[str, Any]:
    """
    Calculates ablation metrics across a run history:
    - Medal Rate: Fraction of leaf implementation runs achieving any medal
    - Gold Rate: Fraction of leaf implementation runs achieving a gold medal
    - Avg tokens per node: Total input/output tokens divided by total nodes
    - Pool hit rate: Ratio of pool_hit nodes to total technique nodes
    - Overcome rate: Fraction of implementation nodes scoring higher than baseline
    """
    tech_nodes = [n for n in nodes_history if n.get("type") == "technique"]
    impl_nodes = [n for n in nodes_history if n.get("type") == "implementation"]
    
    total_tech = len(tech_nodes)
    total_impl = len(impl_nodes)
    
    # Pool hit rate
    pool_hits = sum(1 for n in tech_nodes if n.get("status") == "pool_hit")
    pool_hit_rate = pool_hits / total_tech if total_tech > 0 else 0.0
    
    # Overcome rate and medals
    overcomes = 0
    medals_count = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
    
    for node in impl_nodes:
        score = node.get("score")
        if score is None:
            continue
            
        # Check if overcame baseline
        if maximize:
            if score > baseline_score:
                overcomes += 1
        else:
            if score < baseline_score:
                overcomes += 1
                
        # Calculate medal
        medal = compute_medal(score, baseline_score, maximize=maximize)
        medals_count[medal] += 1
        
    overcome_rate = overcomes / total_impl if total_impl > 0 else 0.0
    
    total_medals = medals_count["Gold"] + medals_count["Silver"] + medals_count["Bronze"]
    medal_rate = total_medals / total_impl if total_impl > 0 else 0.0
    gold_rate = medals_count["Gold"] / total_impl if total_impl > 0 else 0.0
    
    return {
        "pool_hit_rate": pool_hit_rate,
        "overcome_rate": overcome_rate,
        "medal_rate": medal_rate,
        "gold_rate": gold_rate,
        "medals_count": medals_count
    }
