def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    dense_flops = n_experts * d_in * d_out
    moe_flops = k_active * d_in * d_out
    return round(((dense_flops - moe_flops) / dense_flops) * 100.0, 1)

if __name__ == "__main__":
    ans = compute_efficiency(1000, 2, 512, 512)
    print(ans)