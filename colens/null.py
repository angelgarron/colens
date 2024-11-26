def null_snr(
    rho_coh,
    rho_coinc,
):
    # Calculate null SNRs
    null2 = rho_coinc**2 - rho_coh**2
    # Numerical errors may make this negative and break the sqrt, so set
    # negative values to 0.
    null2[null2 < 0] = 0
    null = null2**0.5
    return null
