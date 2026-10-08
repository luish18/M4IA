"""Audited interpretation of the current MatMul/Gemm C dispatch conditions.

The C kernels, build symbol renames, and runtime dispatch remain operational
truth. This helper only prevents the mapper from pricing a known whole-kernel
Generic fallback with a tuned cluster rate.
"""

from __future__ import annotations

CVA6_GENERIC = "cva6.fp32.matmul_gemm.deeploy_generic"
SNITCH_TUNED = "snitch.fp32.matmul_gemm.ssr_frep"
SNITCH_GENERIC = "snitch.fp32.matmul_gemm.deeploy_generic_fallback"
SPATZ_TUNED = "spatz.fp32.matmul_gemm.rvv_tuned"
SPATZ_GENERIC = "spatz.fp32.matmul_gemm.deeploy_generic_fallback"


def _integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _unknown(reason: str) -> dict:
    return {
        "implementation_id": None,
        "fallback_used": None,
        "fallback_from": None,
        "fallback_to": None,
        "fallback_reason": None,
        "implementation_detail": None,
        "unknown_reason": reason,
    }


def _known(
    implementation_id: str,
    *,
    fallback_used: bool,
    fallback_from: str | None = None,
    fallback_reason: str | None = None,
    implementation_detail: str | None = None,
) -> dict:
    return {
        "implementation_id": implementation_id,
        "fallback_used": fallback_used,
        "fallback_from": fallback_from,
        "fallback_to": implementation_id if fallback_used else None,
        "fallback_reason": fallback_reason,
        "implementation_detail": implementation_detail,
        "unknown_reason": None,
    }


def resolve_matmul_gemm_implementation(
    engine,
    op,
    M,
    N,
    O,
    *,
    transA=0,
    transB=0,
) -> dict:
    """Interpret audited primitive arguments; preserve ambiguity as unknown."""
    if op not in {"MatMul", "Gemm"}:
        return _unknown(f"no deterministic implementation rule for operator {op!r}")
    if engine == "cva6":
        return _known(CVA6_GENERIC, fallback_used=False)
    if engine not in {"snitch", "spatz"}:
        return _unknown(f"no deterministic implementation rule for engine {engine!r}")

    m, n, o = (_integer(value) for value in (M, N, O))
    if None in {m, n, o} or min(m, n, o) < 0:
        return _unknown("dimensions are incomplete, non-integer, or negative")

    if engine == "snitch":
        if m == 0 or n == 0:
            return _known(
                SNITCH_GENERIC,
                fallback_used=True,
                fallback_from=SNITCH_TUNED,
                fallback_reason="empty_input_dimension",
            )
        if o < 8:
            return _known(
                SNITCH_GENERIC,
                fallback_used=True,
                fallback_from=SNITCH_TUNED,
                fallback_reason="output_columns_below_ssr_unroll",
            )
        return _known(
            SNITCH_TUNED,
            fallback_used=False,
            implementation_detail=(
                "scalar_tail_for_remaining_output_columns" if o % 8 else None
            ),
        )

    if m == 0 or n == 0 or o == 0:
        return _known(
            SPATZ_GENERIC,
            fallback_used=True,
            fallback_from=SPATZ_TUNED,
            fallback_reason="empty_dimension",
        )
    if op == "Gemm":
        trans_a, trans_b = _integer(transA), _integer(transB)
        if trans_a is None or trans_b is None:
            return _unknown("GEMM transpose arguments are unavailable or non-integer")
        if trans_a or trans_b:
            return _known(
                SPATZ_GENERIC,
                fallback_used=True,
                fallback_from=SPATZ_TUNED,
                fallback_reason="transposed_operand",
            )
    return _known(SPATZ_TUNED, fallback_used=False)
