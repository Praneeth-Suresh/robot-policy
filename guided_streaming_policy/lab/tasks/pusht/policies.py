"""Reference policies for the static condition, without importing research runners."""

from guided_streaming_policy.lab.tasks.base import PolicySpec, TunableSpec

__all__ = ["POLICIES"]

_SI_CHECKPOINT = "Push-T/checkpoints/pusht_state_1000ep_si.ckpt"
_SI_PAPERS = (
    "Streaming Flow Policy (Jiang, Ancha et al., CoRL 2025, arXiv:2505.21851); "
    "Stochastic interpolants (Albergo, Boffi, Vanden-Eijnden, arXiv:2303.08797)"
)

POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec(
        name="pusht/naive",
        summary=(
            "streaming stochastic-interpolant policy with a hand-written repulsive force "
            "around the obstacle."
        ),
        paper=_SI_PAPERS,
        entry_point="inference.static.inference_naive:run_naive_inference",
        checkpoints=(_SI_CHECKPOINT,),
        tunables=(
            TunableSpec(
                "max_guidance_scale",
                6.0,
                "Maximum repulsion scale; Push-T/run_trial.py static default 6.0.",
            ),
        ),
    ),
    PolicySpec(
        name="pusht/ccg",
        summary="same policy steered by a learned collision critic (CCG).",
        paper=_SI_PAPERS + "; CCG: repo-specific method",
        entry_point="inference.static.inference_ccg:run_ccg_inference",
        checkpoints=(_SI_CHECKPOINT, "Push-T/checkpoints/CCG_V6_ep200.pth"),
        tunables=(
            TunableSpec(
                "max_guidance_scale",
                4.0,
                "Maximum critic scale; Push-T/run_trial.py static default 4.0.",
            ),
        ),
    ),
    PolicySpec(
        name="pusht/steg",
        summary="streaming stochastic-interpolant SDE with ensemble guidance (STEG).",
        paper=_SI_PAPERS + "; STEG: repo-specific method",
        entry_point="inference.static.inference_steg:run_steg_sde_inference",
        checkpoints=(_SI_CHECKPOINT,),
        tunables=(
            TunableSpec(
                "guidance_scale",
                1.5,
                "Ensemble guidance scale; run_steg_sde_inference default 1.5 "
                "in Push-T/inference/static/inference_steg.py.",
            ),
            TunableSpec(
                "N_ensemble",
                64,
                "Ensemble size; run_steg_sde_inference default 64 "
                "in Push-T/inference/static/inference_steg.py.",
            ),
        ),
    ),
    PolicySpec(
        name="pusht/dp",
        summary="DDPM diffusion policy with inference-time gradient guidance.",
        paper="Diffusion Policy (Chi et al., RSS 2023, arXiv:2303.04137)",
        entry_point="inference.static.inference_dp_gconva:run_gconv_a_dp_inference",
        checkpoints=("Push-T/checkpoints/pusht_state_100ep_dp.ckpt",),
        tunables=(
            TunableSpec(
                "guidance_scale",
                2.0,
                "Gradient guidance scale; run_gconv_a_dp_inference default 2.0 "
                "in Push-T/inference/static/inference_dp_gconva.py.",
            ),
            TunableSpec(
                "guidance_sign",
                1.0,
                "Gradient sign; run_gconv_a_dp_inference default +1.0 "
                "in Push-T/inference/static/inference_dp_gconva.py.",
            ),
            TunableSpec(
                "grad_clip_norm",
                1.0,
                "Gradient norm limit; run_gconv_a_dp_inference default 1.0 "
                "in Push-T/inference/static/inference_dp_gconva.py.",
            ),
        ),
    ),
    PolicySpec(
        name="pusht/flow",
        summary="flow-matching policy with inference-time gradient guidance.",
        paper=(
            "Flow matching policy as implemented in this repo; "
            "see Streaming Flow Policy (arXiv:2505.21851)"
        ),
        entry_point="inference.static.inference_fp_gconva:run_gcov_a_fp_inference",
        checkpoints=("Push-T/checkpoints/pusht_state_flow_100ep.ckpt",),
        tunables=(
            TunableSpec(
                "guidance_scale",
                5.0,
                "Gradient guidance scale; run_gcov_a_fp_inference default 5.0 "
                "in Push-T/inference/static/inference_fp_gconva.py.",
            ),
        ),
    ),
)
