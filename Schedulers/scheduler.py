from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


# ── Scheduler factories ──────────────────────────────────────────────────────

def identity_lr_lambda(_):
    return 1.0


def _num_optimizer_steps(args):
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    num_steps = int(getattr(args, "num_steps", 0))
    return max(1, (num_steps + grad_accum_steps - 1) // grad_accum_steps)


def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=_num_optimizer_steps(args),
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    step_size = max(1, int(getattr(args, "lr_step_size", 10000)) // grad_accum_steps)
    return StepLR(
        optimizer,
        step_size=step_size,
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def lambda_scheduler(optimizer, args):
    """LambdaLR with a constant factor of 1.0 — learning rate stays fixed."""
    return LambdaLR(optimizer, lr_lambda=identity_lr_lambda)


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}
