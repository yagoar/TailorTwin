"""ABS — the adversarial body simulator (paper §3.2).

ABS searches the SMPL-X shape space for body shapes that the *current*
BMnet predicts badly, and turns them into extra training pairs. Because
the whole chain — shape betas → SMPL-X mesh → silhouette render →
BMnet → measurements, and betas → mesh → ground-truth measurements — is
differentiable, the search is plain gradient *ascent* on the BMnet loss
(Eq. 6, η = 0.1, k = 10, betas clamped to [-3, 3]).

This file provides every differentiable submodule the paper's Figure 1b
needs, re-implemented for SMPL-X:

* ``render_pair``        — soft front+side silhouettes (the renderer R),
                           reusing ``fit.silhouette_render.soft_silhouette``.
* ``mesh_measurements`` — the 14 BodyM measurements as mesh geometry
                           (the extractor g): girths are angle-sorted
                           cross-section perimeters, lengths are joint
                           distances. Differentiable in the vertices.
* ``mesh_height_weight`` — the regressor h, here computed directly:
                           height from the bounding box, weight from the
                           closed-mesh volume × tissue density.
* ``AbsSampler``         — fixed canonical A-pose; samples seed betas,
                           runs the adversarial ascent, and emits
                           ``(input tensor, standardized target)`` pairs
                           in exactly the layout ``dataset.build_input``
                           produces, so ABS batches drop straight into
                           the BMnet training loop.

Differences from the paper, all deliberate and bounded: pose is a fixed
canonical A-pose rather than sampled from BodyM poses (BMnet sees only
silhouettes, and the real-data fine-tune re-anchors pose statistics);
the measurement function g is defined here rather than by registered
BodyM vertex paths, but it is used identically for the synthetic target
and inside the search, so it is self-consistent.
"""
from __future__ import annotations

import numpy as np
import torch

from . import MEAS_COLS
from .model import Standardizer

# Tissue density for the volume→weight estimate (kg/m³). Whole-body
# density sits just under water because of the lungs.
DENSITY_KG_M3 = 1010.0

# Adversarial ascent hyper-parameters (paper Eq. 6).
ABS_LR = 0.1
ABS_STEPS = 10
BETA_CLAMP = 3.0

# SMPL-X body joints used for the length measurements.
J_PELVIS, J_LHIP = 0, 1
J_LKNEE, J_LANKLE = 4, 7
J_LSHOULDER, J_RSHOULDER = 16, 17
J_LELBOW, J_LWRIST = 18, 20

# Girth slice levels. Torso girths are a fraction of full body height;
# limb girths are a fraction of that limb region's own vertical extent
# (0 = distal/low end, 1 = proximal/high end). Definitional — g just has
# to be a fixed, smooth function of the mesh.
_TORSO_LEVEL = {"chest": 0.70, "waist": 0.61, "hip": 0.52}

# Limb girths: (proximal joint, distal joint, fraction along that bone).
_ARM_SEG = {
    "bicep": (J_LSHOULDER, J_LELBOW, 0.50),
    "forearm": (J_LELBOW, J_LWRIST, 0.45),
    "wrist": (J_LELBOW, J_LWRIST, 0.95),
}
_LEG_SEG = {
    "thigh": (J_LHIP, J_LKNEE, 0.30),
    "calf": (J_LKNEE, J_LANKLE, 0.40),
    "ankle": (J_LKNEE, J_LANKLE, 0.97),
}

# Number of directions in the Cauchy-formula perimeter estimate.
_CAUCHY_K = 24


# ----------------------------------------------------------------------
# differentiable measurement extractor  g(mesh)
# ----------------------------------------------------------------------
def _cauchy_perimeter(p: torch.Tensor) -> torch.Tensor:
    """Convex perimeter (m) of a 2-D point set ``p`` (M, 2).

    Cauchy's formula — for a convex curve the perimeter equals the
    integral of its breadth over directions, ``L = ∫₀^π w(θ) dθ`` —
    discretized over ``_CAUCHY_K`` directions. Each breadth ``w(θ)`` is a
    max-minus-min of projections, so the estimate is differentiable in
    the point positions and, unlike a sorted-polygon perimeter, robust
    to slice-band thickness (a thick band of a limb cylinder still
    yields the cylinder's breadth profile)."""
    k = torch.arange(_CAUCHY_K, device=p.device, dtype=p.dtype)
    theta = k * (np.pi / _CAUCHY_K)
    dirs = torch.stack([torch.cos(theta), torch.sin(theta)])   # (2, K)
    proj = p @ dirs                                 # (M, K)
    width = proj.max(0).values - proj.min(0).values            # (K,)
    return (np.pi / _CAUCHY_K) * width.sum()


def _slice_girth(pts: torch.Tensor, lo, hi, axis: int = 1) -> torch.Tensor:
    """Convex perimeter (m) of the cross-section of ``pts`` in the
    horizontal band ``lo ≤ pts[:, axis] ≤ hi`` — used for the upright
    torso girths. ``lo`` / ``hi`` carry no gradient (hard membership)."""
    coord = pts[:, axis]
    if isinstance(lo, torch.Tensor):
        lo = lo.detach()
    if isinstance(hi, torch.Tensor):
        hi = hi.detach()
    sel = (coord >= lo) & (coord <= hi)
    if int(sel.sum()) < 6:
        return pts.new_tensor(0.0)
    other = [a for a in (0, 1, 2) if a != axis]
    return _cauchy_perimeter(pts[sel][:, other])


def _limb_girth(verts: torch.Tensor, mask: torch.Tensor,
                joints: torch.Tensor, j_a: int, j_b: int, frac: float,
                half: float = 0.018) -> torch.Tensor:
    """Girth (m) of a limb sliced *perpendicular to the bone*.

    The bone axis is the joint-to-joint direction ``j_a → j_b``; the cut
    plane sits at ``frac`` along that bone. Slicing perpendicular to the
    bone — rather than by a horizontal Y-band — keeps the girth honest
    for the angled arms and legs of the A-pose. Differentiable: the axis
    and the cut centre come from the (shape-dependent) joints, the
    perimeter from the projected vertices."""
    a, b = joints[j_a], joints[j_b]
    axis = (b - a)
    axis = axis / (axis.norm() + 1e-9)
    centre = a + frac * (b - a)
    rel = verts[mask] - centre
    t = rel @ axis.detach()                         # position along bone
    sel = t.abs() < half
    if int(sel.sum()) < 6:
        return verts.new_tensor(0.0)
    # Orthonormal basis of the cut plane.
    up = axis.new_tensor([0.0, 1.0, 0.0])
    e1 = torch.cross(axis, up, dim=0)
    if float(e1.norm()) < 1e-3:
        e1 = torch.cross(axis, axis.new_tensor([1.0, 0.0, 0.0]), dim=0)
    e1 = e1 / (e1.norm() + 1e-9)
    e2 = torch.cross(axis, e1, dim=0)
    ring = rel[sel]
    p = torch.stack([ring @ e1, ring @ e2], dim=1)  # (M, 2) in-plane
    return _cauchy_perimeter(p)


def mesh_measurements(verts: torch.Tensor, joints: torch.Tensor,
                      masks: dict[str, torch.Tensor]) -> torch.Tensor:
    """The 14 BodyM measurements (cm) of one SMPL-X mesh, in MEAS_COLS
    order. Differentiable in ``verts`` / ``joints``."""
    y = verts[:, 1]
    y_lo, y_hi = y.min(), y.max()
    span = y_hi - y_lo
    torso = masks["torso"]

    def torso_girth(name: str) -> torch.Tensor:
        lvl = y_lo + _TORSO_LEVEL[name] * span
        return _slice_girth(verts[torso], lvl - 0.02, lvl + 0.02)

    arm, leg = masks["left_arm"], masks["left_leg"]

    def arm_girth(name: str) -> torch.Tensor:
        ja, jb, frac = _ARM_SEG[name]
        return _limb_girth(verts, arm, joints, ja, jb, frac)

    def leg_girth(name: str) -> torch.Tensor:
        ja, jb, frac = _LEG_SEG[name]
        return _limb_girth(verts, leg, joints, ja, jb, frac)

    # Lengths from the body joints.
    arm_len = (torch.linalg.norm(joints[J_LELBOW] - joints[J_LSHOULDER])
               + torch.linalg.norm(joints[J_LWRIST] - joints[J_LELBOW]))
    leg_len = (torch.linalg.norm(joints[J_LKNEE] - joints[J_LHIP])
               + torch.linalg.norm(joints[J_LANKLE] - joints[J_LKNEE]))
    sh_breadth = torch.linalg.norm(
        joints[J_LSHOULDER] - joints[J_RSHOULDER])
    crotch_y = verts[torso, 1].min()
    sh_to_crotch = joints[J_LSHOULDER, 1] - crotch_y

    vals = {
        "ankle": leg_girth("ankle"),
        "arm-length": arm_len,
        "bicep": arm_girth("bicep"),
        "calf": leg_girth("calf"),
        "chest": torso_girth("chest"),
        "forearm": arm_girth("forearm"),
        "height": span,
        "hip": torso_girth("hip"),
        "leg-length": leg_len,
        "shoulder-breadth": sh_breadth,
        "shoulder-to-crotch": sh_to_crotch,
        "thigh": leg_girth("thigh"),
        "waist": torso_girth("waist"),
        "wrist": arm_girth("wrist"),
    }
    return torch.stack([vals[c] for c in MEAS_COLS]) * 100.0   # m → cm


def mesh_height_weight(verts: torch.Tensor,
                       faces: torch.Tensor) -> tuple[torch.Tensor,
                                                     torch.Tensor]:
    """Regressor h: height (cm) from the bounding box, weight (kg) from
    the signed closed-mesh volume × tissue density. Differentiable."""
    height = (verts[:, 1].max() - verts[:, 1].min()) * 100.0
    tri = verts[faces]                                   # (F, 3, 3)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    vol = torch.abs((v0 * torch.cross(v1, v2, dim=1)).sum() / 6.0)
    weight = vol * DENSITY_KG_M3
    return height, weight


# ----------------------------------------------------------------------
# differentiable renderer  R(mesh)
# ----------------------------------------------------------------------
_BODY_FRAC = 0.86          # body height as a fraction of the image height


def _face_points(verts: torch.Tensor, faces: torch.Tensor,
                 bary: torch.Tensor) -> torch.Tensor:
    """Dense surface samples — every face sampled on a barycentric grid.
    Linear in the vertices, so the splat stays differentiable in betas."""
    tri = verts[faces]                                   # (F, 3, 3)
    return torch.einsum("bk,fkc->fbc", bary, tri).reshape(-1, 3)


def render_pair(verts: torch.Tensor, faces: torch.Tensor,
                bary: torch.Tensor, kernel: torch.Tensor,
                img_h: int, img_w: int) -> torch.Tensor:
    """One mesh → a (img_h, 2*img_w) soft silhouette: front then side.

    Orthographic projection; the front view reads X/Y, the side view
    reads Z/Y at the same metric pixel scale so the two silhouettes keep
    consistent proportions."""
    from ..fit.silhouette_render import soft_silhouette

    pts = _face_points(verts, faces, bary)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    y_lo, y_hi = y.min(), y.max()
    scale = _BODY_FRAC * img_h / (y_hi - y_lo + 1e-6)
    v = img_h / 2 - scale * (y - 0.5 * (y_lo + y_hi))

    def view(horiz: torch.Tensor) -> torch.Tensor:
        u = img_w / 2 + scale * (horiz - horiz.mean())
        uv = torch.stack([u, v], dim=-1)
        return soft_silhouette(uv, img_h, img_w, kernel)

    return torch.cat([view(x), view(z)], dim=1)          # (H, 2W)


# ----------------------------------------------------------------------
# adversarial sampler
# ----------------------------------------------------------------------
class AbsSampler:
    """Holds the SMPL-X model and renders/measures synthetic bodies.

    One instance is reused for the whole ABS phase: it seeds random
    betas, runs the gradient-ascent search against the live BMnet, and
    returns batches shaped like ``dataset.build_input`` output."""

    def __init__(self, *, model_folder: str = "data/body_models",
                 gender: str = "neutral", num_betas: int = 14,
                 img_h: int = 256, img_w: int = 192,
                 a_pose_shoulder_deg: float = 30.0,
                 device: torch.device | str = "cpu",
                 bary_subdiv: int = 2) -> None:
        import smplx

        from ..fit.refine_to_tape import _build_a_pose
        from ..fit.silhouette_render import _gauss_kernel
        from ..measure.regions import region_vertex_mask

        self.dev = torch.device(device)
        self.img_h, self.img_w = img_h, img_w
        self.num_betas = num_betas

        self.bm = smplx.create(
            model_path=model_folder, model_type="smplx", gender=gender,
            num_betas=num_betas, use_pca=False, flat_hand_mean=True,
            batch_size=1).to(self.dev)
        self.faces = torch.from_numpy(
            self.bm.faces.astype(np.int64)).to(self.dev)
        self.pose = torch.from_numpy(
            _build_a_pose(a_pose_shoulder_deg).reshape(1, -1).astype(
                np.float32)).to(self.dev)
        self.kernel = _gauss_kernel(3.0, self.dev)

        self.masks = {
            r: torch.from_numpy(region_vertex_mask(
                (r,), model_folder=model_folder, gender=gender)).to(self.dev)
            for r in ("torso", "left_arm", "left_leg")}

        # Barycentric grid for dense face sampling.
        n = bary_subdiv
        b = [(i / n, j / n, 1 - i / n - j / n)
             for i in range(n + 1) for j in range(n + 1 - i)]
        self.bary = torch.tensor(b, dtype=torch.float32, device=self.dev)

    # -- forward: betas → mesh ------------------------------------------
    def _mesh(self, betas: torch.Tensor) -> tuple[torch.Tensor,
                                                  torch.Tensor]:
        """One betas vector → (vertices, joints)."""
        out = self.bm(
            betas=betas[None],
            body_pose=self.pose,
            global_orient=torch.zeros(1, 3, device=self.dev),
            transl=torch.zeros(1, 3, device=self.dev))
        return out.vertices[0], out.joints[0]

    # -- betas → BMnet input tensor + true measurement target -----------
    def _input_and_target(self, betas: torch.Tensor,
                           std: Standardizer) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        verts, joints = self._mesh(betas)
        sil = render_pair(verts, self.faces, self.bary, self.kernel,
                          self.img_h, self.img_w)             # (H, 2W)
        meas = mesh_measurements(verts, joints, self.masks)   # (14,) cm
        height, weight = mesh_height_weight(verts, self.faces)

        hw_mean = torch.tensor(std.hw_mean, dtype=torch.float32,
                               device=self.dev)
        hw_std = torch.tensor(std.hw_std, dtype=torch.float32,
                              device=self.dev)
        hwz = (torch.stack([height, weight]) - hw_mean) / hw_std
        planes = hwz[:, None, None].expand(2, *sil.shape)
        x = torch.cat([sil[None], planes], dim=0)             # (3, H, 2W)

        m_mean = torch.tensor(std.meas_mean, dtype=torch.float32,
                              device=self.dev)
        m_std = torch.tensor(std.meas_std, dtype=torch.float32,
                             device=self.dev)
        z = (meas - m_mean) / m_std
        return x, z

    def seed_betas(self, n: int, sigma: float = 1.0) -> torch.Tensor:
        """Random seed shapes — Gaussian in the SMPL-X betas, clamped to
        the valid range. The adversarial step then walks them toward the
        shapes BMnet handles worst."""
        b = sigma * torch.randn(n, self.num_betas, device=self.dev)
        return b.clamp(-BETA_CLAMP, BETA_CLAMP)

    # -- the adversarial search (paper Eq. 6) ---------------------------
    def adversarial(self, model: torch.nn.Module, std: Standardizer,
                    betas: torch.Tensor, *, steps: int = ABS_STEPS,
                    lr: float = ABS_LR) -> torch.Tensor:
        """Gradient-*ascent* on the BMnet loss: nudge each betas vector
        toward the shape whose silhouettes BMnet measures worst.

        BMnet weights are frozen for the search; only betas move. The
        loss couples both differentiable paths — prediction error and
        the geometry target — exactly as in the paper's Figure 1b."""
        m_std = torch.tensor(std.meas_std, dtype=torch.float32,
                             device=self.dev)
        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        adv = betas.detach().clone()
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = adv.new_zeros(())
            for i in range(adv.shape[0]):
                x, z = self._input_and_target(adv[i], std)
                pred = model(x[None])[0]
                # L1 in centimetres — the BMnet training objective.
                loss = loss + (torch.abs(pred - z) * m_std).mean()
            grad, = torch.autograd.grad(loss, adv)
            adv = (adv.detach() + lr * grad.detach()).clamp(
                -BETA_CLAMP, BETA_CLAMP)

        for p in model.parameters():
            p.requires_grad_(True)
        model.train(was_training)
        return adv.detach()

    def make_batch(self, model: torch.nn.Module, std: Standardizer,
                   n: int, *, sigma: float = 1.0,
                   steps: int = ABS_STEPS) -> tuple[torch.Tensor,
                                                    torch.Tensor]:
        """A full adversarial batch: seed → ascent → render the final
        shapes into ``(inputs, standardized targets)`` for BMnet."""
        adv = self.adversarial(model, std, self.seed_betas(n, sigma),
                               steps=steps)
        xs, zs = [], []
        with torch.no_grad():
            for i in range(adv.shape[0]):
                x, z = self._input_and_target(adv[i], std)
                xs.append(x)
                zs.append(z)
        return torch.stack(xs), torch.stack(zs)
