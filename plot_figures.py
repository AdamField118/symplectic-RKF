"""Draw Bernard's plots from a saved trajectory.

    python plot_figures.py --kerr KERR.npy --schwarzschild SCHW.npy --outdir figures/glsl

Each .npy is either what integrate_geodesic() returns, shape (steps+1, 8), or
what the GLSL binary writes, shape (n_geodesics, steps+1, 8), in which case
the first geodesic is plotted. You can pass just one of the two.

For each metric you get:

  kerr_geodesic.png           Appendix A as printed, all 15000 steps.
  fig7a_kerr.png              Figure 7(a) from the paper. It's the same run cut
                              off after 7823 steps, which is what lines up
                              with the published image.
  schwarzschild_geodesic.png  The Figure 6 setup (a = 0, r0 = 15), all steps.
  fig6a_schwarzschild.png     Figure 6(a): the first 5220 steps, viewed from
                              the angle Bernard dragged the plot to before
                              saving it (estimated from the published image).
  *_vs_paper.png              Ours, the published one, and an overlay where
                              red is only ours, blue only theirs, black both.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "paper_figures")

FIG7A_STEPS = 7823
FIG6A_STEPS = 5220
FIG6A_VIEW = dict(elev=29.23, azim=131.01, roll=5.7)


# Graph the geodesic trajectory
# (Appendix A's graph_geodesic. The title and camera angle are arguments here,
# and it saves to a file instead of calling plt.show().)
def graph_geodesic(geodesic, title, out, view=None):
    # Singling out each component of the trajectory as their own vectors
    r = geodesic[:, 1]
    theta = geodesic[:, 2]
    phi = geodesic[:, 3]

    # Coordinate transformations between spherical and cartesian
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)

    # Setting up the graph labels and title
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot(X, Y, Z, 'r', linewidth=1.0, color = 'purple')
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title)
    if view is not None:
        ax.view_init(**view)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _rgb(path):
    a = mpimg.imread(path)
    if a.dtype != np.float32:
        a = a.astype(np.float32) / 255
    return a[..., :3]


def compare(ours_png, paper_png, out):
    """Save ours, theirs and an overlay side by side, and count the curve pixels that differ."""
    a, b = _rgb(ours_png), _rgb(paper_png)

    def purple(x):
        return (x[..., 0] - x[..., 1] > 0.15) & (x[..., 2] - x[..., 1] > 0.15)

    pa, pb = purple(a), purple(b)
    over = np.ones_like(a)
    over[pa] = [1, 0, 0]
    over[pb] = [0, 0, 1]
    over[pa & pb] = [0, 0, 0]
    plt.imsave(out, np.concatenate([a, b, over], axis=1))
    differ = int(((np.abs(a - b).max(-1) > 0.1) & (pa | pb)).sum())
    print(f"wrote {out}  (curve pixels differing from the paper: {differ})")
    return differ


def load(path):
    g = np.load(path)
    return g[0] if g.ndim == 3 else g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kerr")
    p.add_argument("--schwarzschild")
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    o = lambda name: os.path.join(args.outdir, name)  # noqa: E731

    if args.kerr:
        g = load(args.kerr)
        graph_geodesic(g, "Kerr Geodesic", o("kerr_geodesic.png"))
        if len(g) > FIG7A_STEPS:
            graph_geodesic(g[: FIG7A_STEPS + 1], "Kerr Geodesic", o("fig7a_kerr.png"))
            compare(o("fig7a_kerr.png"), os.path.join(PAPER, "fig7a_kerr_bernard.png"), o("fig7a_kerr_vs_paper.png"))
    if args.schwarzschild:
        g = load(args.schwarzschild)
        graph_geodesic(g, "Schwarzschild Geodesic", o("schwarzschild_geodesic.png"))
        if len(g) > FIG6A_STEPS:
            graph_geodesic(g[: FIG6A_STEPS + 1], "Schwarzschild Geodesic", o("fig6a_schwarzschild.png"), view=FIG6A_VIEW)
            compare(o("fig6a_schwarzschild.png"), os.path.join(PAPER, "fig6a_schwarzschild_bernard.png"),
                    o("fig6a_schwarzschild_vs_paper.png"))


if __name__ == "__main__":
    main()
