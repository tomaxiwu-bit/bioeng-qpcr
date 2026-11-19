"""Score legends and negative score labels must stay outside the data area."""

import matplotlib.pyplot as plt
import numpy as np

from bioeng_qpcr.docking.scoring import ScoreBreakdown
from bioeng_qpcr.docking.search import Pose
from bioeng_qpcr.docking.structure import RigidTransform
from bioeng_qpcr.plotting.docking import plot_pose_ranking


def test_ranking_legend_sits_above_axes_and_labels_fit():
    breakdown = ScoreBreakdown(-6, -1, -2, -8, 1, -16)
    pose = Pose(
        RigidTransform(np.eye(3), np.zeros(3)), -16, breakdown, np.zeros((2, 3))
    )
    figure, ax = plot_pose_ranking([pose])
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axes = ax.get_window_extent(renderer)
        legend = ax.get_legend().get_window_extent(renderer)
        assert legend.y0 > axes.y1
        assert figure.bbox.contains(legend.x0, legend.y0)
        assert figure.bbox.contains(legend.x1, legend.y1)
        for label in ax.texts:
            box = label.get_window_extent(renderer)
            assert axes.x0 <= box.x0 < box.x1 <= axes.x1
        assert "arbitrary units" in ax.get_xlabel()
    finally:
        plt.close(figure)
