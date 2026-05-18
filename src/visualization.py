import numpy as np
import matplotlib.pyplot as plt


def overlay_mask(ax, mri_slice, mask_slice, mask_cmap="autumn", alpha=0.55):
    ax.imshow(mri_slice.T, cmap="gray", origin="lower")
    masked = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
    ax.imshow(masked, cmap=mask_cmap, alpha=alpha, origin="lower")
    ax.axis("off")


def build_triptych_figure(mri_data, ut_data, em_data, ut_slices, em_slices):
    common_slices = []
    if ut_slices and em_slices:
        common_slices = sorted(set(ut_slices["all"]).intersection(set(em_slices["all"])))

    if common_slices:
        selected_slice = common_slices[len(common_slices) // 2]
    else:
        ut_slice = ut_slices["middle"] if ut_slices else None
        em_slice = em_slices["middle"] if em_slices else None
        selected_slice = em_slice if em_slice is not None else ut_slice

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4.8),
        facecolor="#0b1020"
    )

    for ax in axes:
        ax.set_facecolor("#0b1020")
        ax.axis("off")

    # MRI original
    axes[0].imshow(
        mri_data[:, :, selected_slice].T,
        cmap="gray",
        origin="lower"
    )

    # Útero
    overlay_mask(
        axes[1],
        mri_data[:, :, selected_slice],
        ut_data[:, :, selected_slice],
        mask_cmap="spring"
    )

    # Endometrioma
    overlay_mask(
        axes[2],
        mri_data[:, :, selected_slice],
        em_data[:, :, selected_slice],
        mask_cmap="autumn"
    )

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=0.05,
        wspace=0.04
    )

    return fig