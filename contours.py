import numpy as np
from matplotlib import pyplot as plt
from unyt import (
    K,
    astronomical_unit,
    Solar_Radius,
    degree_celsius,
)

# Script set-up constants and bounds
contour_resolution = 1000

physical_bounds = {
    'star_temperature': (3e3 * K, 1e4 * K),
    'orbit_semi_major_axis': (1e-2 * astronomical_unit, 10 * astronomical_unit),
    'star_radius': (0.1 * Solar_Radius, 1e3 * Solar_Radius),
    'planet_albedo': (0, 1),
}

# Set the plotting style: use MNRAS template
try:
    plt.style.use('mnras.mplstyle')
except (FileNotFoundError, OSError, NotADirectoryError):
    print('Could not find the mnras.mplstyle style-sheet.')


def equilibrium_temperature(
        star_temperature,
        star_radius,
        orbit_semi_major_axis,
        planet_albedo
):
    """
    Computes the equilibrium temperature of a planet assuming
    Stefan-Boltzmann law for black bodies, no green-house effects
    and no internal energy sources (e.g. tidal heating,
    radioactivity etc).
    :param star_temperature: [Kelvin]
    :param star_radius: [meters or R_Sun]
    :param orbit_semi_major_axis: [meters or AU]
    :param planet_albedo: [dimensionless]
    :return: [Kelvin]
    """
    planet_temperature = star_temperature * np.sqrt(star_radius / orbit_semi_major_axis / 2) * (
            1 - planet_albedo) ** (1 / 4)
    return planet_temperature.to(K)


def draw_radius_contours(
        axes,
        star_radius,
        planet_albedo,
        levels=[1],
        color='green',
        use_labels=True
):
    # Make the norm object to define the image stretch
    x_bins, y_bins = np.meshgrid(
        np.linspace(*physical_bounds['orbit_semi_major_axis'], contour_resolution),
        np.linspace(*physical_bounds['star_temperature'], contour_resolution)
    )
    planet_temperature = equilibrium_temperature(
        star_temperature=y_bins.flatten(),
        orbit_semi_major_axis=x_bins.flatten(),
        star_radius=star_radius,
        planet_albedo=planet_albedo
    )
    planet_temperature = planet_temperature.reshape(x_bins.shape)
    planet_temperature = planet_temperature.to(degree_celsius)

    contours = axes.contour(
        x_bins,
        y_bins,
        planet_temperature,
        levels,
        colors=color,
        linewidths=0.5,
        alpha=1,
        zorder=1000
    )

    if use_labels:
        fmt = {level: f'{level:.0f} ${degree_celsius.latex_repr}$' for level in levels}

        # Place labels closest to middle of the figure
        xmin, xmax, ymin, ymax = axes.axis()
        mid = (xmin + xmax) / 2, (ymin + ymax) / 2

        label_pos = []
        i = 0
        for line in contours.collections:
            for path in line.get_paths():
                logvert = path.vertices
                i += 1

                # find closest point
                logdist = np.linalg.norm(logvert - mid, ord=2, axis=1)
                min_ind = np.argmin(logdist)
                label_pos.append(logvert[min_ind, :])

        # Draw contour labels
        axes.clabel(
            contours,
            inline=True,
            inline_spacing=3,
            rightside_up=True,
            colors=color,
            fontsize=7,
            fmt=fmt,
            manual=label_pos
        )


# Example implementation
fig = plt.figure(constrained_layout=True)
axes = fig.subplots()
# axes.loglog()
draw_radius_contours(
    axes,
    1 * Solar_Radius,
    0.3,
    levels=[-100, 0, 100],
    color='blue',
    use_labels=True
)
axes.scatter([1], [5735], label='Earth-Sun')
axes.set_xlabel('Semi-major axis [AU]')
axes.set_ylabel('Star effective temperature [K]')
axes.legend()
plt.savefig('example.png')
plt.show()
