import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Arc, Circle
from matplotlib.widgets import Button, Slider



L = 2.5
m = 35
g = 9.81
w0 = np.sqrt(g / L)

t_max = 12
n_points = 4800
t = np.linspace(0, t_max, n_points)
dt = t[1] - t[0]

theta0_init = np.pi / 6
omega0_init = 0.0


def pendulum_derivative(state):
    theta, omega = state
    return np.array([omega, -(g / L) * np.sin(theta)])


def solve_nonlinear_rk4(theta0, omega0=0.0):
    states = np.zeros((len(t), 2))
    states[0] = [theta0, omega0]

    for i in range(len(t) - 1):
        y = states[i]
        k1 = pendulum_derivative(y)
        k2 = pendulum_derivative(y + 0.5 * dt * k1)
        k3 = pendulum_derivative(y + 0.5 * dt * k2)
        k4 = pendulum_derivative(y + dt * k3)
        states[i + 1] = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return states[:, 0], states[:, 1]


def solve_nonlinear_euler(theta0, omega0=0.0):
    theta = np.zeros(len(t))
    omega = np.zeros(len(t))
    theta[0] = theta0
    omega[0] = omega0

    for i in range(len(t) - 1):
        theta[i + 1] = theta[i] + dt * omega[i]
        omega[i + 1] = omega[i] - dt * (g / L) * np.sin(theta[i])

    return theta, omega


def solve_small_angle(theta0, omega0=0.0):
    theta = theta0 * np.cos(w0 * t) + (omega0 / w0) * np.sin(w0 * t)
    omega = -theta0 * w0 * np.sin(w0 * t) + omega0 * np.cos(w0 * t)
    return theta, omega


def energies(theta, omega, small_angle=False):
    kinetic = 0.5 * m * (L * omega) ** 2
    if small_angle:
        potential = 0.5 * m * g * L * theta**2
    else:
        potential = m * g * L * (1 - np.cos(theta))
    return kinetic, potential


def compute_curves(theta0):
    theta_rk4, omega_rk4 = solve_nonlinear_rk4(theta0, omega0_init)
    theta_euler, omega_euler = solve_nonlinear_euler(theta0, omega0_init)
    theta_lin, omega_lin = solve_small_angle(theta0, omega0_init)

    ec_rk4, ep_rk4 = energies(theta_rk4, omega_rk4, small_angle=False)
    ec_euler, ep_euler = energies(theta_euler, omega_euler, small_angle=False)
    ec_lin, ep_lin = energies(theta_lin, omega_lin, small_angle=True)

    return {
        "theta_rk4": theta_rk4,
        "omega_rk4": omega_rk4,
        "theta_euler": theta_euler,
        "omega_euler": omega_euler,
        "theta_lin": theta_lin,
        "omega_lin": omega_lin,
        "ec_rk4": ec_rk4,
        "ep_rk4": ep_rk4,
        "ec_euler": ec_euler,
        "ep_euler": ep_euler,
        "ec_lin": ec_lin,
        "ep_lin": ep_lin,
    }


curves = compute_curves(theta0_init)


plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(13, 7.5))
fig.canvas.manager.set_window_title("Balancoire - comparaison petits angles / RK4 / Euler")
grid = fig.add_gridspec(
    2,
    2,
    width_ratios=[0.9, 1.6],
    height_ratios=[1, 1],
    left=0.07,
    right=0.97,
    top=0.9,
    bottom=0.18,
    hspace=0.35,
    wspace=0.28,
)

ax_anim = fig.add_subplot(grid[:, 0])
ax_ec = fig.add_subplot(grid[0, 1])
ax_ep = fig.add_subplot(grid[1, 1])

fig.suptitle(
    "Mouvement pendulaire d'une balancoire : petits angles vs RK4 vs Euler",
    fontsize=15,
    fontweight="bold",
)


ax_anim.set_aspect("equal")
ax_anim.set_xlim(-L * 1.25, L * 1.25)
ax_anim.set_ylim(-L * 1.18, L * 0.35)
ax_anim.axis("off")
ax_anim.set_title("Representation physique", fontweight="bold")

support_y = 0.0
seat_width = 0.6
seat_height = 0.08

#ax_anim.plot([-L * 0.9, L * 0.9], [support_y, support_y], color="#2f3542", lw=5)
ax_anim.plot([-L * 0.85, 0], [-L * 1.05, support_y], color="#57606f", lw=3)
ax_anim.plot([L * 0.85, 0], [-L * 1.05, support_y], color="#57606f", lw=3)
ax_anim.add_patch(Circle((0, 0), 0.06, color="#1e272e", zorder=4))

rope_lin, = ax_anim.plot([], [], color="#d63031", lw=2, alpha=0.25, zorder=1)
bob_lin, = ax_anim.plot([], [], "o", color="#d63031", markersize=15, alpha=0.22, zorder=2)
rope, = ax_anim.plot([], [], color="#2f3542", lw=3, zorder=5)
bob, = ax_anim.plot(
    [],
    [],
    "o",
    color="#0984e3",
    markersize=18,
    markeredgecolor="white",
    markeredgewidth=2,
    zorder=6,
)
shadow, = ax_anim.plot([], [], color="#ced6e0", lw=8, alpha=0.35, solid_capstyle="round", zorder=0)
velocity_vector = ax_anim.quiver(
    [0],
    [0],
    [0],
    [0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="#00b894",
    width=0.008,
    zorder=7,
)
acceleration_vector = ax_anim.quiver(
    [0],
    [0],
    [0],
    [0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="#e84393",
    width=0.008,
    zorder=7,
)
velocity_label = ax_anim.text(0, 0, "v", color="#00b894", fontsize=11, fontweight="bold", zorder=8)
acceleration_label = ax_anim.text(0, 0, "a", color="#e84393", fontsize=11, fontweight="bold", zorder=8)
theta_text = ax_anim.text(
    0,
    -L * 1.25,
    "",
    ha="center",
    va="center",
    fontsize=11,
    color="#2f3542",
)


# Courbes d'energie
colors = {
    "theory": "#d63031",
    "rk4": "#0984e3",
    "euler": "#e67e22",
    "marker": "#2d3436",
}

line_ec_lin, = ax_ec.plot(
    t,
    curves["ec_lin"],
    color=colors["theory"],
    lw=2,
    label="Petits angles",
)
line_ec_rk4, = ax_ec.plot(
    t,
    curves["ec_rk4"],
    color=colors["rk4"],
    lw=2,
    ls="--",
    label="RK4",
)
line_ec_euler, = ax_ec.plot(
    t,
    curves["ec_euler"],
    color=colors["euler"],
    lw=2,
    ls=":",
    label="Euler",
)
ec_marker_lin, = ax_ec.plot([], [], "o", color=colors["theory"], ms=6)
ec_marker_rk4, = ax_ec.plot([], [], "o", color=colors["rk4"], ms=6)
ec_marker_euler, = ax_ec.plot([], [], "o", color=colors["euler"], ms=6)

line_ep_lin, = ax_ep.plot(
    t,
    curves["ep_lin"],
    color=colors["theory"],
    lw=2,
    label="Petits angles",
)
line_ep_rk4, = ax_ep.plot(
    t,
    curves["ep_rk4"],
    color=colors["rk4"],
    lw=2,
    ls="--",
    label="RK4",
)
line_ep_euler, = ax_ep.plot(
    t,
    curves["ep_euler"],
    color=colors["euler"],
    lw=2,
    ls=":",
    label="Euler",
)
ep_marker_lin, = ax_ep.plot([], [], "o", color=colors["theory"], ms=6)
ep_marker_rk4, = ax_ep.plot([], [], "o", color=colors["rk4"], ms=6)
ep_marker_euler, = ax_ep.plot([], [], "o", color=colors["euler"], ms=6)

for ax, ylabel, title in [
    (ax_ec, "Energie cinetique Ec (J)", "Graphique 1 - Energie cinetique"),
    (ax_ep, "Energie potentielle Ep (J)", "Graphique 2 - Energie potentielle"),
]:
    ax.set_xlim(0, t_max)
    ax.set_xlabel("Temps t (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.25)


def set_energy_limits():
    max_energy = max(
        np.max(curves["ec_lin"]),
        np.max(curves["ec_rk4"]),
        np.max(curves["ec_euler"]),
        np.max(curves["ep_lin"]),
        np.max(curves["ep_rk4"]),
        np.max(curves["ep_euler"]),
        1,
    )
    ax_ec.set_ylim(-0.04 * max_energy, 1.12 * max_energy)
    ax_ep.set_ylim(-0.04 * max_energy, 1.12 * max_energy)


def draw_swing(frame_index):
    theta = curves["theta_rk4"][frame_index]
    omega = curves["omega_rk4"][frame_index]
    theta_lin = curves["theta_lin"][frame_index]
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    x_lin = L * np.sin(theta_lin)
    y_lin = -L * np.cos(theta_lin)

    tangent = np.array([np.cos(theta), np.sin(theta)])
    velocity_length = np.clip(0.28 * omega, -0.65, 0.65)
    acceleration_length = np.clip(-0.16 * g * np.sin(theta), -0.65, 0.65)
    velocity = velocity_length * tangent
    acceleration = acceleration_length * tangent

    rope_lin.set_data([0, x_lin], [0, y_lin])
    bob_lin.set_data([x_lin], [y_lin + 0.1])
    rope.set_data([0, x], [0, y])
    bob.set_data([x], [y + 0.1])
    shadow.set_data([x - seat_width * 0.55, x + seat_width * 0.55], [-L * 1.1, -L * 1.1])
    velocity_vector.set_offsets([[x, y + 0.1]])
    velocity_vector.set_UVC([velocity[0]], [velocity[1]])
    acceleration_vector.set_offsets([[x, y - 0.08]])
    acceleration_vector.set_UVC([acceleration[0]], [acceleration[1]])
    velocity_label.set_position((x + velocity[0] + 0.08, y + 0.1 + velocity[1] + 0.04))
    acceleration_label.set_position((x + acceleration[0] + 0.08, y - 0.08 + acceleration[1] - 0.06))
    theta_text.set_text(f"theta(0) = {theta_slider.val:.2f} rad    theta(t) = {theta:.2f} rad")


def draw_markers(frame_index):
    current_t = t[frame_index]
    ec_marker_lin.set_data([current_t], [curves["ec_lin"][frame_index]])
    ec_marker_rk4.set_data([current_t], [curves["ec_rk4"][frame_index]])
    ec_marker_euler.set_data([current_t], [curves["ec_euler"][frame_index]])
    ep_marker_lin.set_data([current_t], [curves["ep_lin"][frame_index]])
    ep_marker_rk4.set_data([current_t], [curves["ep_rk4"][frame_index]])
    ep_marker_euler.set_data([current_t], [curves["ep_euler"][frame_index]])


def refresh_curves(theta0):
    global curves
    curves = compute_curves(theta0)

    line_ec_lin.set_ydata(curves["ec_lin"])
    line_ec_rk4.set_ydata(curves["ec_rk4"])
    line_ec_euler.set_ydata(curves["ec_euler"])
    line_ep_lin.set_ydata(curves["ep_lin"])
    line_ep_rk4.set_ydata(curves["ep_rk4"])
    line_ep_euler.set_ydata(curves["ep_euler"])
    set_energy_limits()
    draw_swing(0)
    draw_markers(0)
    fig.canvas.draw_idle()


def on_slider_change(value):
    refresh_curves(value)


def reset(_event):
    theta_slider.reset()


slider_ax = fig.add_axes([0.18, 0.08, 0.60, 0.035])
theta_slider = Slider(
    ax=slider_ax,
    label="Angle initial theta(0) en rad",
    valmin=0.05,
    valmax=np.pi / 2,
    valinit=theta0_init,
    valstep=0.01,
    color="#0984e3",
)
theta_slider.on_changed(on_slider_change)

button_ax = fig.add_axes([0.83, 0.072, 0.1, 0.05])
reset_button = Button(button_ax, "Reset", color="#f1f2f6", hovercolor="#dfe4ea")
reset_button.on_clicked(reset)

set_energy_limits()


def animate(frame_index):
    draw_swing(frame_index)
    draw_markers(frame_index)
    return (
        rope_lin,
        bob_lin,
        rope,
        bob,
        shadow,
        velocity_vector,
        acceleration_vector,
        velocity_label,
        acceleration_label,
        theta_text,
        ec_marker_lin,
        ec_marker_rk4,
        ec_marker_euler,
        ep_marker_lin,
        ep_marker_rk4,
        ep_marker_euler,
    )


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=np.arange(0, len(t), 4),
    interval=60,
    blit=False,
    repeat=True,
)

plt.show()
