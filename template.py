import numpy as np
import matplotlib.pyplot as plt

class SmartIrrigation:
    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):
        self.a=a
        self.b=b
        self.dt=dt
        self.t = np.arange(0, t_max , dt)
        self.t_max=t_max

    def u_step(self):
        return np.ones_like(self.t)
    def u_ramp(self):
        return 0.1 * self.t
    def u_sin(self):
        return np.sin(0.5 * self.t)
    def u_exponential(self): 
        return 1 - np.exp(-0.3 * self.t)
    def u_pulse(self):
        pulse = np.ones_like(self.t)
        pulse[self.t >= 5] = 0.0 
        return pulse

    def laplace_transform(self, f, s):
        integrand = f * np.exp(-s * self.t)
        return np.trapezoid(integrand, self.t)

    def inverse_laplace(self, s_list, F_s_values):
        N = len(s_list)
        omega = np.imag(s_list)
        W = np.max(omega)
        dw = (2 * W) / (N - 1)

        c = np.real(s_list[0])

        f = np.zeros_like(self.t)

        for i, t_val in enumerate(self.t):
            summation = 0

            for k in range(N):
                weight = 1
                if k == 0 or k == N - 1:
                    weight = 0.5

                summation += weight * F_s_values[k] * np.exp(1j * omega[k] * t_val)

            f[i] = np.exp(c * t_val) * (dw / (2 * np.pi)) * np.real(summation)

        return f

    def H_s(self, s, U_s):
        return (self.b / (s + self.a)) * U_s
    def steady_state(self, h):
        """Mean of last 5% of signal."""
        last_part = h[int(0.95 * len(h)):]
        return np.mean(last_part)

    def time_constant(self, h):
        """Time to first reach 63.2% of steady-state."""
        h_ss = self.steady_state(h)
        target = 0.632 * h_ss

        for i, val in enumerate(h):
            if val >= target:
                return self.t[i]
        return None

    def rise_time(self, h):
        """Time to go from 10% to 90% of steady-state."""
        h_ss = self.steady_state(h)
        low = 0.1 * h_ss
        high = 0.9 * h_ss

        t1, t2 = None, None

        for i, val in enumerate(h):
            if t1 is None and val >= low:
                t1 = self.t[i]
            if t2 is None and val >= high:
                t2 = self.t[i]

        return t2 - t1 if t1 and t2 else None

    def settling_time(self, h):
        """Time after which h(t) stays permanently within ±2% of h_ss."""
        h_ss = self.steady_state(h)
        tol = 0.02 * h_ss

        for i in range(len(h)):
            if np.all(np.abs(h[i:] - h_ss) <= tol):
                return self.t[i]
        return None

    def overshoot(self, h):
        """Percentage overshoot: (h_max - h_ss) / h_ss * 100."""
        h_ss = self.steady_state(h)
        h_max = np.max(h)

        if h_ss == 0:
            return 0.0

        overshoot = (h_max - h_ss) / h_ss * 100
        return max(0.0, overshoot)

    def compute_metrics(self, h):
       
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
        }

    def euler_simulate(self, u):
        """
        Euler method for dh/dt = -a*h(t) + b*u(t)
        h[n+1] = h[n] + dt * (-a*h[n] + b*u[n])
        """
        h = np.zeros_like(self.t)
        for n in range(len(self.t) - 1):
            dhdt = -self.a * h[n] + self.b * u[n]
            h[n + 1] = h[n] + self.dt * dhdt
        return h


#Change values of a, b to experiment with different system dynamics
system = SmartIrrigation(a=0.5, b=1.0, t_max=20, dt=0.01)

inputs = {
    "Step Input":        system.u_step(),
    "Ramp Input":        system.u_ramp(),
    "Sinusoidal Input":  system.u_sin(),
    "Exponential Input": system.u_exponential(),
    "Pulse Input":       system.u_pulse(),
}

# Bromwich contour parameters, set these values
c = 0.51
W = 100
N = 1000
omega = np.linspace(-W, W, N)
s_list = c + 1j * omega

colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

for idx, (name, u) in enumerate(inputs.items()):
    print(f"Processing: {name}...")

    # --- Laplace --- set these values
    U_s_vals = np.array([system.laplace_transform(u, s) for s in s_list])
    H_s_vals = np.array([system.H_s(s, U_s_vals[i]) for i, s in enumerate(s_list)])

    h_laplace = system.inverse_laplace(s_list, H_s_vals)
    print(f"\n  -> {name}")
    metrics = system.compute_metrics(h_laplace)
    for k, v in metrics.items():
        print(f"      {k.replace('_',' ').title():<22}: {v}")

    # --- Euler ---
    h_euler = system.euler_simulate(u)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Smart Irrigation — {name}", fontsize=13, fontweight='bold')

    # Laplace subplot
    axes[0].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
    axes[0].plot(system.t, h_laplace, color=colors[idx], lw=2.2, label="Output h(t)")
    axes[0].set_title("Laplace Transform Simulation", fontweight='bold')
    axes[0].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_ylabel("Water Level / Input", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Euler subplot
    axes[1].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
    axes[1].plot(system.t, h_euler, color='tomato', lw=2.2, label="Output h(t)")
    axes[1].set_title("Euler Method Simulation", fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].set_ylabel("Water Level / Input", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()