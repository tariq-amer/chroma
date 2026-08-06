"""Model an electricity bill as an integral of instantaneous power.

The physical quantity a meter senses is *power* P(t), in kilowatts, and it
changes continuously as appliances switch on and off.  Energy -- the thing you
are actually billed for -- is the area under that curve:

    E(t) = integral from 0 to t of P(s) ds          [kWh, when t is in hours]

Everything else in a real tariff is a weighting applied to that area.  Writing
the marginal price of the next kilowatt-hour as p(t, c), where t is the time
and c is how much you have consumed so far in the billing period, the whole
bill is one integral:

    Bill = F + integral from 0 to T of p(t, E(t)) * P(t) dt + R_d * D + tax

    F   fixed / standing charge for the period
    R_d demand charge rate, per kW of peak demand
    D   peak demand = max over t of (1/W) * integral from t to t+W of P(s) ds

The three tariff families in common use are just three choices of p:

    flat        p(t, c) = r                -> Bill = r * E(T)
    time-of-use p(t, c) = r(t)              -> integral of r(t) P(t) dt
    tiered      p(t, c) = m(c)              -> integral of m(c) dc, by the
                                               substitution u = E(t),
                                               du = P(t) dt

That last line is the interesting one: the tiered bill looks time-dependent
but the substitution collapses it to an integral over energy alone, which has
a closed form.  `demo_tiered_substitution` checks the two against each other.

No third-party dependencies; standard library only.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

# P(t): hours since the start of the billing period -> kilowatts
Profile = Callable[[float], float]

# p(t, c): (hours, kWh consumed so far) -> currency per kWh
PriceFn = Callable[[float, float], float]

HOURS_PER_DAY = 24.0


# ---------------------------------------------------------------------------
# Quadrature: turning "thinner rectangles" into an actual number
# ---------------------------------------------------------------------------


def riemann(f: Profile, a: float, b: float, n: int, rule: str = "mid") -> float:
    """Riemann sum with n rectangles -- the picture, made literal.

    rule is "left", "right", or "mid".  Left and right are first-order
    accurate (error ~ h), midpoint is second-order (error ~ h^2).
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    h = (b - a) / n
    offset = {"left": 0.0, "mid": 0.5, "right": 1.0}[rule]
    return h * math.fsum(f(a + (i + offset) * h) for i in range(n))


def trapezoid(f: Profile, a: float, b: float, n: int) -> float:
    """Trapezoid rule.  Error ~ h^2; exact for straight-line segments."""
    if n < 1:
        raise ValueError("n must be at least 1")
    h = (b - a) / n
    interior = math.fsum(f(a + i * h) for i in range(1, n))
    return h * (0.5 * f(a) + interior + 0.5 * f(b))


def simpson(f: Profile, a: float, b: float, n: int) -> float:
    """Simpson's rule.  Error ~ h^4 -- but only if f is smooth on [a, b].

    An appliance switching on is a jump discontinuity, and across a jump this
    degrades to no better than the trapezoid rule.  Split the interval at the
    switch times (see `integrate`'s breakpoints argument) to keep the fourth
    order where it matters.
    """
    if n < 2 or n % 2:
        raise ValueError("n must be even and at least 2")
    h = (b - a) / n
    odd = math.fsum(f(a + i * h) for i in range(1, n, 2))
    even = math.fsum(f(a + i * h) for i in range(2, n, 2))
    return h / 3.0 * (f(a) + 4.0 * odd + 2.0 * even + f(b))


def integrate(
    f: Profile,
    a: float,
    b: float,
    n: int = 2880,
    method: str = "simpson",
    breakpoints: Sequence[float] | None = None,
) -> float:
    """Integrate f over [a, b], optionally splitting at known discontinuities.

    breakpoints are times where the load jumps (an oven turning on, a tariff
    period changing).  Integrating each smooth piece separately is what keeps
    a high-order rule honest; without it the error is dominated by whichever
    rectangle straddles the jump.
    """
    if b <= a:
        return 0.0

    cuts = [a, b]
    if breakpoints:
        cuts.extend(t for t in breakpoints if a < t < b)
    cuts = sorted(set(cuts))

    total = 0.0
    span = b - a
    for lo, hi in zip(cuts, cuts[1:]):
        # Distribute the sample budget in proportion to each piece's width.
        sub_n = max(2, int(round(n * (hi - lo) / span)))
        g = _one_sided(f, lo, hi)
        if method == "simpson":
            if sub_n % 2:
                sub_n += 1
            total += simpson(g, lo, hi, sub_n)
        elif method == "trapezoid":
            total += trapezoid(g, lo, hi, sub_n)
        elif method in ("left", "right", "mid"):
            total += riemann(g, lo, hi, sub_n, method)
        else:
            raise ValueError(f"unknown method: {method!r}")
    return total


def _one_sided(f: Profile, lo: float, hi: float) -> Profile:
    """Evaluate f at the endpoints from strictly inside (lo, hi).

    A breakpoint is exactly where the load jumps, so f(lo) returns whichever
    side of the jump the author's comparison happened to pick -- and a closed
    rule like the trapezoid or Simpson's weights that endpoint, importing a
    value from the wrong segment.  The resulting error decays only like h,
    destroying the high order that splitting was meant to preserve.

    Nudging endpoint evaluations inward by a nanoscopic fraction of the
    segment takes the one-sided limit instead, which is the value the segment
    actually has.  For continuous f the change is far below rounding error.
    """
    eps = (hi - lo) * 1e-9

    def g(t: float) -> float:
        if t <= lo:
            return f(lo + eps)
        if t >= hi:
            return f(hi - eps)
        return f(t)

    return g


# ---------------------------------------------------------------------------
# Load profiles: what P(t) actually looks like in a house
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Appliance:
    """A load that runs at a fixed power during given windows of each day.

    windows are (start_hour, end_hour) pairs in local time; a window that
    wraps past midnight (e.g. an EV charging 23:00 -> 05:00) is allowed.
    days selects which weekdays it runs, Monday = 0; None means every day.
    """

    name: str
    power_kw: float
    windows: tuple[tuple[float, float], ...] = ()
    days: tuple[int, ...] | None = None

    def is_on(self, t: float) -> bool:
        if self.days is not None:
            if int(t // HOURS_PER_DAY) % 7 not in self.days:
                return False
        hour = t % HOURS_PER_DAY
        for start, end in self.windows:
            if start <= end:
                if start <= hour < end:
                    return True
            elif hour >= start or hour < end:  # wraps midnight
                return True
        return False

    def power_at(self, t: float) -> float:
        return self.power_kw if self.is_on(t) else 0.0

    def switch_times(self, horizon_h: float) -> list[float]:
        """Every instant this appliance jumps, over [0, horizon_h]."""
        out: list[float] = []
        days = int(math.ceil(horizon_h / HOURS_PER_DAY))
        for day in range(days):
            base = day * HOURS_PER_DAY
            for start, end in self.windows:
                for edge in (start, end):
                    t = base + edge
                    if 0.0 < t < horizon_h:
                        out.append(t)
        return out


@dataclass
class Household:
    """A whole-home load profile: always-on baseline + weather + appliances.

    baseline_kw is the floor you never escape -- fridge, router, standby
    power.  It looks negligible instant to instant, but it is the only load
    multiplied by the full 720 hours in the month, so it routinely outranks
    appliances that draw ten times as much (see `energy_by_source`).

    hvac_amplitude_kw and hvac_peak_hour add a smooth daily cycle peaking in
    the late afternoon, standing in for heating or cooling.
    """

    baseline_kw: float = 0.25
    hvac_amplitude_kw: float = 0.0
    hvac_peak_hour: float = 16.0
    appliances: list[Appliance] = field(default_factory=list)

    def power_at(self, t: float) -> float:
        p = self.baseline_kw
        if self.hvac_amplitude_kw:
            phase = 2.0 * math.pi * (t - self.hvac_peak_hour) / HOURS_PER_DAY
            # Raised cosine: never negative, mean = amplitude / 2.
            p += self.hvac_amplitude_kw * 0.5 * (1.0 + math.cos(phase))
        for appliance in self.appliances:
            p += appliance.power_at(t)
        return p

    def __call__(self, t: float) -> float:
        return self.power_at(t)

    def switch_times(self, horizon_h: float) -> list[float]:
        times: set[float] = set()
        for appliance in self.appliances:
            times.update(appliance.switch_times(horizon_h))
        return sorted(times)


# ---------------------------------------------------------------------------
# Price functions p(t, c)
# ---------------------------------------------------------------------------


def flat_rate(rate: float) -> PriceFn:
    """Every kWh costs the same."""
    return lambda t, c: rate


def time_of_use(periods: Sequence[tuple[float, float, float]], default: float) -> PriceFn:
    """Price depends on the hour of day.

    periods is a list of (start_hour, end_hour, rate); windows may wrap past
    midnight.  Any hour not covered falls back to `default`.
    """

    def price(t: float, c: float) -> float:
        hour = t % HOURS_PER_DAY
        for start, end, rate in periods:
            if start <= end:
                if start <= hour < end:
                    return rate
            elif hour >= start or hour < end:
                return rate
        return default

    return price


def tiered(tiers: Sequence[tuple[float, float]]) -> PriceFn:
    """Price depends on cumulative consumption this period.

    tiers is [(width_kwh, rate), ...] in ascending order: the first width_kwh
    cost the first rate, the next width_kwh the next rate, and so on.  Use
    math.inf for the width of the last, open-ended tier.

    The returned function is m(c) from the module docstring -- it ignores t.
    """
    if not tiers:
        raise ValueError("need at least one tier")

    edges: list[float] = []
    running = 0.0
    for width, _ in tiers[:-1]:
        running += width
        edges.append(running)
    rates = [rate for _, rate in tiers]

    def price(t: float, c: float) -> float:
        return rates[bisect_right(edges, c)]

    return price


def tiered_cost_closed_form(total_kwh: float, tiers: Sequence[tuple[float, float]]) -> float:
    """Evaluate integral from 0 to total_kwh of m(c) dc exactly.

    m is piecewise constant, so the integral is a finite sum of block areas.
    This is the substitution u = E(t) done by hand, and it is what
    `demo_tiered_substitution` compares the time-domain integral against.
    """
    remaining = total_kwh
    cost = 0.0
    for width, rate in tiers:
        if remaining <= 0.0:
            break
        used = min(remaining, width)
        cost += used * rate
        remaining -= used
    if remaining > 0.0:
        raise ValueError("tiers do not cover the total consumption; last width should be inf")
    return cost


def combine(*price_fns: PriceFn) -> PriceFn:
    """Add price components, e.g. a time-of-use energy rate plus a tier surcharge."""
    return lambda t, c: math.fsum(fn(t, c) for fn in price_fns)


# ---------------------------------------------------------------------------
# The bill
# ---------------------------------------------------------------------------


@dataclass
class Tariff:
    """A billing contract.  Rates are placeholders -- replace with your own."""

    marginal_price: PriceFn = field(default_factory=lambda: flat_rate(0.15))
    fixed_charge: float = 0.0  # per billing period
    demand_rate: float = 0.0  # per kW of peak demand
    demand_window_h: float = 0.25  # utilities typically average over 15 or 30 min
    tax_rate: float = 0.0  # fraction, applied to the subtotal
    currency: str = "$"


@dataclass
class Bill:
    energy_kwh: float
    energy_cost: float
    peak_demand_kw: float
    demand_cost: float
    fixed_charge: float
    tax: float
    total: float
    currency: str = "$"
    hourly_kwh: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        c = self.currency
        lines = [
            f"Energy consumed      {self.energy_kwh:12.2f} kWh",
            f"Energy charge        {c}{self.energy_cost:11.2f}",
        ]
        if self.demand_cost or self.peak_demand_kw:
            lines.append(f"Peak demand          {self.peak_demand_kw:12.2f} kW")
        if self.demand_cost:
            lines.append(f"Demand charge        {c}{self.demand_cost:11.2f}")
        if self.fixed_charge:
            lines.append(f"Fixed charge         {c}{self.fixed_charge:11.2f}")
        if self.tax:
            lines.append(f"Tax                  {c}{self.tax:11.2f}")
        lines.append(f"{'-' * 33}")
        lines.append(f"TOTAL                {c}{self.total:11.2f}")
        if self.energy_kwh:
            blended = self.total / self.energy_kwh
            lines.append(f"Blended rate         {c}{blended:11.4f} / kWh")
        return "\n".join(lines)


def compute_bill(
    profile: Profile,
    tariff: Tariff,
    days: float = 30.0,
    steps_per_hour: int = 60,
    breakpoints: Sequence[float] | None = None,
) -> Bill:
    """Integrate a load profile against a tariff over a billing period.

    Because the price can depend on cumulative energy E(t), the integrand is
    state-dependent and cannot be evaluated by a plain quadrature rule -- the
    integral has to be marched forward in time, carrying E(t) along.  Each
    step uses the trapezoid rule for the slice of energy and evaluates the
    price at the slice midpoint, which is second-order accurate overall.

    Passing `breakpoints` (appliance switch times, tariff period boundaries)
    forces grid points onto the discontinuities so no single step straddles a
    jump.

    Each slice samples the load just inside its own endpoints rather than at
    the shared grid points.  Sharing them would let a load that switches on at
    a boundary contribute half a slice of energy to the slice *before* it
    started -- which under time-of-use pricing is billed at the previous
    period's rate.  The error is small but systematic, and it lands exactly
    where tariffs change, so it is worth the extra evaluations.
    """
    horizon = days * HOURS_PER_DAY
    if horizon <= 0:
        raise ValueError("days must be positive")

    grid = _build_grid(horizon, steps_per_hour, breakpoints)

    cumulative_kwh = 0.0
    energy_cost = 0.0
    hourly: list[float] = []
    current_hour = 0
    hour_accumulator = 0.0

    for i in range(len(grid) - 1):
        t0, t1 = grid[i], grid[i + 1]
        dt = t1 - t0
        if dt <= 0.0:
            continue
        # Energy in this slice: trapezoid, exact for a load ramping linearly
        # and exact for a constant load, which covers most appliance segments.
        inset = dt * 1e-9
        slice_kwh = 0.5 * (profile(t0 + inset) + profile(t1 - inset)) * dt
        # Price at the midpoint of the slice, in both time and consumption.
        rate = tariff.marginal_price(0.5 * (t0 + t1), cumulative_kwh + 0.5 * slice_kwh)
        energy_cost += rate * slice_kwh
        cumulative_kwh += slice_kwh

        hour_accumulator += slice_kwh
        while t1 >= (current_hour + 1) - 1e-12 and current_hour < int(horizon):
            hourly.append(hour_accumulator)
            hour_accumulator = 0.0
            current_hour += 1

    # Always reported, since it is worth seeing; only charged if the rate > 0.
    peak_kw = peak_demand(profile, horizon, tariff.demand_window_h, steps_per_hour)
    demand_cost = tariff.demand_rate * peak_kw
    subtotal = energy_cost + demand_cost + tariff.fixed_charge
    tax = subtotal * tariff.tax_rate

    return Bill(
        energy_kwh=cumulative_kwh,
        energy_cost=energy_cost,
        peak_demand_kw=peak_kw,
        demand_cost=demand_cost,
        fixed_charge=tariff.fixed_charge,
        tax=tax,
        total=subtotal + tax,
        currency=tariff.currency,
        hourly_kwh=hourly,
    )


def _build_grid(
    horizon: float, steps_per_hour: int, breakpoints: Sequence[float] | None
) -> list[float]:
    """Uniform grid over [0, horizon] with breakpoints forced in."""
    n = max(1, int(round(horizon * steps_per_hour)))
    grid = [horizon * i / n for i in range(n + 1)]
    if breakpoints:
        grid.extend(t for t in breakpoints if 0.0 < t < horizon)
        grid = sorted(set(grid))
    return grid


def peak_demand(
    profile: Profile,
    horizon_h: float,
    window_h: float = 0.25,
    steps_per_hour: int = 60,
) -> float:
    """Largest window-averaged power over the period.

        D = max over t of (1 / W) * integral from t to t+W of P(s) ds

    This is why demand charges punish simultaneity rather than total use: two
    3 kW loads running back to back integrate to the same energy as running
    them together, but the second arrangement doubles D.
    """
    if window_h <= 0:
        raise ValueError("window_h must be positive")

    steps = max(1, int(round(window_h * steps_per_hour)))
    dt = window_h / steps
    total_steps = max(steps, int(round(horizon_h / dt)))

    samples = [profile(i * dt) for i in range(total_steps + 1)]

    # Rolling trapezoid sum over a fixed-width window.
    def window_energy(start: int) -> float:
        seg = samples[start : start + steps + 1]
        return dt * (0.5 * seg[0] + math.fsum(seg[1:-1]) + 0.5 * seg[-1])

    best = 0.0
    for start in range(0, len(samples) - steps):
        best = max(best, window_energy(start) / window_h)
    return best


# ---------------------------------------------------------------------------
# Real meter data
# ---------------------------------------------------------------------------


def energy_from_samples(times_h: Sequence[float], power_kw: Sequence[float]) -> float:
    """Integrate a smart-meter power series with the trapezoid rule.

    Sample spacing need not be uniform.  This is the function to use on a CSV
    exported from a utility portal or an energy monitor.
    """
    if len(times_h) != len(power_kw):
        raise ValueError("times and power must be the same length")
    if len(times_h) < 2:
        return 0.0
    return math.fsum(
        0.5 * (power_kw[i] + power_kw[i + 1]) * (times_h[i + 1] - times_h[i])
        for i in range(len(times_h) - 1)
    )


def power_from_meter_readings(
    times_h: Sequence[float], readings_kwh: Sequence[float]
) -> list[tuple[float, float]]:
    """Differentiate cumulative meter readings back into average power.

    The other half of the fundamental theorem: the dial on your meter *is*
    E(t), so P = dE/dt.  Returns (midpoint_time, average_kW) per interval.
    """
    if len(times_h) != len(readings_kwh):
        raise ValueError("times and readings must be the same length")
    out = []
    for i in range(len(times_h) - 1):
        dt = times_h[i + 1] - times_h[i]
        if dt <= 0:
            raise ValueError("times must be strictly increasing")
        out.append(
            (
                0.5 * (times_h[i] + times_h[i + 1]),
                (readings_kwh[i + 1] - readings_kwh[i]) / dt,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Marginal analysis: what does one more appliance-hour actually cost?
# ---------------------------------------------------------------------------


def marginal_cost_of_load(
    profile: Profile,
    tariff: Tariff,
    extra_kw: float,
    start_h: float,
    duration_h: float,
    days: float = 30.0,
    steps_per_hour: int = 60,
    breakpoints: Sequence[float] | None = None,
    repeat_daily: bool = True,
) -> float:
    """Cost of adding a constant load on top of an existing profile.

    With repeat_daily the load runs in the same window every day of the
    period; otherwise it runs once, starting at start_h.

    Under a flat rate the answer is just extra_kw * duration_h * r, and you do
    not need this function.  Under tiered pricing it is not, because the added
    energy pushes E(t) into higher bands, and under time-of-use it depends on
    *when* the window falls.  Either way the answer only comes from
    re-integrating the whole period -- which is cheap, so just do that.
    """
    base = compute_bill(profile, tariff, days, steps_per_hour, breakpoints)

    extra = Appliance("extra", extra_kw, ((start_h, start_h + duration_h),))
    horizon = days * HOURS_PER_DAY

    if repeat_daily:
        def with_extra(t: float) -> float:
            return profile(t) + extra.power_at(t)

        extra_breaks = list(breakpoints or []) + extra.switch_times(horizon)
    else:
        stop = start_h + duration_h

        def with_extra(t: float) -> float:
            return profile(t) + (extra_kw if start_h <= t < stop else 0.0)

        extra_breaks = list(breakpoints or []) + [start_h, stop]

    after = compute_bill(with_extra, tariff, days, steps_per_hour, extra_breaks)
    return after.total - base.total


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------


def example_household() -> Household:
    """A plausible all-electric home.  Edit the numbers to match yours."""
    return Household(
        baseline_kw=0.28,  # fridge, router, standby loads -- never off
        hvac_amplitude_kw=1.8,  # cooling load, peaking late afternoon
        hvac_peak_hour=16.0,
        appliances=[
            Appliance("kettle + coffee", 2.2, ((6.75, 7.0),)),
            Appliance("oven", 2.4, ((18.0, 19.0),)),
            Appliance("dishwasher", 1.8, ((20.0, 21.5),)),
            Appliance("washing machine", 2.0, ((10.0, 11.0),), days=(5, 6)),
            Appliance("EV charger", 7.2, ((1.0, 5.0),)),
            Appliance("lighting + TV", 0.45, ((18.0, 23.0),)),
        ],
    )


def example_tariffs() -> dict[str, Tariff]:
    """Three contracts over the same load, to compare by integration."""
    return {
        "flat": Tariff(
            marginal_price=flat_rate(0.16),
            fixed_charge=12.0,
            tax_rate=0.05,
        ),
        "time-of-use": Tariff(
            marginal_price=time_of_use(
                [
                    (0.0, 7.0, 0.09),  # overnight
                    (7.0, 17.0, 0.17),  # daytime
                    (17.0, 21.0, 0.34),  # evening peak
                    (21.0, 24.0, 0.17),
                ],
                default=0.17,
            ),
            fixed_charge=12.0,
            tax_rate=0.05,
        ),
        "tiered": Tariff(
            marginal_price=tiered([(300.0, 0.11), (300.0, 0.19), (math.inf, 0.28)]),
            fixed_charge=12.0,
            tax_rate=0.05,
        ),
    }


def _smooth_profile(t: float) -> float:
    """A baseline plus a daily cycle; smooth, with an antiderivative in closed form."""
    return 0.4 + 1.2 * 0.5 * (1.0 + math.cos(2.0 * math.pi * (t - 16.0) / 24.0))


def _smooth_exact(a: float, b: float) -> float:
    """Exact integral of _smooth_profile over [a, b], by hand."""
    w = 2.0 * math.pi / 24.0

    def antideriv(t: float) -> float:
        return 1.0 * t + 0.6 * math.sin(w * (t - 16.0)) / w

    return antideriv(b) - antideriv(a)


def demo_convergence() -> None:
    """The video's claim, quantified: how fast does the error shrink?"""
    # Integrate over a partial day (06:00 to 22:00).  See the note below for
    # why a *whole* day would be a misleading test.
    a, b = 6.0, 22.0
    exact = _smooth_exact(a, b)

    print(f"Smooth profile over {a:.0f}:00-{b:.0f}:00, exact = {exact:.9f} kWh")
    print()
    print(f"{'n':>6} {'left':>12} {'midpoint':>12} {'trapezoid':>12} {'Simpson':>12}")
    prev: tuple[float, ...] | None = None
    for n in (4, 8, 16, 32, 64, 128):
        errs = (
            abs(riemann(_smooth_profile, a, b, n, "left") - exact),
            abs(riemann(_smooth_profile, a, b, n, "mid") - exact),
            abs(trapezoid(_smooth_profile, a, b, n) - exact),
            abs(simpson(_smooth_profile, a, b, n) - exact),
        )
        print(f"{n:>6} " + " ".join(f"{e:>12.3e}" for e in errs))
        if prev is not None:
            ratios = " ".join(
                f"{(p / e if e > 1e-14 else float('inf')):>12.1f}" for p, e in zip(prev, errs)
            )
            print(f"{'  x':>6} " + ratios)
        prev = errs
    print()
    print("The 'x' rows are error ratios from halving the step.  They settle at")
    print("about 2, 4, 4 and 16 -- first order for the left rule, second for")
    print("midpoint and trapezoid, fourth for Simpson.  That is exactly the")
    print("payoff for thinner rectangles, and it is why nobody sums rectangles")
    print("by hand: going from n=4 to n=128 with Simpson buys ten digits.")
    print()

    whole_day = abs(trapezoid(_smooth_profile, 0.0, 24.0, 6) - _smooth_exact(0.0, 24.0))
    print(f"Aside: over a WHOLE day the trapezoid rule with just n=6 errs by")
    print(f"{whole_day:.2e} kWh -- essentially exact.  A periodic load integrated")
    print("over a whole number of periods makes the Euler-Maclaurin correction")
    print("terms cancel at the endpoints.  Useful to know, but it flatters the")
    print("method: monthly bills do not start and end at matching load states.")
    print()

    # Same exercise across a discontinuity: an oven switching on.  The switch
    # times are deliberately off-grid -- 18:24 and 19:18 -- because a jump that
    # happens to land on a sample point costs nothing.
    on, off = 18.4, 19.3

    def square(t: float) -> float:
        return 0.4 + (2.4 if on <= t < off else 0.0)

    exact_sq = 0.4 * 24.0 + 2.4 * (off - on)
    print(f"Now a jump discontinuity (oven on {on:.2f}h-{off:.2f}h), exact = {exact_sq:.4f} kWh")
    print(f"{'n':>6} {'Simpson (naive)':>18} {'Simpson (split)':>18}")
    for n in (24, 48, 96, 192, 384):
        naive = abs(simpson(square, 0, 24, n) - exact_sq)
        split = abs(integrate(square, 0, 24, n, "simpson", breakpoints=[on, off]) - exact_sq)
        print(f"{n:>6} {naive:>18.3e} {split:>18.3e}")
    print()
    print("Simpson's fourth order evaporates at the jump -- the naive column")
    print("crawls, roughly halving, because the error is dominated by the one")
    print("rectangle straddling the switch.  Splitting the integral there makes")
    print("it exact at any n, since each piece is constant.  This is the single")
    print("most useful practical trick here: integrate between switch times.")


def demo_tiered_substitution() -> None:
    """Check integral of m(E(t)) P(t) dt == integral of m(u) du numerically."""
    home = example_household()
    tiers = [(300.0, 0.11), (300.0, 0.19), (math.inf, 0.28)]
    tariff = Tariff(marginal_price=tiered(tiers))
    breaks = home.switch_times(30 * HOURS_PER_DAY)

    bill = compute_bill(home, tariff, days=30, steps_per_hour=120, breakpoints=breaks)
    closed = tiered_cost_closed_form(bill.energy_kwh, tiers)

    print("Tiered tariff, one month")
    print(f"  total energy                        {bill.energy_kwh:10.3f} kWh")
    print(f"  time-domain integral  m(E(t)) P dt  {bill.energy_cost:10.4f}")
    print(f"  substitution           m(u) du      {closed:10.4f}")
    print(f"  relative difference                 {abs(bill.energy_cost - closed) / closed:10.2e}")
    print()
    print("They agree because u = E(t), du = P(t) dt is a valid substitution:")
    print("P(t) >= 0 always, so E is nondecreasing and the change of variable")
    print("is well defined.  A house that exported solar power would break this")
    print("-- P would go negative, E would retrace, and the tier you are in")
    print("would depend on the path, not just the total.")


def demo_compare_tariffs() -> None:
    home = example_household()
    breaks = home.switch_times(30 * HOURS_PER_DAY)
    for name, tariff in example_tariffs().items():
        bill = compute_bill(home, tariff, days=30, steps_per_hour=120, breakpoints=breaks)
        print(f"--- {name} " + "-" * (28 - len(name)))
        print(bill)
        print()


def demo_load_shifting() -> None:
    """Same energy, different times: what time-of-use pricing really charges for."""
    tou = example_tariffs()["time-of-use"]
    base = Household(baseline_kw=0.28, hvac_amplitude_kw=1.8)
    breaks = base.switch_times(30 * HOURS_PER_DAY)

    print("A 7.2 kW EV charger, 4 h every night for 30 days, under the")
    print("time-of-use tariff -- billed by when you plug it in:")
    print()
    costs = {}
    for start in (1.0, 9.0, 14.0, 18.0):
        costs[start] = marginal_cost_of_load(
            base, tou, extra_kw=7.2, start_h=start, duration_h=4.0,
            days=30, steps_per_hour=120, breakpoints=breaks, repeat_daily=True,
        )
        print(f"  start {start:>5.1f}h   monthly cost {tou.currency}{costs[start]:8.2f}")
    print()
    cheapest, dearest = min(costs.values()), max(costs.values())
    print(f"Identical energy every time: 7.2 kW x 4 h x 30 = {7.2 * 4 * 30:.0f} kWh.")
    print("The integral of P dt does not change; the integral of r(t) P(t) dt")
    print(f"does, by {tou.currency}{dearest - cheapest:.2f} a month "
          f"({dearest / cheapest:.1f}x) for the same charge.")


def energy_by_source(home: Household, days: float = 30.0) -> list[tuple[str, float, float]]:
    """Monthly kWh per load, as (name, peak_kw, kWh), largest first.

    Each row is its own integral over the period.  Because power adds, the
    integral is linear -- the sum of the rows is the whole-home total, with no
    interaction term to worry about.
    """
    horizon = days * HOURS_PER_DAY
    rows = [("baseline (always on)", home.baseline_kw, home.baseline_kw * horizon)]

    if home.hvac_amplitude_kw:
        def hvac(t: float) -> float:
            phase = 2.0 * math.pi * (t - home.hvac_peak_hour) / HOURS_PER_DAY
            return home.hvac_amplitude_kw * 0.5 * (1.0 + math.cos(phase))

        rows.append(("heating / cooling", home.hvac_amplitude_kw,
                     integrate(hvac, 0.0, horizon, n=4000, method="simpson")))

    for appliance in home.appliances:
        kwh = integrate(
            appliance.power_at, 0.0, horizon, n=4000, method="trapezoid",
            breakpoints=appliance.switch_times(horizon),
        )
        rows.append((appliance.name, appliance.power_kw, kwh))

    return sorted(rows, key=lambda r: r[2], reverse=True)


def demo_baseline_share() -> None:
    """Which loads actually own the bill -- power x time, not power."""
    home = example_household()
    tariff = example_tariffs()["flat"]
    breaks = home.switch_times(30 * HOURS_PER_DAY)
    full = compute_bill(home, tariff, days=30, steps_per_hour=120, breakpoints=breaks)
    rate = full.energy_cost / full.energy_kwh

    rows = energy_by_source(home, days=30)
    print(f"{'load':<24}{'peak kW':>9}{'kWh/mo':>10}{'share':>8}{'cost':>10}")
    for name, kw, kwh in rows:
        print(f"{name:<24}{kw:>9.2f}{kwh:>10.1f}{kwh / full.energy_kwh:>8.1%}"
              f"{tariff.currency + format(kwh * rate, '.2f'):>10}")
    total = math.fsum(kwh for _, _, kwh in rows)
    print(f"{'-' * 61}")
    print(f"{'sum of parts':<24}{'':>9}{total:>10.1f}")
    print(f"{'whole-home integral':<24}{'':>9}{full.energy_kwh:>10.1f}")
    print()
    print("The two totals match because integration is linear: the area under")
    print("a sum of loads is the sum of their areas.")
    print()
    print("Note the ordering.  The kettle draws 2.2 kW -- nearly as much as the")
    print("oven -- but runs 15 minutes, so it lands near the bottom.  Cost is")
    print("the integral, not the nameplate: a load's bill is power TIMES time.")
    print()
    standby = 0.05 * 30 * HOURS_PER_DAY
    oven_kwh = next(kwh for name, _, kwh in rows if name == "oven")
    print(f"Concretely: shaving 50 W of standby saves {standby:.1f} kWh/mo, versus")
    print(f"{oven_kwh:.1f} kWh/mo for the oven.  Half the oven's contribution, from")
    print("a load too small to notice -- because it never switches off.")


def demo_hourly_shape() -> None:
    """ASCII plot of one day's energy, hour by hour."""
    home = example_household()
    tariff = example_tariffs()["flat"]
    breaks = home.switch_times(HOURS_PER_DAY)
    bill = compute_bill(home, tariff, days=1, steps_per_hour=240, breakpoints=breaks)

    peak = max(bill.hourly_kwh) if bill.hourly_kwh else 1.0
    print("One day, energy per hour (each hour is one rectangle of the sum):")
    print()
    for hour, kwh in enumerate(bill.hourly_kwh):
        bar = "#" * int(round(40 * kwh / peak))
        print(f"  {hour:02d}:00 {kwh:6.2f} kWh |{bar}")
    print()
    print(f"  Total {math.fsum(bill.hourly_kwh):6.2f} kWh  <- the area under P(t)")


def main() -> None:
    sections = [
        ("CONVERGENCE: thinner rectangles, quantified", demo_convergence),
        ("THE DAILY LOAD CURVE", demo_hourly_shape),
        ("COMPARING TARIFFS OVER ONE MONTH", demo_compare_tariffs),
        ("TIERED PRICING BY SUBSTITUTION", demo_tiered_substitution),
        ("LOAD SHIFTING UNDER TIME-OF-USE", demo_load_shifting),
        ("THE COST OF DOING NOTHING", demo_baseline_share),
    ]
    for title, fn in sections:
        print()
        print("=" * 70)
        print(title)
        print("=" * 70)
        fn()


if __name__ == "__main__":
    main()
