"""Tests for the integral electricity-bill model.

Every test checks a numerical result against something known independently:
a closed-form integral, a hand computation, or an algebraic identity that has
to hold regardless of the quadrature used.

Run with:  python3 -m unittest test_bill_model -v
"""

from __future__ import annotations

import math
import unittest

from bill_model import (
    Appliance,
    Household,
    Tariff,
    combine,
    compute_bill,
    energy_by_source,
    energy_from_samples,
    flat_rate,
    integrate,
    marginal_cost_of_load,
    peak_demand,
    power_from_meter_readings,
    riemann,
    simpson,
    tiered,
    tiered_cost_closed_form,
    time_of_use,
    trapezoid,
)

HOURS = 24.0


class TestQuadrature(unittest.TestCase):
    def test_constant_load_is_exact(self):
        """A constant P integrates to P * T under every rule."""
        for rule in ("left", "mid", "right"):
            self.assertAlmostEqual(riemann(lambda t: 2.5, 0, 24, 13, rule), 60.0, places=10)
        self.assertAlmostEqual(trapezoid(lambda t: 2.5, 0, 24, 13), 60.0, places=10)
        self.assertAlmostEqual(simpson(lambda t: 2.5, 0, 24, 14), 60.0, places=10)

    def test_linear_ramp_exact_for_trapezoid(self):
        """The trapezoid rule is exact on straight lines, at any n."""
        for n in (1, 2, 7, 100):
            self.assertAlmostEqual(trapezoid(lambda t: 0.5 * t, 0, 10, n), 25.0, places=10)

    def test_simpson_exact_on_cubics(self):
        """Simpson's rule integrates cubics exactly -- one order beyond design."""
        exact = 4.0**4 / 4.0
        self.assertAlmostEqual(simpson(lambda t: t**3, 0, 4, 2), exact, places=10)

    def test_convergence_orders(self):
        """Error ratios under step halving reveal each rule's order."""
        f = math.sin
        a, b = 0.0, 2.0
        exact = math.cos(a) - math.cos(b)

        def err(fn, n):
            return abs(fn(n) - exact)

        # Compare n=64 against n=128; ratios approach 2, 4, 4, 16.
        left = err(lambda n: riemann(f, a, b, n, "left"), 64) / err(
            lambda n: riemann(f, a, b, n, "left"), 128
        )
        mid = err(lambda n: riemann(f, a, b, n, "mid"), 64) / err(
            lambda n: riemann(f, a, b, n, "mid"), 128
        )
        trap = err(lambda n: trapezoid(f, a, b, n), 64) / err(
            lambda n: trapezoid(f, a, b, n), 128
        )
        simp = err(lambda n: simpson(f, a, b, n), 32) / err(lambda n: simpson(f, a, b, n), 64)

        self.assertAlmostEqual(left, 2.0, delta=0.05)
        self.assertAlmostEqual(mid, 4.0, delta=0.1)
        self.assertAlmostEqual(trap, 4.0, delta=0.1)
        self.assertAlmostEqual(simp, 16.0, delta=1.0)

    def test_breakpoints_make_square_waves_exact(self):
        """Splitting at switch times integrates a piecewise-constant load exactly."""
        on, off = 18.4, 19.3

        def square(t):
            return 0.4 + (2.4 if on <= t < off else 0.0)

        exact = 0.4 * 24.0 + 2.4 * (off - on)
        for n in (24, 96, 385):
            got = integrate(square, 0, 24, n, "simpson", breakpoints=[on, off])
            self.assertAlmostEqual(got, exact, places=9, msg=f"n={n}")

    def test_naive_simpson_is_worse_across_a_jump(self):
        """Without breakpoints the same integral carries real error."""
        on, off = 18.4, 19.3

        def square(t):
            return 0.4 + (2.4 if on <= t < off else 0.0)

        exact = 0.4 * 24.0 + 2.4 * (off - on)
        naive = abs(simpson(square, 0, 24, 96) - exact)
        split = abs(integrate(square, 0, 24, 96, "simpson", breakpoints=[on, off]) - exact)
        self.assertGreater(naive, 1e-3)
        self.assertLess(split, 1e-9)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            simpson(lambda t: 1.0, 0, 1, 3)  # odd n
        with self.assertRaises(ValueError):
            riemann(lambda t: 1.0, 0, 1, 0)
        with self.assertRaises(ValueError):
            integrate(lambda t: 1.0, 0, 1, 10, method="gauss")

    def test_empty_and_reversed_interval(self):
        self.assertEqual(integrate(lambda t: 1.0, 5.0, 5.0), 0.0)
        self.assertEqual(integrate(lambda t: 1.0, 5.0, 1.0), 0.0)


class TestAppliance(unittest.TestCase):
    def test_simple_window(self):
        oven = Appliance("oven", 2.4, ((18.0, 19.0),))
        self.assertEqual(oven.power_at(18.5), 2.4)
        self.assertEqual(oven.power_at(17.9), 0.0)
        self.assertEqual(oven.power_at(19.0), 0.0)  # half-open
        self.assertEqual(oven.power_at(24 + 18.5), 2.4)  # repeats daily

    def test_window_wrapping_midnight(self):
        ev = Appliance("ev", 7.0, ((23.0, 2.0),))
        for hour in (23.0, 23.5, 0.0, 1.9):
            self.assertEqual(ev.power_at(hour), 7.0, msg=f"hour={hour}")
        for hour in (2.0, 12.0, 22.9):
            self.assertEqual(ev.power_at(hour), 0.0, msg=f"hour={hour}")

    def test_day_restriction(self):
        """days=(5, 6) means weekends only, with t=0 on a Monday."""
        washer = Appliance("washer", 2.0, ((10.0, 11.0),), days=(5, 6))
        self.assertEqual(washer.power_at(10.5), 0.0)  # Monday
        self.assertEqual(washer.power_at(5 * 24 + 10.5), 2.0)  # Saturday
        self.assertEqual(washer.power_at(6 * 24 + 10.5), 2.0)  # Sunday
        self.assertEqual(washer.power_at(7 * 24 + 10.5), 0.0)  # next Monday

    def test_energy_matches_power_times_time(self):
        oven = Appliance("oven", 2.4, ((18.0, 19.0),))
        horizon = 7 * 24.0
        kwh = integrate(
            oven.power_at, 0, horizon, 2000, "trapezoid",
            breakpoints=oven.switch_times(horizon),
        )
        self.assertAlmostEqual(kwh, 2.4 * 1.0 * 7, places=8)


class TestPriceFunctions(unittest.TestCase):
    def test_flat_rate(self):
        p = flat_rate(0.15)
        self.assertEqual(p(0.0, 0.0), 0.15)
        self.assertEqual(p(500.0, 9999.0), 0.15)

    def test_time_of_use_selects_by_hour(self):
        p = time_of_use([(0.0, 8.0, 0.10), (8.0, 24.0, 0.20)], default=0.99)
        self.assertEqual(p(0.0, 0.0), 0.10)
        self.assertEqual(p(7.99, 0.0), 0.10)
        self.assertEqual(p(8.0, 0.0), 0.20)
        self.assertEqual(p(24.0 + 3.0, 0.0), 0.10)  # next day, same shape

    def test_time_of_use_wraps_midnight(self):
        p = time_of_use([(22.0, 6.0, 0.08)], default=0.20)
        self.assertEqual(p(23.0, 0.0), 0.08)
        self.assertEqual(p(2.0, 0.0), 0.08)
        self.assertEqual(p(12.0, 0.0), 0.20)

    def test_tiered_selects_by_cumulative_energy(self):
        p = tiered([(300.0, 0.11), (300.0, 0.19), (math.inf, 0.28)])
        self.assertEqual(p(0.0, 0.0), 0.11)
        self.assertEqual(p(0.0, 299.9), 0.11)
        self.assertEqual(p(0.0, 300.1), 0.19)
        self.assertEqual(p(0.0, 600.1), 0.28)
        self.assertEqual(p(0.0, 1e9), 0.28)

    def test_tiered_closed_form_by_hand(self):
        tiers = [(300.0, 0.11), (300.0, 0.19), (math.inf, 0.28)]
        self.assertAlmostEqual(tiered_cost_closed_form(0.0, tiers), 0.0)
        self.assertAlmostEqual(tiered_cost_closed_form(100.0, tiers), 11.0)
        self.assertAlmostEqual(tiered_cost_closed_form(300.0, tiers), 33.0)
        self.assertAlmostEqual(tiered_cost_closed_form(450.0, tiers), 33.0 + 150 * 0.19)
        self.assertAlmostEqual(
            tiered_cost_closed_form(700.0, tiers), 33.0 + 57.0 + 100 * 0.28
        )

    def test_tiered_requires_unbounded_last_tier(self):
        with self.assertRaises(ValueError):
            tiered_cost_closed_form(500.0, [(100.0, 0.1)])
        with self.assertRaises(ValueError):
            tiered([])

    def test_combine_adds_components(self):
        p = combine(flat_rate(0.10), tiered([(100.0, 0.0), (math.inf, 0.05)]))
        self.assertAlmostEqual(p(0.0, 50.0), 0.10)
        self.assertAlmostEqual(p(0.0, 150.0), 0.15)


class TestBill(unittest.TestCase):
    def test_flat_bill_matches_hand_arithmetic(self):
        """1 kW held for 30 days = 720 kWh; the rest is simple arithmetic."""
        tariff = Tariff(marginal_price=flat_rate(0.20), fixed_charge=10.0, tax_rate=0.10)
        bill = compute_bill(lambda t: 1.0, tariff, days=30, steps_per_hour=4)

        self.assertAlmostEqual(bill.energy_kwh, 720.0, places=6)
        self.assertAlmostEqual(bill.energy_cost, 144.0, places=6)
        self.assertAlmostEqual(bill.fixed_charge, 10.0, places=6)
        self.assertAlmostEqual(bill.tax, 15.4, places=6)
        self.assertAlmostEqual(bill.total, 169.4, places=6)

    def test_time_of_use_bill_by_hand(self):
        """Constant 1 kW: 8 h at 0.10 + 16 h at 0.20 = 4.00 per day."""
        tariff = Tariff(
            marginal_price=time_of_use([(0.0, 8.0, 0.10), (8.0, 24.0, 0.20)], 0.20)
        )
        bill = compute_bill(lambda t: 1.0, tariff, days=1, steps_per_hour=60)
        self.assertAlmostEqual(bill.energy_cost, 4.0, places=6)

    def test_tiered_bill_by_hand(self):
        """Constant 1 kW for a day = 24 kWh: 10 at 0.10, then 14 at 0.20."""
        tariff = Tariff(marginal_price=tiered([(10.0, 0.10), (math.inf, 0.20)]))
        bill = compute_bill(lambda t: 1.0, tariff, days=1, steps_per_hour=600)
        self.assertAlmostEqual(bill.energy_cost, 1.0 + 2.8, places=3)

    def test_tiered_substitution_identity(self):
        """integral of m(E(t)) P(t) dt must equal integral of m(u) du."""
        home = Household(
            baseline_kw=0.3,
            hvac_amplitude_kw=1.5,
            appliances=[Appliance("oven", 2.4, ((18.0, 19.0),))],
        )
        tiers = [(300.0, 0.11), (300.0, 0.19), (math.inf, 0.28)]
        tariff = Tariff(marginal_price=tiered(tiers))
        breaks = home.switch_times(30 * HOURS)

        bill = compute_bill(home, tariff, days=30, steps_per_hour=240, breakpoints=breaks)
        closed = tiered_cost_closed_form(bill.energy_kwh, tiers)
        self.assertAlmostEqual(bill.energy_cost / closed, 1.0, places=5)

    def test_tiered_error_shrinks_with_step_size(self):
        """The only tiered error is the slice straddling a tier edge: O(h).

        The edge is deliberately at 10.3 kWh, which a 1 kW load crosses at
        t = 10.3 h -- not a multiple of any of the step sizes used here.  An
        edge that happens to land on a slice boundary is billed exactly and
        would hide the effect entirely.
        """
        tiers = [(10.3, 0.10), (math.inf, 0.20)]
        tariff = Tariff(marginal_price=tiered(tiers))
        exact = 10.3 * 0.10 + (24.0 - 10.3) * 0.20

        # Only the one slice containing the crossing is mispriced, and it is
        # priced by its midpoint.  If the crossing sits a fraction phi into
        # that slice, the mispriced width is min(phi, 1 - phi), so
        #     error = (rate jump) * h * min(phi, 1 - phi)
        # This is O(h), but NOT monotone in the step size: phi jumps around as
        # the grid moves, and s=16 and s=32 below happen to tie exactly.
        for s in (8, 16, 32, 64, 128):
            h = 1.0 / s
            phi = (10.3 * s) % 1.0
            predicted = 0.10 * h * min(phi, 1.0 - phi)
            got = abs(
                compute_bill(lambda t: 1.0, tariff, 1, steps_per_hour=s).energy_cost - exact
            )
            self.assertAlmostEqual(got, predicted, places=10, msg=f"steps_per_hour={s}")
            self.assertLessEqual(got, 0.05 * h + 1e-12)  # worst case, phi = 1/2

    def test_load_starting_on_a_tariff_boundary_is_billed_at_the_new_rate(self):
        """Regression: a load switching on exactly when the peak period starts.

        Sampling the profile at shared grid points let the trapezoid rule
        credit half a slice of the load to the slice before it started, which
        is billed at the *previous* period's rate.  The whole 4 h session sits
        inside the peak window, so the exact answer is entirely at 0.34.
        """
        tariff = Tariff(
            marginal_price=time_of_use([(7.0, 17.0, 0.17), (17.0, 21.0, 0.34)], 0.09)
        )
        ev = Appliance("ev", 7.2, ((17.0, 21.0),))
        bill = compute_bill(
            ev.power_at, tariff, days=1, steps_per_hour=120,
            breakpoints=ev.switch_times(HOURS),
        )
        self.assertAlmostEqual(bill.energy_kwh, 7.2 * 4.0, places=6)
        self.assertAlmostEqual(bill.energy_cost, 7.2 * 4.0 * 0.34, places=6)

    def test_square_wave_energy_is_exact(self):
        """With breakpoints supplied, a switched load integrates exactly."""
        oven = Appliance("oven", 2.4, ((18.4, 19.3),))
        bill = compute_bill(
            oven.power_at, Tariff(), days=7, steps_per_hour=60,
            breakpoints=oven.switch_times(7 * HOURS),
        )
        self.assertAlmostEqual(bill.energy_kwh, 2.4 * 0.9 * 7, places=8)

    def test_energy_is_independent_of_tariff(self):
        """Rates weight the area; they do not change it."""
        home = Household(baseline_kw=0.3, hvac_amplitude_kw=1.2)
        breaks = home.switch_times(30 * HOURS)
        totals = [
            compute_bill(home, Tariff(marginal_price=p), 30, 120, breaks).energy_kwh
            for p in (
                flat_rate(0.16),
                time_of_use([(0.0, 7.0, 0.09), (7.0, 24.0, 0.30)], 0.20),
                tiered([(300.0, 0.11), (math.inf, 0.28)]),
            )
        ]
        for value in totals[1:]:
            self.assertAlmostEqual(value, totals[0], places=8)

    def test_hourly_breakdown_sums_to_total(self):
        home = Household(baseline_kw=0.3, hvac_amplitude_kw=1.2)
        bill = compute_bill(home, Tariff(), days=1, steps_per_hour=120)
        self.assertEqual(len(bill.hourly_kwh), 24)
        self.assertAlmostEqual(math.fsum(bill.hourly_kwh), bill.energy_kwh, places=8)

    def test_rejects_nonpositive_period(self):
        with self.assertRaises(ValueError):
            compute_bill(lambda t: 1.0, Tariff(), days=0)


class TestPeakDemand(unittest.TestCase):
    def test_constant_load(self):
        self.assertAlmostEqual(peak_demand(lambda t: 2.0, 24.0, 0.25), 2.0, places=6)

    def test_sustained_pulse_reaches_full_power(self):
        """A pulse longer than the averaging window registers at full power."""

        def pulse(t):
            return 6.0 if 10.0 <= t < 11.0 else 0.5

        self.assertAlmostEqual(peak_demand(pulse, 24.0, 0.25, 240), 6.0, places=3)

    def test_short_spike_is_averaged_down(self):
        """A spike shorter than the window cannot reach full power -- that is
        the entire point of averaging demand over 15 minutes."""

        def spike(t):
            return 12.0 if 10.0 <= t < 10.05 else 0.0  # 3 minutes at 12 kW

        demand = peak_demand(spike, 24.0, 0.25, 1200)
        self.assertAlmostEqual(demand, 12.0 * 0.05 / 0.25, delta=0.05)
        self.assertLess(demand, 12.0)

    def test_simultaneity_costs_more_than_energy(self):
        """Same energy, different peak: two loads together vs. back to back."""

        def together(t):
            return 6.0 if 10.0 <= t < 11.0 else 0.0

        def sequential(t):
            return 3.0 if 10.0 <= t < 12.0 else 0.0

        e1 = integrate(together, 0, 24, 2000, "trapezoid", breakpoints=[10, 11])
        e2 = integrate(sequential, 0, 24, 2000, "trapezoid", breakpoints=[10, 12])
        self.assertAlmostEqual(e1, e2, places=6)  # identical energy
        self.assertGreater(peak_demand(together, 24, 0.25, 240),
                           peak_demand(sequential, 24, 0.25, 240) * 1.9)

    def test_rejects_bad_window(self):
        with self.assertRaises(ValueError):
            peak_demand(lambda t: 1.0, 24.0, window_h=0.0)


class TestMeterData(unittest.TestCase):
    def test_uniform_samples_match_analytic_integral(self):
        times = [i * 0.25 for i in range(97)]  # 15-minute data for a day
        power = [1.0 + math.sin(2 * math.pi * t / 24.0) for t in times]
        got = energy_from_samples(times, power)
        exact = 24.0  # the sine integrates to zero over a full period
        self.assertAlmostEqual(got, exact, places=4)

    def test_irregular_spacing(self):
        """Real exports have gaps; the trapezoid rule does not care."""
        times = [0.0, 0.5, 3.0, 3.1, 10.0]
        power = [2.0, 2.0, 2.0, 2.0, 2.0]
        self.assertAlmostEqual(energy_from_samples(times, power), 20.0, places=9)

    def test_degenerate_inputs(self):
        self.assertEqual(energy_from_samples([], []), 0.0)
        self.assertEqual(energy_from_samples([1.0], [2.0]), 0.0)
        with self.assertRaises(ValueError):
            energy_from_samples([0.0, 1.0], [1.0])

    def test_differentiating_readings_recovers_power(self):
        """P = dE/dt: the meter dial is the antiderivative of power."""
        times = [float(i) for i in range(25)]
        readings = [1.5 * t for t in times]  # a steady 1.5 kW draw
        for _, kw in power_from_meter_readings(times, readings):
            self.assertAlmostEqual(kw, 1.5, places=9)

    def test_round_trip_integrate_then_differentiate(self):
        def profile(t):
            return 0.5 + 0.4 * math.sin(2 * math.pi * t / 24.0)

        times = [i * 0.5 for i in range(49)]
        readings = [integrate(profile, 0.0, t, 400, "simpson") if t else 0.0 for t in times]
        for mid, kw in power_from_meter_readings(times, readings):
            self.assertAlmostEqual(kw, profile(mid), places=3)

    def test_readings_must_increase_in_time(self):
        with self.assertRaises(ValueError):
            power_from_meter_readings([0.0, 0.0], [0.0, 1.0])


class TestAnalysis(unittest.TestCase):
    def test_marginal_cost_under_flat_rate_is_power_times_time(self):
        """With a flat rate the integral collapses to arithmetic -- a good check
        that the re-integration machinery agrees with the obvious answer."""
        tariff = Tariff(marginal_price=flat_rate(0.20))
        cost = marginal_cost_of_load(
            lambda t: 0.5, tariff, extra_kw=2.0, start_h=13.0, duration_h=3.0,
            days=30, steps_per_hour=120, repeat_daily=True,
        )
        self.assertAlmostEqual(cost, 2.0 * 3.0 * 30 * 0.20, places=4)

    def test_marginal_cost_depends_on_timing_under_tou(self):
        tariff = Tariff(
            marginal_price=time_of_use(
                [(0.0, 7.0, 0.09), (7.0, 17.0, 0.17), (17.0, 21.0, 0.34)], 0.17
            )
        )
        kwargs = dict(extra_kw=7.2, duration_h=4.0, days=30, steps_per_hour=120)
        overnight = marginal_cost_of_load(lambda t: 0.3, tariff, start_h=1.0, **kwargs)
        evening = marginal_cost_of_load(lambda t: 0.3, tariff, start_h=17.0, **kwargs)

        self.assertAlmostEqual(overnight, 7.2 * 4 * 30 * 0.09, places=3)
        self.assertAlmostEqual(evening, 7.2 * 4 * 30 * 0.34, places=3)
        self.assertGreater(evening, overnight * 3.5)

    def test_marginal_cost_single_session(self):
        tariff = Tariff(marginal_price=flat_rate(0.20))
        cost = marginal_cost_of_load(
            lambda t: 0.5, tariff, extra_kw=2.0, start_h=13.0, duration_h=3.0,
            days=30, steps_per_hour=120, repeat_daily=False,
        )
        self.assertAlmostEqual(cost, 2.0 * 3.0 * 0.20, places=4)

    def test_energy_by_source_is_linear(self):
        """Integration is linear: the parts must sum to the whole."""
        home = Household(
            baseline_kw=0.28,
            hvac_amplitude_kw=1.8,
            appliances=[
                Appliance("oven", 2.4, ((18.0, 19.0),)),
                Appliance("ev", 7.2, ((1.0, 5.0),)),
                Appliance("washer", 2.0, ((10.0, 11.0),), days=(5, 6)),
            ],
        )
        rows = energy_by_source(home, days=30)
        parts = math.fsum(kwh for _, _, kwh in rows)
        whole = compute_bill(
            home, Tariff(), days=30, steps_per_hour=240,
            breakpoints=home.switch_times(30 * HOURS),
        ).energy_kwh
        self.assertAlmostEqual(parts, whole, places=4)

    def test_energy_by_source_is_sorted(self):
        rows = energy_by_source(
            Household(
                baseline_kw=0.28,
                appliances=[
                    Appliance("kettle", 2.2, ((6.75, 7.0),)),
                    Appliance("ev", 7.2, ((1.0, 5.0),)),
                ],
            ),
            days=30,
        )
        self.assertEqual([name for name, _, _ in rows][0], "ev")
        self.assertEqual(rows, sorted(rows, key=lambda r: r[2], reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
