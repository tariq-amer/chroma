# Calculating an electricity bill with integrals

A working model of a household electricity bill, built as one integral. Pure
Python standard library — no dependencies.

```
python3 bill_model.py              # run all the demonstrations
python3 -m unittest test_bill_model -v   # 45 tests
```

## The idea

Your meter does not measure energy. It measures **power** `P(t)` — kilowatts,
right now — and that number moves every time a compressor cycles or a kettle
boils. What you are billed for is **energy**, the accumulation of power over
time, which is the area under that curve:

$$E(t) = \int_0^t P(s)\,ds \qquad \text{kWh, for } t \text{ in hours}$$

This is why the unit is a kilowatt-*hour*: it is literally kilowatts
multiplied by hours, and multiplication only works directly when `P` is
constant. When it varies — and it always varies — the product becomes an
integral.

The relationship runs both ways. The dial on your meter *is* `E(t)`, so power
is its derivative:

$$P(t) = \frac{dE}{dt}$$

That is the fundamental theorem of calculus, sitting in a box on the side of
your house. `energy_from_samples` goes one way (meter power readings → kWh),
`power_from_meter_readings` goes the other (cumulative dial readings → average
kW per interval).

## The whole bill, in one equation

Let `p(t, c)` be the price of the next kilowatt-hour, where `t` is the time
and `c` is how much you have already used this billing period. Then:

$$\text{Bill} = \underbrace{F}_{\text{standing charge}} + \underbrace{\int_0^T p\big(t, E(t)\big)\,P(t)\,dt}_{\text{energy charge}} + \underbrace{R_d \cdot \max_t \frac{1}{W}\int_t^{t+W} P(s)\,ds}_{\text{demand charge}}$$

Every tariff you will be offered is a choice of `p`:

| Tariff | Price function | Integral becomes |
|---|---|---|
| Flat rate | `p = r` | `r · E(T)` — the integral collapses to multiplication |
| Time-of-use | `p = r(t)` | `∫ r(t) P(t) dt` — weight each moment by its rate |
| Tiered / block | `p = m(c)` | `∫ m(E(t)) P(t) dt` — see below |

### Tiered pricing is a substitution problem

The tiered case looks the worst: the price depends on `E(t)`, which is itself
the integral you are computing. But substitute `u = E(t)`, so `du = P(t) dt`:

$$\int_0^T m\big(E(t)\big) P(t)\,dt = \int_0^{E(T)} m(u)\,du$$

The time dependence vanishes. Since `m` is piecewise constant, the right-hand
side is just a sum of block areas — a closed form, computed by
`tiered_cost_closed_form`. **Only the monthly total matters, not when you used
it.** `demo_tiered_substitution` verifies the two agree to ~1e-6.

This substitution needs `E` to be nondecreasing, which holds because `P ≥ 0`.
A house with rooftop solar exporting to the grid breaks it: `P` goes negative,
`E` retraces, and the tier you end up in genuinely depends on the *path*, not
just the total. That is a real billing dispute, not a mathematical curiosity.

### Demand charges integrate over a sliding window

$$D = \max_t \frac{1}{W}\int_t^{t+W} P(s)\,ds$$

Utilities average over 15 or 30 minutes rather than taking the instantaneous
peak, which is why a 3-minute 12 kW spike registers as only 2.4 kW on a
15-minute window (`test_short_spike_is_averaged_down`). Two 3 kW loads run
back-to-back and run simultaneously consume identical energy, but the second
arrangement doubles `D` — demand charges price *simultaneity*, which the plain
energy integral is blind to.

## Computing it: the rectangles from the video, quantified

The video's claim — thinner rectangles, more accurate — is right, and how
*much* more accurate depends on the rule. Halving the step size divides the
error by:

| Rule | Error | Factor per halving |
|---|---|---|
| Left/right Riemann | O(h) | 2 |
| Midpoint | O(h²) | 4 |
| Trapezoid | O(h²) | 4 |
| Simpson | O(h⁴) | 16 |

`demo_convergence` prints the measured ratios; they land on 2.0, 4.0, 4.0 and
16.0. Going from 4 intervals to 128 buys Simpson about ten digits.

### Two traps worth knowing

**1. Discontinuities destroy the high-order rules.** All of the above assumes
a smooth curve. An oven switching on is a jump, and across a jump Simpson's
rule is no better than first order — the error is dominated by the single
rectangle straddling the switch. The fix is to integrate *between* switch
times, which `integrate(..., breakpoints=[...])` does:

```
     n    Simpson (naive)    Simpson (split)
    24          1.040e+00          1.776e-15
   384          6.000e-02          1.776e-15
```

Naive Simpson merely halves its error; splitting is exact at any `n`, because
each piece is constant. This is the single most useful practical trick here —
appliance load profiles are square waves, not smooth curves.

**2. At a breakpoint, evaluate from strictly inside.** `P(18.4)` returns
whichever side of the jump your comparison happened to pick, so a closed rule
weighting that endpoint imports a value from the wrong segment, and the O(h)
error creeps straight back. `_one_sided` nudges endpoint evaluations inward by
a nanoscopic fraction of the segment to take the correct one-sided limit.

The same bug bit `compute_bill` in a more expensive way: sampling the profile
at shared grid points credited half a slice of a load to the slice *before* it
switched on. When a load starts exactly as the peak period opens — an EV
charger at 17:00 — that half slice got billed at the off-peak rate. Small, but
systematic, and it lands precisely where tariffs change
(`test_load_starting_on_a_tariff_boundary_is_billed_at_the_new_rate`).

### One error that does not shrink smoothly

Under tiered pricing the only mispriced slice is the one where cumulative
consumption crosses a tier edge. If the crossing sits a fraction φ into that
slice:

$$\text{error} = \Delta r \cdot h \cdot \min(\varphi,\, 1-\varphi)$$

This is O(h) but *not monotone* in step size — φ jumps around as the grid
moves, and in `test_tiered_error_shrinks_with_step_size` the errors at 16 and
32 steps/hour tie exactly. The test asserts the formula, which holds to ten
decimal places, rather than an order of convergence that isn't there.

## Using it on your own bill

```python
from bill_model import Appliance, Household, Tariff, compute_bill, time_of_use

home = Household(
    baseline_kw=0.28,           # fridge, router, standby — never off
    hvac_amplitude_kw=1.8,      # daily heating/cooling cycle
    hvac_peak_hour=16.0,
    appliances=[
        Appliance("oven", 2.4, ((18.0, 19.0),)),
        Appliance("EV charger", 7.2, ((1.0, 5.0),)),
        Appliance("washing machine", 2.0, ((10.0, 11.0),), days=(5, 6)),  # weekends
    ],
)

tariff = Tariff(
    marginal_price=time_of_use([
        (0.0, 7.0, 0.09),    # overnight
        (7.0, 17.0, 0.17),
        (17.0, 21.0, 0.34),  # evening peak
    ], default=0.17),
    fixed_charge=12.0,
    tax_rate=0.05,
)

bill = compute_bill(home, tariff, days=30, steps_per_hour=120,
                    breakpoints=home.switch_times(30 * 24))
print(bill)
```

**The rates in `example_tariffs()` are placeholders.** Replace them with the
numbers off your own bill — the rate table is usually on the second page. Then
tune `baseline_kw` until the model's monthly total matches your actual bill;
baseline is the hardest number to guess and the easiest to calibrate this way.

If you have interval data from a utility portal or an energy monitor, skip the
synthetic profile entirely and use `energy_from_samples(times_h, power_kw)`,
which handles irregular sample spacing.

## What the model shows

Running it on the example household (~1967 kWh/month, EV and electric
cooling):

**Cost is power × time, not power.** The kettle draws 2.2 kW, nearly as much
as the oven, but runs 15 minutes a day and lands near the bottom of the table.
The baseline draws 0.28 kW — a rounding error instantaneously — but it is the
only load multiplied by all 720 hours in the month, so it outranks the oven,
the dishwasher and the lighting combined:

```
load                      peak kW    kWh/mo   share      cost
EV charger                   7.20     864.0   43.9%   $138.24
heating / cooling            1.80     648.0   33.0%   $103.68
baseline (always on)         0.28     201.6   10.3%    $32.26
dishwasher                   1.80      81.0    4.1%    $12.96
oven                         2.40      72.0    3.7%    $11.52
kettle + coffee              2.20      16.5    0.8%     $2.64
```

Shaving 50 W of standby power saves 36 kWh/month — half of what the oven
contributes, from a load too small to notice, purely because it never switches
off.

**When you use it can matter as much as how much.** The same 864 kWh of EV
charging costs $81.65 or $269.89 a month depending only on start time. `∫P dt`
is unchanged; `∫r(t)P(t)dt` is not.

**Tariff choice is worth checking.** The identical load profile costs $342.99
flat, $350.50 time-of-use, or $508.88 tiered. The tiered plan is punishing
here because high consumption sits deep in the top block — but that ordering
flips for a smaller household, and swapping one line of the model tells you
which way it falls for yours.

## File map

| | |
|---|---|
| `riemann`, `trapezoid`, `simpson` | the quadrature rules, for comparison |
| `integrate` | the one to actually use — handles breakpoints |
| `Appliance`, `Household` | building a `P(t)` |
| `flat_rate`, `time_of_use`, `tiered`, `combine` | price functions `p(t, c)` |
| `compute_bill` | the main integral, marched forward carrying `E(t)` |
| `peak_demand` | sliding-window maximum |
| `energy_from_samples`, `power_from_meter_readings` | real meter data, both directions |
| `marginal_cost_of_load` | what one more appliance actually costs |
| `energy_by_source` | per-load breakdown |
