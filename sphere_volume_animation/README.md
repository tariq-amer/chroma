# Volume of a Sphere with Calculus: Animation

`sphere_volume_animation.py` renders a 26-second animation (1280×720,
24 fps). It shows how the disk method gives the volume of a sphere:

$$V=\int_{-R}^{R}\pi\,(R^2-x^2)\,dx=\tfrac{4}{3}\pi R^3$$

| Step | What you see |
|------|--------------|
| 1. Revolve | The half-circle $y=\sqrt{R^2-x^2}$ spins around the x-axis and sweeps out a sphere. |
| 2. Slice | A slice at position $x$ is a disk. Pythagoras gives $r=\sqrt{R^2-x^2}$, so the slice's area is $A(x)=\pi(R^2-x^2)$. |
| 3. Sum | The sphere is built from $n = 2, 3, \dots, 48$ disks, and each disk matches one bar under $A(x)$. The Riemann sum gets closer to $4.18879\,R^3$ as $n$ grows. |
| 4. Limit | As $\Delta x\to 0$ the sum becomes an integral. The solid fills in while the area under $A(x)$ is shaded. |
| 5. Evaluate | The integral is worked out line by line, ending at $V=\frac{4}{3}\pi R^3$. |

## Run it

```bash
pip install numpy matplotlib        # ffmpeg is needed for .mp4 output

python sphere_volume_animation.py                        # writes sphere_volume.mp4
python sphere_volume_animation.py -o sphere_volume.gif   # animated GIF
python sphere_volume_animation.py --show                 # play in a window
python sphere_volume_animation.py --frame 300 -o f.png   # save one still frame
```

Rendering the full video takes about 2–3 minutes.
