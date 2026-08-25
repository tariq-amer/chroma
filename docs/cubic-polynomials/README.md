# Cubic polynomials — plots and solution

Generates `Cubic-Polynomials-Plots-and-Solution.docx`: plots of

- `f(a) = a^3 - 6a^2 + 11a - 6`
- `g(a) = a^3 - 6a^2 + 11a`

and a worked solution of `a^3 - 6a^2 + 11a = 6`, whose roots are `a = 1, 2, 3`.

## Rebuilding

```sh
pip install matplotlib numpy
npm install docx
python3 plot_cubics.py   # writes figs/*.png at 200 dpi
node build_doc.js        # writes the .docx
```

`plot_cubics.py` re-derives every number quoted in the document (roots, turning
points, inflection) rather than restating it, so the prose and the plots cannot
drift apart.
