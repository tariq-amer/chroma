const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
  Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = require("docx");

const INK = "0B0B0B", INK2 = "52514E", ACCENT = "1C5CAB", RULE = "D8D7D2";
const BLUE = "2A78D6", ORANGE = "EB6834", TINT = "EEF4FD";
const BODY = "Calibri", MATH = "Cambria", MONO = "Consolas";
const CONTENT_W = 9360; // Letter, 1" margins

const NONE = { style: BorderStyle.NONE, size: 0, color: "auto" };
const noBorders = { top: NONE, bottom: NONE, left: NONE, right: NONE };

// ---------- small builders ---------------------------------------------------
const p = (text, o = {}) => new Paragraph({
  alignment: o.align,
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: o.line ?? 264 },
  indent: o.indent,
  border: o.border,
  children: [new TextRun({
    text, font: o.font ?? BODY, size: o.size ?? 22, bold: o.bold,
    italics: o.italics, color: o.color ?? INK,
  })],
});

// rich paragraph: array of [text, {opts}]
const rp = (runs, o = {}) => new Paragraph({
  alignment: o.align,
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: o.line ?? 264 },
  indent: o.indent,
  children: runs.map(([t, ro = {}]) => new TextRun({
    text: t, font: ro.font ?? o.font ?? BODY, size: ro.size ?? o.size ?? 22,
    bold: ro.bold, italics: ro.italics, color: ro.color ?? o.color ?? INK,
  })),
});

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 160 },
  keepNext: true, keepLines: true,
  children: [new TextRun({ text, font: BODY, size: 30, bold: true, color: ACCENT })],
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 260, after: 120 },
  keepNext: true, keepLines: true,
  children: [new TextRun({ text, font: BODY, size: 24, bold: true, color: INK })],
});

// a display-maths line, indented and set in a serif face
const math = (text, o = {}) => p(text, {
  font: MATH, size: o.size ?? 24, indent: { left: 480 },
  before: o.before ?? 100, after: o.after ?? 140, color: o.color ?? INK, bold: o.bold,
});

const bullet = (text) => new Paragraph({
  numbering: { reference: "dots", level: 0 },
  spacing: { after: 90, line: 264 },
  children: [new TextRun({ text, font: BODY, size: 22, color: INK })],
});

const step = (text) => new Paragraph({
  numbering: { reference: "steps", level: 0 },
  spacing: { after: 90, line: 264 },
  children: [new TextRun({ text, font: BODY, size: 22, color: INK })],
});

const rule = () => new Paragraph({
  spacing: { before: 60, after: 220 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 1 } },
  children: [new TextRun({ text: "", size: 2 })],
});

const figure = (file, w, hgt, caption) => ([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    keepNext: true,
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(file),
      transformation: { width: w, height: hgt },
    })],
  }),
  rp([[caption, { size: 18, italics: true, color: INK2 }]],
     { align: AlignmentType.CENTER, after: 180 }),
]);

const cell = (runs, width, o = {}) => new TableCell({
  width: { size: width, type: WidthType.DXA },
  shading: o.fill ? { type: ShadingType.CLEAR, color: "auto", fill: o.fill } : undefined,
  margins: { top: 90, bottom: 90, left: 130, right: 130 },
  verticalAlign: "center",
  children: [new Paragraph({
    alignment: o.align,
    spacing: { before: 0, after: 0, line: 240 },
    children: runs.map(([t, ro = {}]) => new TextRun({
      text: t, font: ro.font ?? o.font ?? BODY, size: ro.size ?? o.size ?? 20,
      bold: ro.bold ?? o.bold, italics: ro.italics, color: ro.color ?? o.color ?? INK,
    })),
  })],
});

const table = (widths, rows) => new Table({
  columnWidths: widths,
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  borders: {
    top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    left: NONE, right: NONE,
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "EDECE8" },
    insideVertical: NONE,
  },
  rows,
});

// a tinted call-out box built from a one-cell table
const callout = (children) => new Table({
  columnWidths: [CONTENT_W],
  width: { size: CONTENT_W, type: WidthType.DXA },
  borders: {
    top: NONE, bottom: NONE, right: NONE,
    left: { style: BorderStyle.SINGLE, size: 18, color: BLUE },
    insideHorizontal: NONE, insideVertical: NONE,
  },
  rows: [new TableRow({
    children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, color: "auto", fill: TINT },
      margins: { top: 200, bottom: 200, left: 280, right: 240 },
      children,
    })],
  })],
});

// ---------- content ----------------------------------------------------------
const C = { align: AlignmentType.CENTER };
const R = { align: AlignmentType.RIGHT };

const verifyRows = [
  new TableRow({
    tableHeader: true, cantSplit: true,
    children: [
      cell([["a", { italics: true }]], 1160, { fill: "F2F1ED", bold: true, align: AlignmentType.CENTER }),
      cell([["a³"]], 1640, { fill: "F2F1ED", bold: true, align: AlignmentType.RIGHT }),
      cell([["− 6a²"]], 1640, { fill: "F2F1ED", bold: true, align: AlignmentType.RIGHT }),
      cell([["+ 11a"]], 1640, { fill: "F2F1ED", bold: true, align: AlignmentType.RIGHT }),
      cell([["Total"]], 1640, { fill: "F2F1ED", bold: true, align: AlignmentType.RIGHT }),
      cell([["= 6 ?"]], 1640, { fill: "F2F1ED", bold: true, align: AlignmentType.CENTER }),
    ],
  }),
  ...[
    ["1", "1", "− 6", "+ 11", "6"],
    ["2", "8", "− 24", "+ 22", "6"],
    ["3", "27", "− 54", "+ 33", "6"],
  ].map(([a, c1, c2, c3, tot]) => new TableRow({
    cantSplit: true,
    children: [
      cell([[a, { bold: true, font: MATH, size: 22 }]], 1160, { align: AlignmentType.CENTER }),
      cell([[c1]], 1640, { align: AlignmentType.RIGHT }),
      cell([[c2]], 1640, { align: AlignmentType.RIGHT }),
      cell([[c3]], 1640, { align: AlignmentType.RIGHT }),
      cell([[tot, { bold: true }]], 1640, { align: AlignmentType.RIGHT }),
      cell([["✓  yes", { color: "0CA30C", bold: true }]], 1640, { align: AlignmentType.CENTER }),
    ],
  })),
];

const featureRows = [
  new TableRow({
    tableHeader: true, cantSplit: true,
    children: [
      cell([["Property"]], 2600, { fill: "F2F1ED", bold: true }),
      cell([["f(a) = a³ − 6a² + 11a − 6", { size: 18 }]], 3380, { fill: "F2F1ED", bold: true, color: BLUE }),
      cell([["g(a) = a³ − 6a² + 11a", { size: 18 }]], 3380, { fill: "F2F1ED", bold: true, color: ORANGE }),
    ],
  }),
  ...[
    ["Factored form", "(a − 1)(a − 2)(a − 3)", "a(a² − 6a + 11)"],
    ["Real zeros", "a = 1, 2, 3", "a = 0 only"],
    ["Non-real zeros", "none", "a = 3 ± i√2"],
    ["Value at a = 0", "− 6", "0"],
    ["Local maximum", "(1.4226, 0.3849)", "(1.4226, 6.3849)"],
    ["Local minimum", "(2.5774, − 0.3849)", "(2.5774, 5.6151)"],
    ["Point of inflection", "(2, 0)", "(2, 6)"],
    ["Symmetry centre", "(2, 0)", "(2, 6)"],
  ].map(([k, v1, v2]) => new TableRow({
    cantSplit: true,
    children: [
      cell([[k]], 2600),
      cell([[v1, { font: MATH, size: 21 }]], 3380),
      cell([[v2, { font: MATH, size: 21 }]], 3380),
    ],
  })),
];

const codeLines = fs.readFileSync("plot_cubics.py", "utf8").replace(/\s+$/, "").split("\n");
const codeParas = codeLines.map((line) => new Paragraph({
  spacing: { before: 0, after: 0, line: 210 },
  indent: { left: 200 },
  children: [new TextRun({
    text: line.length ? line : " ",
    font: MONO, size: 15, color: line.trimStart().startsWith("#") ? INK2 : INK,
  })],
}));

const doc = new Document({
  creator: "Claude Code",
  title: "Cubic Polynomials: Two Plots and One Equation",
  description: "Plots of a³ − 6a² + 11a − 6 and a³ − 6a² + 11a, with the solution of a³ − 6a² + 11a = 6",
  styles: {
    default: {
      document: { run: { font: BODY, size: 22, color: INK } },
    },
  },
  numbering: {
    config: [
      {
        reference: "dots",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } },
        }],
      },
      {
        reference: "steps",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: {
            paragraph: { indent: { left: 460, hanging: 320 } },
            run: { bold: true, color: ACCENT },
          },
        }],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: {
          top: convertInchesToTwip(1), bottom: convertInchesToTwip(1),
          left: convertInchesToTwip(1), right: convertInchesToTwip(1),
        },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES],
            font: BODY, size: 16, color: INK2,
          })],
        })],
      }),
    },
    children: [
      // ---- title block ----
      rp([["Cubic Polynomials", { size: 44, bold: true, color: INK }]], { after: 60 }),
      rp([["Two plots and one equation", { size: 26, color: INK2 }]], { after: 100 }),
      rule(),
      rp([
        ["f(a) = a³ − 6a² + 11a − 6", { font: MATH, size: 22, color: BLUE, bold: true }],
        ["      ", {}],
        ["g(a) = a³ − 6a² + 11a", { font: MATH, size: 22, color: ORANGE, bold: true }],
        ["      ", {}],
        ["solve  a³ − 6a² + 11a = 6", { font: MATH, size: 22, color: INK2 }],
      ], { after: 320 }),

      // ---- the answer, up front ----
      callout([
        rp([["The answer", { size: 20, bold: true, color: ACCENT }]], { after: 100 }),
        rp([
          ["a = 1,  a = 2,  a = 3", { font: MATH, size: 36, bold: true, color: INK }],
        ], { after: 100 }),
        rp([
          ["The cubic a³ − 6a² + 11a − 6 factors completely over the integers as ", {}],
          ["(a − 1)(a − 2)(a − 3)", { font: MATH, size: 22, bold: true }],
          [", so the equation has exactly three real solutions and no others — real or complex.", {}],
        ], { after: 0 }),
      ]),

      h1("1.  The two functions"),
      p("This document plots two cubics and then solves an equation that connects them. Writing them as f and g:"),
      math("f(a) = a³ − 6a² + 11a − 6"),
      math("g(a) = a³ − 6a² + 11a"),
      p("They differ by a constant, and that constant is exactly the number on the right-hand side of the equation:"),
      math("g(a) = f(a) + 6"),
      p("So the second curve is the first one lifted six units, and asking where g(a) = 6 is the same as asking where f(a) = 0. That equivalence is the whole solution, and the plots show it directly."),
      p("A tidier way to see the shape of both is to shift the variable to the centre of symmetry. Putting u = a − 2 removes the quadratic term:"),
      math("f(a) = u³ − u          g(a) = u³ − u + 6,     where u = a − 2"),
      p("Both curves are therefore the odd function u³ − u, rotated about (2, 0) and (2, 6) respectively — which is why each has a hump and a dip of the same size, 2√3⁄9 ≈ 0.3849, on either side of its centre."),
      p("The plots below use the window 0.3 ≤ a ≤ 3.7. That is deliberately tight: the turning points rise and fall by only about 0.385, so on a wider window the interesting part of the curve flattens into the axis."),

      h1("2.  Plot of f(a) = a³ − 6a² + 11a − 6"),
      p("The curve rises, dips, and rises again, crossing the horizontal axis three times. Each crossing is a simple root, so the curve passes straight through rather than touching and turning back."),
      ...figure("figs/fig1_f.png", 576, 272,
        "Figure 1 — f(a) = a³ − 6a² + 11a − 6 on 0.3 ≤ a ≤ 3.7. The three marked points are the roots a = 1, 2 and 3."),
      p("Reading the plot: f is negative to the left of a = 1, positive between 1 and 2, negative between 2 and 3, and positive after 3. The sign changes at each root confirm the roots are simple, and there are no further crossings — a cubic has at most three."),

      h1("3.  Plot of g(a) = a³ − 6a² + 11a"),
      p("The same shape, six units higher. Because the whole curve has been lifted clear of the axis in this window, its only real zero (at a = 0) lies off to the left of the plot; the other two zeros of g are the complex pair 3 ± i√2, so the curve never returns to the axis on the right."),
      ...figure("figs/fig2_g.png", 576, 272,
        "Figure 2 — g(a) = a³ − 6a² + 11a with the line y = 6. The two meet at a = 1, 2 and 3."),
      p("The dashed line is y = 6, the right-hand side of the equation to be solved. The curve meets it three times, and those three meeting points are the solutions."),

      h1("4.  The two plots together"),
      p("Overlaying the curves makes the relationship explicit: the vertical gap between them is a constant 6 everywhere, and the points where g meets y = 6 sit directly above the points where f meets y = 0."),
      ...figure("figs/fig3_both.png", 576, 321,
        "Figure 3 — both cubics on one axis. The dotted verticals link each solution of g(a) = 6 to the matching root of f(a) = 0."),

      h1("5.  Solving a³ − 6a² + 11a = 6"),
      h2("Method 1 — rational roots, then factor"),
      step("Bring everything to one side. Subtracting 6 turns the equation into a root-finding problem for f:"),
      math("a³ − 6a² + 11a − 6 = 0"),
      step("Look for a rational root. Any rational root p⁄q of a polynomial with leading coefficient 1 and constant term −6 must be an integer dividing 6, so the only candidates are ±1, ±2, ±3, ±6. Testing the smallest:"),
      math("f(1) = 1 − 6 + 11 − 6 = 0        so (a − 1) is a factor."),
      step("Divide it out. Synthetic division of 1, −6, 11, −6 by a = 1 gives the coefficients 1, −5, 6 with remainder 0:"),
      math("a³ − 6a² + 11a − 6 = (a − 1)(a² − 5a + 6)"),
      step("Factor the quadratic. Two numbers multiplying to 6 and adding to 5 are 2 and 3:"),
      math("a² − 5a + 6 = (a − 2)(a − 3)"),
      step("Read off the solutions. A product is zero only when one of its factors is:"),
      math("(a − 1)(a − 2)(a − 3) = 0    ⟹    a = 1,  a = 2,  a = 3", { bold: true }),

      h2("Method 2 — shift the variable"),
      p("Substituting a = u + 2 (the centre of symmetry found in §1) kills the quadratic term and leaves a cubic that factors on sight:"),
      math("u³ − u = 0    ⟹    u(u − 1)(u + 1) = 0    ⟹    u = −1, 0, 1"),
      p("Undoing the substitution with a = u + 2 gives a = 1, 2, 3 — the same three values, reached without any trial roots."),

      h2("Checks"),
      bullet("Vieta's formulas. For a³ − 6a² + 11a − 6 the roots must sum to 6, sum in pairs to 11, and multiply to 6. Indeed 1 + 2 + 3 = 6, (1·2) + (1·3) + (2·3) = 11, and 1·2·3 = 6."),
      bullet("Discriminant. In the shifted form u³ + pu + q with p = −1 and q = 0, the discriminant is −4p³ − 27q² = 4 > 0, which guarantees three distinct real roots — matching the three crossings in Figure 1."),
      bullet("Completeness. A cubic has exactly three roots counted with multiplicity, so a = 1, 2, 3 is the full solution set; there is no fourth answer and no complex one."),

      h2("Substituting back"),
      p("Each candidate put into the original left-hand side, a³ − 6a² + 11a:"),
      table([1160, 1640, 1640, 1640, 1640, 1640], verifyRows),
      rp([["Table 1 — all three values return exactly 6, so all three satisfy the original equation.",
           { size: 18, italics: true, color: INK2 }]], { before: 120, after: 240 }),

      h1("6.  Reference: features of the two curves"),
      p("Turning points come from the derivative 3a² − 12a + 11 = 0, giving a = 2 ± √3⁄3 ≈ 1.4226 and 2.5774. The inflection point is where the second derivative 6a − 12 vanishes, at a = 2."),
      table([2600, 3380, 3380], featureRows),
      rp([["Table 2 — the two curves are identical in shape; every y-value differs by exactly 6.",
           { size: 18, italics: true, color: INK2 }]], { before: 120, after: 240 }),

      rule(),
      h1("Appendix — code used to produce the plots"),
      p("Python 3 with NumPy and Matplotlib. The script draws all three figures at 200 dpi and re-derives every number quoted in this document rather than restating it, so the prose and the plots cannot drift apart."),
      ...codeParas,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Cubic-Polynomials-Plots-and-Solution.docx", buf);
  console.log("wrote Cubic-Polynomials-Plots-and-Solution.docx", buf.length, "bytes");
});
