# Course Project LaTeX Template

A minimal, opinionated template for course-project writeups. Works on Overleaf and locally.

---

## Quick Start

1. **Copy the template** (clone or download).
2. **Edit metadata** in `main.tex` (title, authors, date).
3. **Write content** by adding `.tex` files under `sections/` and `\input{...}` them from `main.tex`.
4. **Add figures** to the `figure/` folder and include them with `\includegraphics`.
5. **Manage references** in `refs.bib` and cite with `\cite{...}`.
6. **Compile** on Overleaf or locally (see below).

---

## Compile Options

- **Online:** Overleaf — upload the repo and set `main.tex` as the root document.
- **Local:** VS Code with LaTeX or any editor.

---

## Project Structure

```
.
├── main.tex                 # Entry point: metadata, packages, inputs, bibliography
├── preamble_packages.tex    # Package imports (comment out what you don't need)
├── preamble_symbols.tex     # Common symbols and math operators
├── shortcuts.tex            # Project-specific commands/macros
├── refs.bib                 # BibTeX database
├── sections/                # Source for individual sections
│   ├── intro.tex
│   ├── related_work.tex
│   └── ...
├── figure/                  # Images and plots
│   ├── system_diagram.pdf
│   └── ...
└── .gitignore               # Files to exclude from version control
```


---

## How to Use Each File

- **`main.tex`**
  - Sets the document class, title, authors, packages, and bibliography.
  - Includes content, e.g.:
    ```tex
    \input{preamble_packages}
    \input{preamble_symbols}
    \input{shortcuts}

    \title{Project Title}
    \author{Alice Smith \and Bob Jones}
    \date{\today}

    \begin{document}
    \maketitle

    \input{sections/intro}
    \input{sections/related_work}
    \input{sections/method}
    \input{sections/experiments}
    \input{sections/conclusion}

    \bibliographystyle{abbrvnat}
    \bibliography{refs}
    \end{document}
    ```

- **`preamble_packages.tex`**
  - Curated package list. Comment out lines you don’t need to keep the build lean.

- **`preamble_symbols.tex`**
  - Common math symbols/operators (e.g., `\R`, `\E`, `\argmin`). Extend as needed.

- **`shortcuts.tex`**
  - Project-specific macros:
    ```tex
    \newcommand{\method}{\textsc{OurMethod}\xspace}
    ```

- **`refs.bib`**
  - Add BibTeX entries and cite them:
    ```tex
    As shown by \cite{mnih2015dqn}, ...
    ```

- **`sections/`**
  - Split the paper into maintainable pieces:
    - `intro.tex`, `related_work.tex`, `method.tex`, `experiments.tex`, `conclusion.tex`, etc.
  - Include with `\input{sections/<name>}` (no `.tex` extension required).

- **`figure/`**
  - Store figures/plots and include them:
    ```tex
    \begin{figure}[t]
      \centering
      \includegraphics[width=\linewidth]{figure/system_diagram}
      \caption{System overview.}
      \label{fig:system}
    \end{figure}
    ```

- **`.gitignore`**
  - Keeps build artifacts and OS/editor files out of Git (e.g., `*.aux`, `*.log`, `*.out`, `*.synctex.gz`).

---

## Best Practices

- **Labels & refs:** `\label{sec:method}` then `\S\ref{sec:method}`; figures with `\ref{fig:system}`; equations with `\eqref{eq:loss}`.
- **Tables/Figures:** Prefer vector formats (`.pdf`, `.eps`) for diagrams; use high-resolution `.png` for raster images.
- **Keep it lean:** Only load packages you need; define macros once in `shortcuts.tex`.

---

## Troubleshooting

- **Missing references/citations:** Run LaTeX → BibTeX → LaTeX → LaTeX, or just use `latexmk -pdf`.
- **Undefined control sequence:** A macro may live in `shortcuts.tex` or a package is missing—ensure it’s included.

---

Happy writing!
