# Policy Trace Lab brand guide

## Name and description

- Name: **Policy Trace Lab**. Write it in full, with these capitals. Short forms in prose: "the lab". The command is `gsp lab`, the package is `guided_streaming_policy`; neither changes.
- Tagline: **Validate robot data. Compare policies. Replay regressions.**
- One line: Policy Trace Lab is an open-source workbench that validates robot data, compares policies in paired simulation runs, and replays the first regression.

## Idea

Traceability. A policy run is a line through time. When two runs start from the same state and one goes wrong, the lines split, and a marker sits on the split. The mark draws exactly that: two lines share a path and part at a red ring. The wordmark repeats the marker as the dot of the "i".

## Voice

Plain and technical. Say what a command reads, writes and returns. No superlatives, no claims about which policy is better in general, no physical-safety claims. Proxies are called proxies. Fixture data is called synthetic.

## Files

| File | Use |
| --- | --- |
| `logo-mark.svg`, `logo-mark-dark.svg` | The mark alone, for avatars and small spaces. Light and dark variants. |
| `logo-wordmark.svg`, `logo-wordmark-dark.svg` | Mark plus name, for headers. |
| `readme-banner.svg`, `readme-banner-dark.svg` | README hero, served through `<picture>`. |
| `pipeline.svg`, `pipeline-dark.svg` | Pipeline diagram: simulator, episode trace, data checks, paired evaluation, replay. |
| `favicon.svg` | Report favicon; switches colour with the system theme. |
| `social-preview.png` | 1280 x 640 GitHub social preview, uploaded by the maintainer. |
| `tokens.json` | Single source of truth for colour, type and spacing. |

The letters of the wordmark are drawn as strokes, so the wordmark does not depend on an installed font. The text inside `pipeline*.svg` uses system fonts and may differ slightly between platforms.

## Logo use

- Clear space: keep at least the height of the ring (about one eighth of the mark's height) free on every side.
- Minimum size: mark 16 px high, wordmark 28 px high.
- Use the light files on light backgrounds and the dark files on dark backgrounds. Do not place the light mark on a photo or a busy background.

## Colour

Defined in `tokens.json`; light and dark sets share names.

- **Ink** is the line colour and the text colour. **Paper** is the page. **Surface** is cards and tables.
- **Baseline** (blue) and **candidate** (purple) identify the two runs in plots.
- **Pass** (green), **fail** (red) and **unknown** (slate) are semantic. They are never shown by colour alone: each also carries an icon (check, cross, question mark) and a word (Pass, Fail, Unknown).
- Every text and background pair listed in `contrast_pairs` meets WCAG AA (4.5:1); a test checks this.

## Type

- Display (report title, section headings): the system serif stack in `tokens.json` (Iowan Old Style, Palatino, Georgia).
- Body: the system sans stack. Use tabular numerals for numbers in tables.
- Mono: only for hashes, paths and commands.
- No web fonts and no CDN: assets must work offline.

## Never

- Do not recolour the marker, redraw the mark, add gradients, shadows or outlines to it, or rotate it.
- Do not use the fail red for anything that is not a failure or the marker.
- Do not copy the style of another project's logo or reuse a third-party mark.
- Do not show pass, fail or unknown by colour alone.
- Do not present synthetic fixture results as a benchmark.
