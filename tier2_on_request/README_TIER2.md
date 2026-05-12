# Tier 2 — On-Request Release Preparation

This directory holds preparation materials for the **on-request release** of the four representative cam1 raw frames used in Fig. 6 of the IEEE TIM v2 paper. Estimated total size when released: ~280 MB (4 frames × ~70 MB each).

Tier 2 frames are **not** publicly deposited. The corresponding author at KICT fulfils requests from researchers on a case-by-case basis after confirming the requestor's affiliation and intended use.

---

## Contents of This Directory

| File | Purpose | Approx. Size |
|------|---------|--------------|
| `README_TIER2.md` | This document — release workflow and anonymization checklist | < 5 KB |
| `metadata.json` | Dataset descriptor (technical metadata, citation block) bundled with each release | < 1 KB |
| `ANONYMIZATION_LOG_template.txt` | Per-frame checklist completed before each release | < 5 KB |
| `raw_frames/` | Placeholder; populated only when fulfilling an approved request | — |

---

## Frame List (Sources for Fig. 6 panels)

| Filename | Condition | Fig. 6 panel | Source path |
|----------|-----------|--------------|-------------|
| `cam1_v60_iso200_d25.png` | 60 km/h, ISO 200, 2.5 m | (a) Best-case | `03_src/data/raw/crack/cam1/` |
| `cam1_v80_iso200_d45.png` | 80 km/h, ISO 200, 4.5 m | (b) Typical | same |
| `cam1_v60_iso200_d65.png` | 60 km/h, ISO 200, 6.5 m | (c) Boundary | same |
| `cam1_v60_iso1600_d25.png` | 60 km/h, ISO 1600, 2.5 m | (d) Saturated | same |

Frames are full-resolution cam1 raw images (8K nominal) at native bit depth.

---

## Anonymization Checklist (MANDATORY before each release)

Each of the four frames MUST be visually and metadata-cleared before being delivered. Record the result in a copy of `ANONYMIZATION_LOG_template.txt`.

- [ ] **(1) Facility identification** — No tunnel signage, facility logos, exit numbers, or named landmarks visible in frame. The cam1 field of view captures only the test chart panel mounted on the tunnel wall, so residual risk is low; verify by visual inspection at 100 % zoom.
- [ ] **(2) EXIF / metadata stripping** — Remove all camera EXIF tags (timestamps, GPS coordinates, serial numbers, owner/copyright fields). Use exiftool:
      ```bash
      exiftool -all= cam1_v60_iso200_d25.png
      ```
      Verify with `exiftool cam1_v60_iso200_d25.png` afterward (expect minimal metadata).
- [ ] **(3) People / vehicles / license plates** — Confirm absence of any third-party identifiers (faces, plates, identifying clothing). Fig. 6 frames were captured during dedicated test runs with no third parties present, but verify by inspection.
- [ ] **(4) Chart-only visibility** — Confirm all four frames show only the chart-bearing region — no incidental capture of equipment serial numbers, operator name tags, or non-public infrastructure.

A signed copy of the completed checklist is bundled with each release to document compliance.

---

## On-Request Release Procedure

1. **Receive request** at the corresponding-author email (see paper for address).
2. **Verify** requestor affiliation and stated purpose (academic / reproducibility / methodological cross-check).
3. **Run the anonymization checklist** above on each of the four frames; record results in `ANONYMIZATION_LOG.txt` (copy of the template in this directory).
4. **Bundle** `raw_frames/*.png` + completed `ANONYMIZATION_LOG.txt` + `metadata.json` into a zip.
5. **Deliver** via institutional file-transfer service (e.g., KICT secure exchange) or a shared cloud link.
6. **Record** the release in an internal log (date, requestor identity, intended use).

---

## Citation Bundled With Each Release

```
Lee, C., Kim, D., Kim, D., An, J. (2026). A POD-Based Image Quality Acceptance
Framework for Crack Detection in High-Speed Mobile Tunnel Inspection.
IEEE Transactions on Instrumentation and Measurement. [DOI: TBD upon acceptance]

Code and processed data: https://github.com/chuligod22-ux/tunnel-pod-acceptance
```

License upon release: **Creative Commons Attribution 4.0 International (CC-BY 4.0)** — matches the paper's Data Availability Statement.

---

## Notes

- **Tier 3 raw archive** (all 50 cam1 conditions, ~3.5 GB) is handled under the same on-request workflow but is a larger bundle. The same anonymization checklist applies.
- The 1,500-frame acquisition pool (~100 GB) remains internal at KICT.
- No public deposit channel (Zenodo / figshare / IEEE DataPort) is used for Tier 2 in the current submission; this decision is documented for reproducibility transparency.
