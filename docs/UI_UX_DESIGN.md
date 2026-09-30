# Workspace redesign — 8 September 2026

## Research and design decisions

The old interface combined a long composition sidebar, advanced sampling controls, large warning blocks and repetitive candidate cards. The redesign organizes the page around a user's task: choose a material, enter property targets, then inspect candidate processes.

- **Progressive disclosure:** composition editing, equipment limits and algorithm settings sit in labeled expanders. Common tasks need fewer visible controls. This follows [NN/g's progressive-disclosure guidance](https://www.nngroup.com/articles/progressive-disclosure/); the principle is established rather than a new 2026 trend.
- **Analytical clarity:** inspect one candidate at a time. A horizontal dot/whisker chart shows each property as a percentage of its target, with a shaded tolerance band. Exact values remain available in a table. This applies [Carbon's dashboard guidance](https://carbondesignsystem.com/data-visualization/dashboards/) to reduce distractions and support interpretation.
- **Clear forms:** readable property names, persistent unit labels, explicit primary actions and actionable errors replace internal model terminology in the main task flow. [GOV.UK text-input guidance](https://design-system.service.gov.uk/components/text-input/) informs the labels and supporting descriptions. This is not a claim of complete GOV.UK-pattern compliance or a formal accessibility audit.

## Visual system

Solid white (#FFFFFF) background, restrained evergreen accent, dark neutral text, thin borders and a consistent type scale. The main background was updated from warm off-white following user feedback. Pure white matches the global background in [IBM Carbon's White theme](https://carbondesignsystem.com/elements/color/overview/); it is an established choice, not a universally superior shade for every interface. Muted input surfaces and thin borders retain visual separation. No gradients, glows, remote font downloads or custom theme switcher. Streamlit's native controls retain keyboard semantics, while visible focus outlines support navigation. The interface has no sidebar drawer.

Desktop layout places targets beside results. At narrower widths these areas stack, retaining paired inputs where practical. The empty state explains what will appear after a search instead of displaying invented results. Research limitations remain one clearly labeled disclosure, with visible match/no-match feedback beside actual results.

## Preserved behavior

The underlying model and screening engine are unchanged. Search controls, reproducibility, target tolerances, candidate inspection, report exports, invalid-input rejection and result invalidation remain available. Unsupported material families show their current status rather than a prediction workflow. Forward prediction and the material library have their own workspace navigation choices.

## Verification

The expanded suite passes 20 tests, including six Streamlit interaction checks for inverse search, changing targets, candidate selection, invalid targets, unsupported families/library navigation and forward prediction (library and unsupported-family behavior are separate cases). Core/model/data tests remain unchanged. Browser checks inspect the compact layout, desktop two-column layout and candidate results. Source changes are limited to the presentation layer, theme, UI tests and this documentation.

This is a researched implementation, not a user-tested usability study. Next evaluation should observe materials engineers completing a target search, inspecting a target miss, changing equipment limits and exporting a report; record completion rate, errors and time to a correct interpretation.
