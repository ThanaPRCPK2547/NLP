---
name: Data Engineering Design System
colors:
  surface: '#121315'
  surface-dim: '#121315'
  surface-bright: '#38393b'
  surface-container-lowest: '#0d0e10'
  surface-container-low: '#1b1c1e'
  surface-container: '#1f2022'
  surface-container-high: '#292a2c'
  surface-container-highest: '#343537'
  on-surface: '#e3e2e5'
  on-surface-variant: '#bbc9cf'
  inverse-surface: '#e3e2e5'
  inverse-on-surface: '#303033'
  outline: '#859399'
  outline-variant: '#3c494e'
  surface-tint: '#4cd6ff'
  primary: '#a4e6ff'
  on-primary: '#003543'
  primary-container: '#00d1ff'
  on-primary-container: '#00566a'
  inverse-primary: '#00677f'
  secondary: '#c0c1ff'
  on-secondary: '#1000a9'
  secondary-container: '#3131c0'
  on-secondary-container: '#b0b2ff'
  tertiary: '#1afaa5'
  on-tertiary: '#003921'
  tertiary-container: '#00db8f'
  on-tertiary-container: '#005a38'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#b7eaff'
  primary-fixed-dim: '#4cd6ff'
  on-primary-fixed: '#001f28'
  on-primary-fixed-variant: '#004e60'
  secondary-fixed: '#e1e0ff'
  secondary-fixed-dim: '#c0c1ff'
  on-secondary-fixed: '#07006c'
  on-secondary-fixed-variant: '#2f2ebe'
  tertiary-fixed: '#50ffaf'
  tertiary-fixed-dim: '#00e293'
  on-tertiary-fixed: '#002111'
  on-tertiary-fixed-variant: '#005232'
  background: '#121315'
  on-background: '#e3e2e5'
  surface-variant: '#343537'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 40px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  code-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.5'
    letterSpacing: 0.05em
  label-caps:
    fontFamily: Space Grotesk
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1.0'
    letterSpacing: 0.1em
spacing:
  base: 8px
  xs: 4px
  sm: 12px
  md: 24px
  lg: 40px
  xl: 64px
  gutter: 24px
  margin: 32px
---

## Brand & Style
This design system is engineered for technical precision, targeting data architects and backend engineers who require high-density information environments. The brand personality is authoritative, "low-latency," and sophisticated. 

The aesthetic direction blends **Minimalism** with **Glassmorphism**, emphasizing structural integrity and technical clarity. By utilizing a dark-mode-first approach, the interface mimics a premium IDE or a high-end command center. Visual interest is generated through the contrast between deep charcoal surfaces and high-luminance accent colors, creating an atmosphere of advanced computation and reliable data orchestration.

## Colors
The palette is anchored by a deep charcoal base to reduce eye strain during prolonged technical sessions. 

- **Primary (Electric Blue):** Reserved exclusively for high-priority actions, primary buttons, and active focus states.
- **Secondary (Vibrant Indigo):** Used for secondary brand elements, data category indicators, and subtle visual accents.
- **Tertiary (Mint Green):** Strictly applied to success states, active data flows, and healthy system status indicators.
- **Neutral Scale:** A range of cool greys starting from `#0B0C0E` for the canvas, moving to `#1A1D23` for card surfaces, and `#2D3139` for borders. 
- **Data Visualization:** Use the primary and secondary colors as the start of a cold-to-warm gradient for pipeline health and throughput metrics.

## Typography
The system employs a dual-font strategy to balance readability with a technical aesthetic. **Space Grotesk** is utilized for headlines and status labels, providing a geometric, futuristic feel that aligns with data infrastructure themes. **Inter** is used for all body copy, chat messages, and UI controls to ensure maximum legibility at small sizes.

For code snippets and terminal outputs, use a monospaced font at 14px with a slightly reduced line height to accommodate dense blocks of JSON or SQL. All labels and metadata should be rendered in uppercase with increased letter spacing to differentiate them from interactive text.

## Layout & Spacing
The layout follows a **fluid grid** logic with strict 8px incremental spacing. The chatbot interface utilizes a sidebar-and-stage model:
- **Sidebar (320px):** Fixed width for pipeline lists and historical sessions.
- **Main Stage (Fluid):** The chat and data visualization area.

Content is organized into card-based modules. Use a 12-column grid for dashboard views, where widgets span 3, 6, or 12 columns. Internal card padding is standardized at 24px (`md`) to maintain a professional, airy feel despite the high-tech density.

## Elevation & Depth
Depth is achieved through **Glassmorphism** and tonal layering rather than traditional drop shadows.
1. **Background Layer:** The darkest charcoal (`#0B0C0E`).
2. **Surface Layer:** Cards use a semi-transparent fill (`rgba(26, 29, 35, 0.8)`) with a `20px` backdrop blur.
3. **Borders:** Every card and interactive element must have a 1px solid border (`#2D3139`). For active states, the border transitions to the Primary Electric Blue.
4. **Highlights:** Use a subtle top-down inner glow (0.5px white at 10% opacity) on cards to simulate a light source from above, emphasizing the "sharp" edge.

## Shapes
To reinforce the professional and high-tech nature of the system, all UI elements feature **sharp corners (0px radius)**. This applies to buttons, cards, input fields, and dropdowns. This "brutalist-lite" approach communicates precision and efficiency, mirroring the rigid structure of data schemas and code blocks.

## Components
- **Buttons:** Primary buttons are solid Electric Blue with black text. Secondary buttons are outlined (1px) in Indigo with Indigo text. High-tech "Ghost" buttons are used for utility actions.
- **Chat Bubbles:** Unlike consumer apps, chat bubbles are rectangular with sharp corners. The Bot's response uses the Indigo-tinted glass effect, while User messages are simple outlined boxes.
- **Data Chips:** Small, sharp-edged tags used for status (e.g., "SQL", "In Production", "Failed"). Success chips use a Mint Green border and glow.
- **Input Fields:** Bottom-border only or fully outlined with 1px charcoal. On focus, the border glows Electric Blue.
- **Code Blocks:** Deep black background (`#000000`) with syntax highlighting using the primary, secondary, and tertiary palette. Includes a "Copy" utility in the top-right corner.
- **Pipelines / Nodes:** For data flow diagrams, use 2px Electric Blue lines to connect sharp-edged node containers, representing the flow of information.
- **Icons:** Use thin-stroke (1.5pt) linear icons. Icons must be technical (e.g., `terminal` for console, `database` for sources, `activity` for logs).