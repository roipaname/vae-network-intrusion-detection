// Chart libraries need literal color values (SVG attributes don't resolve
// CSS custom properties), so these mirror the palette defined in index.css.
export const COLORS = {
  bg: '#f5f6f2',
  primaryDark: '#15191a',
  secondaryDark: '#252b2b',
  accent: '#8fd3b6',
  accentSoft: '#b8e5d0',
  success: '#79c9a4',
  warning: '#d6b56d',
  threat: '#d76b68',
  border: '#dde2de',
  textMuted: 'rgba(21, 25, 26, 0.45)',
  textSecondary: 'rgba(21, 25, 26, 0.65)',
};

export const BASELINE_COLOR = COLORS.secondaryDark;
export const AUGMENTED_COLOR = COLORS.accent;
