/**
 * Light/dark theming.
 *
 * Every colour in the app is a `--color-*` custom property declared in
 * index.css, so a theme is just a different set of values for those properties.
 * Switching means setting one attribute on <html>; nothing re-renders, and the
 * browser repaints with the new palette.
 *
 * Dark is the default and stays the default. A stored preference is the only
 * thing that produces light, so the system's own colour-scheme setting is
 * deliberately ignored: this app was designed dark and should open dark.
 */

const STORAGE_KEY = 'sure-theme'
export const DEFAULT_THEME = 'dark'

/** Fired on `window` after the attribute changes, for canvases that paint
 *  their own pixels and cannot inherit CSS. */
export const THEME_EVENT = 'sure:themechange'

export function readTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored === 'light' || stored === 'dark' ? stored : DEFAULT_THEME
  } catch {
    // Private windows and blocked site data both throw here.
    return DEFAULT_THEME
  }
}

export function applyTheme(theme) {
  const next = theme === 'light' ? 'light' : 'dark'
  document.documentElement.setAttribute('data-theme', next)
  try {
    localStorage.setItem(STORAGE_KEY, next)
  } catch {
    // Not being able to remember the choice is not a reason to refuse it.
  }
  window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: next }))
  return next
}

/**
 * Read one of the theme's colour tokens as a real colour string.
 *
 * For anything drawn outside the DOM -- the Cytoscape canvas, a PNG export --
 * where a class name means nothing and the value has to be resolved by hand.
 */
export function cssColor(token, fallback = '#000000') {
  try {
    const value = getComputedStyle(document.documentElement)
      .getPropertyValue(`--color-${token}`)
      .trim()
    return value || fallback
  } catch {
    return fallback
  }
}
