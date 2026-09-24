/**
 * One colour per entity type, shared by the graph canvas, its legend and the
 * entity lists so a type reads the same everywhere. Types match ENTITY_TYPES in
 * backend/app/graph/extractor.py.
 *
 * Cytoscape draws on a canvas and cannot read CSS variables, hence plain hex.
 * The values are tuned for the app's dark surface (#121826).
 */
export const TYPE_COLORS = {
  PERSON: '#f472b6',
  ORGANIZATION: '#818cf8',
  LOCATION: '#34d399',
  PRODUCT: '#fbbf24',
  TECHNOLOGY: '#22d3ee',
  CONCEPT: '#a78bfa',
  EVENT: '#fb923c',
  METRIC: '#f87171',
  DOCUMENT: '#60a5fa',
  OTHER: '#7c89a6',
}

export function typeColor(type) {
  return TYPE_COLORS[type] || TYPE_COLORS.OTHER
}

/** "ORGANIZATION" -> "Organization" for labels. */
export function typeLabel(type) {
  const value = type || 'OTHER'
  return value.charAt(0) + value.slice(1).toLowerCase()
}

/** "APPOINTED_BY" -> "appointed by" for relation labels. */
export function predicateLabel(predicate) {
  return (predicate || 'related to').replace(/_/g, ' ').toLowerCase()
}
