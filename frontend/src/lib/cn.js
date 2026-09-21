/** Join class names, dropping falsy values. Keeps conditional classes readable. */
export function cn(...parts) {
  return parts.filter(Boolean).join(' ')
}
