/**
 * Tiny conditional-class joiner. Avoids pulling in a full `clsx`
 * dependency for a couple of call sites.
 */
export function cn(
  ...classes: Array<string | false | null | undefined>
): string {
  return classes.filter(Boolean).join(" ");
}
